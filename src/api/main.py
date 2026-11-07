"""
RepoTriage REST API Service (Stage 7)
=====================================
FastAPI backend exposing:
- GET /api/health
- GET /api/repositories
- POST /api/analyze (Issue classification + Hybrid retrieval + Ollama synthesis + DeBERTa NLI)
- POST /api/search (Similar issues / doc search)
- GET /api/analysis/{trace_id} (Retrieve saved analysis trace)
- GET /api/evaluation (Retrieval ablation + Guardrail evaluation + Benchmarks)

Concurrency Control:
- Global lock for /api/analyze returning HTTP 429 when pipeline is already executing.
- Safe trace retrieval strictly enforcing ^[A-Za-z0-9_-]+$ with relative paths.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import src.config
from src.pipeline import RepoTriagePipeline
from src.retrieval.index import HybridRetriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "processed" / "repotriage.db"
TRACES_DIR = PROJECT_ROOT / "output" / "traces"
REPORTS_DIR = PROJECT_ROOT / "eval" / "reports"

app = FastAPI(
    title="RepoTriage API",
    description="Local-first GitHub issue copilot with hybrid retrieval and NLI claim verification.",
    version="1.0.0",
)

# Enable CORS for local Vite frontend dev server and local clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Concurrency control: only one analysis at a time
_analysis_lock = asyncio.Lock()

# Global pipeline instance (lazily initialized)
_pipeline: RepoTriagePipeline | None = None
_retriever: HybridRetriever | None = None


def get_pipeline() -> RepoTriagePipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RepoTriagePipeline()
    return _pipeline


def get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever


def _sanitize_paths(obj: Any) -> Any:
    """Recursively convert absolute local paths to relative/clean strings."""
    root_str = str(PROJECT_ROOT)
    if isinstance(obj, dict):
        return {k: _sanitize_paths(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_paths(v) for v in obj]
    elif isinstance(obj, str):
        if root_str in obj:
            return obj.replace(root_str, ".").replace("\\", "/")
        return obj
    return obj


# ============================================================================
# Request / Response Schemas
# ============================================================================

class AnalyzeRequest(BaseModel):
    text: str = Field(default="", description="Free-form issue title and body")
    repo: str | None = Field(default=None, description="Repository slug, e.g. pydantic/pydantic")
    issue_url: str | None = Field(default=None, description="Direct GitHub issue URL")


class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query string")
    repo: str | None = Field(default=None, description="Target repository slug")
    top_k: int = Field(default=5, ge=1, le=50, description="Number of results to return")
    source_types: list[str] | None = Field(
        default=None, description="Filter source types: ['issue', 'comment', 'doc']"
    )
    method: str = Field(default="hybrid_rerank", description="bm25 | dense | hybrid | hybrid_rerank")


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/api/health", tags=["System"])
async def health_check() -> dict[str, Any]:
    """Check system health, GPU availability, database chunk counts, and models."""
    total_chunks = 0
    repo_counts: dict[str, int] = {}

    if DB_PATH.exists():
        try:
            conn = sqlite3.connect(str(DB_PATH))
            cursor = conn.cursor()
            cursor.execute("SELECT count(*) FROM chunks")
            total_chunks = cursor.fetchone()[0]

            cursor.execute("SELECT repo, count(*) FROM chunks GROUP BY repo")
            for r, c in cursor.fetchall():
                repo_counts[r] = c
            conn.close()
        except Exception as e:
            logger.warning("Failed to query SQLite DB stats: %s", e)

    gpu_available = False
    gpu_name = None
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        if gpu_available:
            gpu_name = torch.cuda.get_device_name(0)
    except Exception:
        pass

    return {
        "status": "healthy",
        "service": "RepoTriage Local Copilot",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "gpu": {
            "available": gpu_available,
            "device": gpu_name,
        },
        "database": {
            "exists": DB_PATH.exists(),
            "total_chunks": total_chunks,
            "repositories": repo_counts,
        },
        "models": {
            "classifier": "RepoTriage-Pooled-TFIDF-LogReg",
            "dense_retrieval": "BAAI/bge-base-en-v1.5",
            "reranker": "BAAI/bge-reranker-base",
            "llm": "qwen3.5:4b (via Ollama)",
            "guardrail_nli": "cross-encoder/nli-deberta-v3-base",
        },
    }


@app.get("/api/repositories", tags=["Repositories"])
async def list_repositories() -> dict[str, Any]:
    """List all indexed repositories with issue, comment, and doc chunk counts."""
    if not DB_PATH.exists():
        return {"repositories": []}

    def _query_repos() -> list[dict[str, Any]]:
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        query = """
            SELECT repo,
                   SUM(CASE WHEN source_type = 'issue' THEN 1 ELSE 0 END) as issue_chunks,
                   SUM(CASE WHEN source_type = 'comment' THEN 1 ELSE 0 END) as comment_chunks,
                   SUM(CASE WHEN source_type = 'doc' THEN 1 ELSE 0 END) as doc_chunks,
                   count(*) as total_chunks
            FROM chunks
            GROUP BY repo
            ORDER BY repo ASC
        """
        cursor.execute(query)
        rows = cursor.fetchall()
        conn.close()

        repos = []
        for r, issues, comments, docs, total in rows:
            repos.append({
                "repo": r,
                "issue_chunks": issues,
                "comment_chunks": comments,
                "doc_chunks": docs,
                "total_chunks": total,
            })
        return repos

    repos = await asyncio.to_thread(_query_repos)
    return {"repositories": repos}


@app.post("/api/analyze", tags=["Triage"])
async def analyze_issue(req: AnalyzeRequest) -> dict[str, Any]:
    """
    Execute full issue triage pipeline:
    1. Issue classification (frozen TF-IDF + Logistic Regression)
    2. Hybrid retrieval (BM25 + BGE Dense + RRF + Cross-Encoder Reranker)
    3. LLM synthesis with grounded claim extraction (Ollama qwen3.5:4b)
    4. NLI Claim verification (DeBERTa-v3 multi-premise dominance rule)

    Rate limiting / single analysis lock:
    Returns HTTP 429 if another analysis is currently running.
    """
    if not req.text.strip() and not req.issue_url:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either 'text' or 'issue_url' must be provided.",
        )

    # Global lock non-blocking check
    if _analysis_lock.locked():
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Pipeline busy: another issue analysis is currently in progress. Please retry in a few moments.",
        )

    pipeline = get_pipeline()

    try:
        async with _analysis_lock:
            # Run synchronous pipeline in worker thread
            result = await asyncio.to_thread(
                pipeline.analyze,
                text=req.text,
                repo=req.repo,
                issue_url=req.issue_url,
                save_trace=True,
            )
            return _sanitize_paths(result)
    except Exception as e:
        logger.error("Analysis pipeline failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {str(e)}",
        )


@app.post("/api/search", tags=["Retrieval"])
async def search_chunks(req: SearchRequest) -> dict[str, Any]:
    """Execute hybrid retrieval search and return ranked candidates."""
    retriever = get_retriever()

    try:
        results = await asyncio.to_thread(
            retriever.search,
            query=req.query,
            repo=req.repo,
            top_k=req.top_k,
            method=req.method,
            source_types=req.source_types,
            rerank_alpha=0.50,
            return_trace=False,
        )
        return {
            "query": req.query,
            "repo": req.repo,
            "method": req.method,
            "count": len(results),
            "results": _sanitize_paths(results),
        }
    except Exception as e:
        logger.error("Search failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}",
        )


@app.get("/api/analysis/{trace_id}", tags=["Triage"])
async def get_analysis_trace(trace_id: str) -> dict[str, Any]:
    """
    Retrieve a previously executed analysis trace by trace_id.
    Strictly validates trace_id against ^[A-Za-z0-9_-]+$ to prevent path traversal.
    """
    if not re.match(r"^[A-Za-z0-9_-]+$", trace_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid trace_id. Must match '^[A-Za-z0-9_-]+$'.",
        )

    clean_id = trace_id.replace("trace_", "")
    candidate_1 = TRACES_DIR / f"trace_{clean_id}.json"
    candidate_2 = TRACES_DIR / f"{trace_id}.json"

    trace_file = candidate_1 if candidate_1.exists() else (candidate_2 if candidate_2.exists() else None)

    if not trace_file or not trace_file.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis trace '{trace_id}' not found.",
        )

    try:
        with open(trace_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return _sanitize_paths(data)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to read trace file: {str(e)}",
        )


@app.get("/api/evaluation", tags=["Evaluation"])
async def get_evaluation_metrics() -> dict[str, Any]:
    """
    Serve measured benchmark evaluation results:
    - Retrieval 4-Way Ablation with 95% Bootstrap CIs & Reranker Sensitivity
    - Guardrail B1 (Adversarial ANLI 3-class test metrics & confusion matrix)
    - Guardrail B2 (In-domain negative control false-support rates & threshold sweep)
    - Guardrail Perturbation Test (300 rule-constructed pairs)
    - Classifier Live Ingest Generalization Check
    - Classifier Summary Benchmarks (Pooled LogReg, SetFit, Dummy)
    - Human Check status ("not performed" unless labeled)
    """
    files_to_load = {
        "retrieval_ablation": REPORTS_DIR / "retrieval_ablation.json",
        "guardrail_evaluation": REPORTS_DIR / "guardrail_eval.json",
        "perturbation_test": REPORTS_DIR / "perturbation_test.json",
        "classifier_live_check": REPORTS_DIR / "classifier_live_check.json",
        "classifier_summary": REPORTS_DIR / "classifier_summary.json",
    }

    result: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "human_check_status": "not performed",
    }

    for key, path in files_to_load.items():
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    result[key] = json.load(f)
            except Exception as e:
                logger.warning("Failed to load %s: %s", path.name, e)
                result[key] = {"error": f"Failed to parse {path.name}: {str(e)}"}
        else:
            result[key] = None

    return _sanitize_paths(result)
