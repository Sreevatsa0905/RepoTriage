"""
End-to-End Issue Resolution and Triage Pipeline
================================================
Coordinates:
1. Frozen classifier prediction (TF-IDF + Logistic Regression).
2. Hybrid retrieval (BM25 + BGE Dense + RRF + Cross-Encoder reranking) with parent-issue deduplication.
3. Structured generation via Ollama (qwen3.5:4b) with strict schema and evidence citations (E1..En).
4. Citation-guided NLI guardrail verification (cross-encoder/nli-deberta-v3-base).
5. Comprehensive JSON execution trace recording per-stage timings and scores.

Enforces strict hardware safety:
- Never holds PyTorch GPU models and Ollama in VRAM simultaneously.
- Sequential lazy loading with explicit unloading and CUDA cache clearing.
"""

from __future__ import annotations

import gc
import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from src.classify.predict import predict_issue_type
from src.generate.llm import OllamaClient
from src.generate.schema import AnalysisResult
from src.guardrail.nli import NLIGuardrail
from src.guardrail.verify import ClaimVerificationEngine
from src.ingest.github_client import GitHubClient
from src.retrieval.index import HybridRetriever

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRACES_DIR = PROJECT_ROOT / "output" / "traces"


def parse_github_issue_url(url: str) -> tuple[str, str, int] | None:
    """Extract owner, repo, issue_number from a GitHub issue URL."""
    pattern = r"github\.com/([^/]+)/([^/]+)/issues/(\d+)"
    match = re.search(pattern, url)
    if match:
        owner, repo_name, num_str = match.groups()
        return owner, repo_name, int(num_str)
    return None


class RepoTriagePipeline:
    """Coordinator for the full local-first issue triage pipeline."""

    def __init__(
        self,
        retriever: HybridRetriever | None = None,
        ollama_client: OllamaClient | None = None,
        guardrail_engine: ClaimVerificationEngine | None = None,
    ):
        self._retriever = retriever
        self._ollama = ollama_client or OllamaClient()
        self._guardrail = guardrail_engine

    def _get_retriever(self, repo: str | None = None) -> HybridRetriever:
        if self._retriever is None:
            self._retriever = HybridRetriever(repo=repo)
        elif repo and self._retriever.repo != repo:
            self._retriever.reload_corpus(repo=repo)
        return self._retriever

    def _get_guardrail(self) -> ClaimVerificationEngine:
        if self._guardrail is None:
            self._guardrail = ClaimVerificationEngine()
        return self._guardrail

    def analyze(
        self,
        text: str = "",
        repo: str | None = None,
        issue_url: str | None = None,
        save_trace: bool = True,
    ) -> dict[str, Any]:
        """
        Execute end-to-end triage analysis.

        Parameters
        ----------
        text : str
            Free text of issue (title + body).
        repo : str | None
            Target repository (e.g. 'pydantic/pydantic').
        issue_url : str | None
            Direct GitHub issue URL.
        save_trace : bool
            Whether to write full JSON trace to disk.

        Returns
        -------
        dict[str, Any]
            Full triage report and verification results.
        """
        pipeline_start = time.perf_counter()
        timings: dict[str, float] = {}

        # 0. Handle GitHub Issue URL if provided
        fetched_issue: dict[str, Any] | None = None
        if issue_url:
            parsed = parse_github_issue_url(issue_url)
            if parsed:
                owner, repo_name, issue_num = parsed
                inferred_repo = f"{owner}/{repo_name}"
                if not repo:
                    repo = inferred_repo
                gh_client = GitHubClient()
                logger.info("Fetching issue %s#%d via GitHub client...", inferred_repo, issue_num)
                fetched_issue = gh_client.fetch_single_issue(owner, repo_name, issue_num)
                if fetched_issue:
                    title = fetched_issue.get("title", "")
                    body = fetched_issue.get("body", "")
                    text = f"{title}\n\n{body}" if body else title

        if not text.strip():
            raise ValueError("No issue text or valid issue URL provided.")

        # Split title and body for classification
        parts = text.strip().split("\n", 1)
        title = parts[0].strip()
        body = parts[1].strip() if len(parts) > 1 else ""

        # 1. Stage 0 Classifier (Frozen TF-IDF + Logistic Regression)
        t0 = time.perf_counter()
        classification_result = predict_issue_type(title=title, body=body)
        timings["classifier_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        predicted_label = classification_result.get("label", "uncertain")

        # 2. Stage 2 Retrieval (Hybrid RRF + Reranker with parent-issue dedup)
        t0 = time.perf_counter()
        retriever = self._get_retriever(repo=repo)
        # Retrieve top 6 evidence items
        evidence_items = retriever.search(
            query=text,
            repo=repo,
            top_k=6,
            method="hybrid_rerank",
            return_trace=False,
        )
        timings["retrieval_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        # Structure evidence for UI & downstream use
        similar_issues = []
        doc_excerpts = []
        for idx, item in enumerate(evidence_items, 1):
            source_type = item.get("source_type")
            if source_type in ("issue", "comment"):
                similar_issues.append({
                    "evidence_id": item.get("evidence_id", f"E{idx}"),
                    "number": item.get("issue_number"),
                    "heading": item.get("heading"),
                    "url": item.get("url"),
                    "rank": item.get("rerank_rank", idx),
                    "score": item.get("rerank_score", 0.0),
                    "excerpt": item.get("text", "")[:300],
                })
            elif source_type == "doc":
                doc_excerpts.append({
                    "evidence_id": item.get("evidence_id", f"E{idx}"),
                    "file_path": item.get("file_path"),
                    "heading": item.get("heading"),
                    "url": item.get("url"),
                    "rank": item.get("rerank_rank", idx),
                    "score": item.get("rerank_score", 0.0),
                    "excerpt": item.get("text", "")[:300],
                })

        # CRITICAL HARDWARE SAFETY: Unload PyTorch models from GPU before invoking Ollama!
        if retriever.reranker._model is not None:
            retriever.reranker.unload_model()
        if retriever.dense._model is not None:
            retriever.dense.unload_model()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 3. Stage 4 Generation (Ollama Qwen 4B with JSON schema)
        t0 = time.perf_counter()
        analysis_obj, gen_trace = self._ollama.generate_analysis(
            issue_text=text,
            repo=repo,
            evidence_items=evidence_items,
            predicted_type=predicted_label,
            auto_unload=True,  # Ensure Ollama frees GPU VRAM immediately when done
        )
        timings["generation_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        # 4. Stage 5 NLI Guardrail Verification
        t0 = time.perf_counter()
        guardrail = self._get_guardrail()
        verified_claims = guardrail.verify_claims(
            claims=analysis_obj.claims,
            evidence_items=evidence_items,
        )
        # Unload NLI model from GPU
        guardrail.nli.unload_model()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        timings["guardrail_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        total_latency_ms = round((time.perf_counter() - pipeline_start) * 1000, 2)
        timings["total_ms"] = total_latency_ms

        # Compile final response
        response = {
            "query": {
                "text": text,
                "repo": repo,
                "issue_url": issue_url,
            },
            "classification": classification_result,
            "similar_issues": similar_issues,
            "doc_excerpts": doc_excerpts,
            "evidence_items": evidence_items,
            "analysis": analysis_obj.model_dump(),
            "verified_claims": verified_claims,
            "timings_ms": timings,
            "generation_trace": gen_trace,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Save trace to disk
        if save_trace:
            TRACES_DIR.mkdir(parents=True, exist_ok=True)
            trace_filename = f"trace_{int(time.time() * 1000)}.json"
            trace_path = TRACES_DIR / trace_filename
            try:
                with open(trace_path, "w", encoding="utf-8") as f:
                    json.dump(response, f, indent=2, ensure_ascii=False)
                response["trace_file"] = str(trace_path)
                logger.info("Trace written to %s", trace_path)
            except Exception as e:
                logger.warning("Failed writing trace: %s", e)

        return response
