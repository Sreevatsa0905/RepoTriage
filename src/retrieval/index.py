"""
Hybrid Retrieval Coordinator for RepoTriage
============================================
Coordinates BM25, BGE dense embeddings, RRF fusion, and Cross-Encoder reranking
over the SQLite chunk store.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from pathlib import Path
from typing import Any, Literal, Sequence

from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.rerank import Reranker

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DB_PATH = PROJECT_ROOT / "data" / "processed" / "repotriage.db"

RetrievalMethod = Literal["bm25", "dense", "hybrid", "hybrid_rerank"]


def deduplicate_evidence_candidates(
    candidates: Sequence[dict[str, Any]],
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    """
    Collapse issue and comment results to unique parent issue_number (keeping the best-ranked
    chunk per issue), while deduplicating doc chunks by file_path + heading so doc chunks with
    issue_number=None are not collapsed together.
    """
    seen_issues: set[int] = set()
    seen_docs: set[tuple[str | None, str | None]] = set()
    deduped: list[dict[str, Any]] = []

    for c in candidates:
        issue_num = c.get("issue_number")
        if issue_num is not None:
            if issue_num in seen_issues:
                continue
            seen_issues.add(issue_num)
        else:
            doc_key = (c.get("file_path"), c.get("heading"))
            if doc_key in seen_docs:
                continue
            seen_docs.add(doc_key)

        deduped.append(c)
        if top_k is not None and len(deduped) >= top_k:
            break

    return deduped


class HybridRetriever:
    """
    Unified multi-stage retrieval coordinator for RepoTriage.
    Loads chunks from SQLite, maintains BM25 and dense indices,
    and executes configurable retrieval pipelines.
    """

    def __init__(
        self,
        db_path: Path = DB_PATH,
        repo: str | None = None,
        lazy_dense: bool = False,
    ):
        self.db_path = db_path
        self.repo = repo
        self.chunks = self._load_chunks_from_db(repo=repo)

        # Sparse index
        self.bm25 = BM25Retriever(self.chunks)

        # Dense index (lazy initialization - do not embed until needed)
        self.dense = DenseRetriever(chunks=None)
        self._dense_prepared = False

        # Reranker (lazy)
        self.reranker = Reranker()

    def _ensure_dense_ready(self) -> None:
        """Ensure dense retriever corpus is embedded and ready."""
        if not self._dense_prepared and self.chunks:
            self.dense.prepare_corpus(self.chunks)
            self._dense_prepared = True

    def _load_chunks_from_db(
        self,
        repo: str | None = None,
        exclude_issue_number: int | None = None,
        max_created_at: str | None = None,
    ) -> list[dict[str, Any]]:
        """Load chunk records from SQLite with optional repo and temporal filters."""
        if not self.db_path.exists():
            logger.warning("Database %s does not exist.", self.db_path)
            return []

        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        query = """
            SELECT c.chunk_id, c.repo, c.source_type, c.issue_number, c.file_path,
                   c.heading, c.url, c.content_hash, c.text, c.token_count,
                   i.created_at as issue_created_at
            FROM chunks c
            LEFT JOIN issues i ON (c.repo = i.repo AND c.issue_number = i.number)
            WHERE 1=1
        """
        params: list[Any] = []

        if repo:
            query += " AND c.repo = ?"
            params.append(repo)

        if exclude_issue_number is not None:
            query += " AND (c.issue_number IS NULL OR c.issue_number != ?)"
            params.append(exclude_issue_number)

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        records = []
        for r in rows:
            rec = dict(r)
            issue_created = rec.pop("issue_created_at", None)

            # Temporal filtering: exclude chunks created after max_created_at
            if max_created_at and issue_created and issue_created > max_created_at:
                continue

            records.append(rec)

        logger.info("Loaded %d chunks from %s (repo=%s)", len(records), self.db_path.name, repo)
        return records

    def reload_corpus(
        self,
        repo: str | None = None,
        exclude_issue_number: int | None = None,
        max_created_at: str | None = None,
    ) -> None:
        """Reload chunks with specific filtering criteria (used during evaluation)."""
        self.repo = repo
        self.chunks = self._load_chunks_from_db(
            repo=repo,
            exclude_issue_number=exclude_issue_number,
            max_created_at=max_created_at,
        )
        self.bm25 = BM25Retriever(self.chunks)
        self._dense_prepared = False

    def search(
        self,
        query: str,
        repo: str | None = None,
        top_k: int = 6,
        method: RetrievalMethod = "hybrid_rerank",
        return_trace: bool = False,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """
        Execute multi-stage retrieval.

        Methods:
        - 'bm25': Pure BM25 sparse search
        - 'dense': Pure BGE dense semantic search
        - 'hybrid': BM25 + Dense fused with RRF (k=60)
        - 'hybrid_rerank': Hybrid fused candidates reranked with bge-reranker-base
        """
        start_time = time.perf_counter()

        # If repo is specified and differs from current loaded corpus, filter
        if repo and repo != self.repo:
            self.reload_corpus(repo=repo)

        bm25_results: list[dict[str, Any]] = []
        dense_results: list[dict[str, Any]] = []
        fused_results: list[dict[str, Any]] = []
        final_results: list[dict[str, Any]] = []

        t_bm25 = 0.0
        t_dense = 0.0
        t_rrf = 0.0
        t_rerank = 0.0

        if method in ("bm25", "hybrid", "hybrid_rerank"):
            t0 = time.perf_counter()
            bm25_results = self.bm25.search(query, top_k=50)
            t_bm25 = time.perf_counter() - t0

        if method in ("dense", "hybrid", "hybrid_rerank"):
            self._ensure_dense_ready()
            t0 = time.perf_counter()
            dense_results = self.dense.search(query, top_k=50)
            t_dense = time.perf_counter() - t0

        if method == "bm25":
            final_results = deduplicate_evidence_candidates(bm25_results, top_k=top_k)
            for idx, item in enumerate(final_results, 1):
                item["evidence_id"] = f"E{idx}"
        elif method == "dense":
            final_results = deduplicate_evidence_candidates(dense_results, top_k=top_k)
            for idx, item in enumerate(final_results, 1):
                item["evidence_id"] = f"E{idx}"
        elif method == "hybrid":
            t0 = time.perf_counter()
            fused_results = reciprocal_rank_fusion(bm25_results, dense_results, k=60, top_k=50)
            t_rrf = time.perf_counter() - t0
            final_results = deduplicate_evidence_candidates(fused_results, top_k=top_k)
            for idx, item in enumerate(final_results, 1):
                item["evidence_id"] = f"E{idx}"
        elif method == "hybrid_rerank":
            t0 = time.perf_counter()
            fused_results = reciprocal_rank_fusion(bm25_results, dense_results, k=60, top_k=50)
            t_rrf = time.perf_counter() - t0

            # Deduplicate fused candidates by parent issue / doc before reranking pool
            deduped_fused = deduplicate_evidence_candidates(fused_results)

            t0 = time.perf_counter()
            final_results = self.reranker.rerank(query, deduped_fused, candidate_pool=25, top_k=top_k)
            t_rerank = time.perf_counter() - t0

        total_latency_ms = (time.perf_counter() - start_time) * 1000.0

        if not return_trace:
            return final_results

        return {
            "query": query,
            "repo": repo or self.repo,
            "method": method,
            "total_latency_ms": round(total_latency_ms, 2),
            "stage_timings_ms": {
                "bm25": round(t_bm25 * 1000, 2),
                "dense": round(t_dense * 1000, 2),
                "rrf": round(t_rrf * 1000, 2),
                "rerank": round(t_rerank * 1000, 2),
            },
            "results": final_results,
            "bm25_top5": [{"chunk_id": r["chunk_id"], "score": r["bm25_score"]} for r in bm25_results[:5]],
            "dense_top5": [{"chunk_id": r["chunk_id"], "score": r["dense_score"]} for r in dense_results[:5]],
        }
