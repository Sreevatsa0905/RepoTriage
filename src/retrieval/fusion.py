"""
Rank Fusion for RepoTriage
==========================
Implements Reciprocal Rank Fusion (RRF, k=60) combining sparse (BM25)
and dense (BGE) retrieval results, deduplicating by chunk_id and preserving traces.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

logger = logging.getLogger(__name__)

DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(
    bm25_results: Sequence[dict[str, Any]],
    dense_results: Sequence[dict[str, Any]],
    k: int = DEFAULT_RRF_K,
    top_k: int = 50,
) -> list[dict[str, Any]]:
    """
    Fuse BM25 and Dense search results using Reciprocal Rank Fusion (RRF).

    Formula:
        RRF_score(d) = sum(1.0 / (k + rank_m(d))) for each method m

    Parameters
    ----------
    bm25_results : list[dict]
        Ranked chunks from BM25 (each having 'chunk_id', 'bm25_rank', 'bm25_score').
    dense_results : list[dict]
        Ranked chunks from Dense retrieval (each having 'chunk_id', 'dense_rank', 'dense_score').
    k : int
        RRF smoothing constant (default: 60).
    top_k : int
        Number of fused candidates to return.

    Returns
    -------
    list[dict]
        Fused chunks ordered by rrf_score descending, with per-method ranks preserved.
    """
    chunk_map: dict[str, dict[str, Any]] = {}
    rrf_scores: dict[str, float] = {}

    # Process BM25
    for rank, item in enumerate(bm25_results, 1):
        cid = item["chunk_id"]
        if cid not in chunk_map:
            chunk_map[cid] = dict(item)
        else:
            chunk_map[cid]["bm25_rank"] = item.get("bm25_rank", rank)
            chunk_map[cid]["bm25_score"] = item.get("bm25_score", 0.0)

        chunk_map[cid]["bm25_rank"] = item.get("bm25_rank", rank)
        chunk_map[cid]["bm25_score"] = item.get("bm25_score", 0.0)
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (k + rank))

    # Process Dense
    for rank, item in enumerate(dense_results, 1):
        cid = item["chunk_id"]
        if cid not in chunk_map:
            chunk_map[cid] = dict(item)
        else:
            chunk_map[cid]["dense_rank"] = item.get("dense_rank", rank)
            chunk_map[cid]["dense_score"] = item.get("dense_score", 0.0)

        chunk_map[cid]["dense_rank"] = item.get("dense_rank", rank)
        chunk_map[cid]["dense_score"] = item.get("dense_score", 0.0)
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (k + rank))

    # Sort candidates by RRF score descending
    sorted_cids = sorted(
        rrf_scores.keys(),
        key=lambda cid: (
            rrf_scores[cid],
            -chunk_map[cid].get("dense_rank", 9999),
            -chunk_map[cid].get("bm25_rank", 9999),
        ),
        reverse=True,
    )

    fused_results: list[dict[str, Any]] = []
    for rank, cid in enumerate(sorted_cids[:top_k], 1):
        record = chunk_map[cid]
        record["rrf_score"] = rrf_scores[cid]
        record["rrf_rank"] = rank
        fused_results.append(record)

    return fused_results
