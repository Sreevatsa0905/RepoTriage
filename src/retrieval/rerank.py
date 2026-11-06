"""
Cross-Encoder Reranking for RepoTriage
======================================
Reranks fused retrieval candidates using BAAI/bge-reranker-base.
Scores top 20-30 candidates, formats top 5-8 evidence items with stable IDs (E1, E2, ...).
"""

from __future__ import annotations

import gc
import logging
from typing import Any, Sequence

import torch

logger = logging.getLogger(__name__)

RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"
DEFAULT_CANDIDATE_POOL = 25
DEFAULT_TOP_EVIDENCE = 6


class Reranker:
    """
    Cross-encoder reranker using BAAI/bge-reranker-base.
    Scores (query, passage) pairs directly.
    """

    def __init__(
        self,
        model_name: str = RERANKER_MODEL_NAME,
        device: str | None = None,
    ):
        self.model_name = model_name
        self._model = None

        if device is not None:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _get_model(self):
        """Lazy loader for CrossEncoder model."""
        if self._model is None:
            from sentence_transformers import CrossEncoder
            logger.info("Loading reranker %s on %s...", self.model_name, self.device)
            self._model = CrossEncoder(self.model_name, device=self.device)
        return self._model

    def unload_model(self) -> None:
        """Unload model from memory and clear GPU cache."""
        if self._model is not None:
            logger.info("Unloading reranker %s from %s", self.model_name, self.device)
            del self._model
            self._model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def rerank(
        self,
        query: str,
        candidates: Sequence[dict[str, Any]],
        candidate_pool: int = DEFAULT_CANDIDATE_POOL,
        top_k: int = DEFAULT_TOP_EVIDENCE,
    ) -> list[dict[str, Any]]:
        """
        Rerank top candidates and assign stable evidence IDs (E1, E2, ...).

        Parameters
        ----------
        query : str
            User query or issue text.
        candidates : list[dict]
            Fused candidate chunks from RRF.
        candidate_pool : int
            Number of top fused candidates to score with the cross-encoder (default: 25).
        top_k : int
            Final number of evidence items to return (default: 6).

        Returns
        -------
        list[dict]
            Top evidence chunks with rerank_score, rerank_rank, and evidence_id (E1..En).
        """
        pool = list(candidates[:candidate_pool])
        if not pool:
            return []

        model = self._get_model()
        pairs = [[query, c["text"]] for c in pool]

        scores = model.predict(pairs, show_progress_bar=False)

        # Sort pool by cross-encoder score descending
        scored_candidates = []
        for c, score in zip(pool, scores):
            item = dict(c)
            item["rerank_score"] = float(score)
            scored_candidates.append(item)

        scored_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

        final_evidence = []
        for idx, item in enumerate(scored_candidates[:top_k], 1):
            item["rerank_rank"] = idx
            item["evidence_id"] = f"E{idx}"
            final_evidence.append(item)

        return final_evidence
