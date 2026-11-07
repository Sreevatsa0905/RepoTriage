"""
Cross-Encoder Reranking for RepoTriage
======================================
Reranks fused retrieval candidates using BAAI/bge-reranker-base.
Scores top 20-30 candidates, formats top 5-8 evidence items with stable IDs (E1, E2, ...).
"""

from __future__ import annotations

import gc
import logging
import math
from typing import Any, Sequence

import numpy as np
import torch

import src.config

logger = logging.getLogger(__name__)

RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"
DEFAULT_CANDIDATE_POOL = 25
DEFAULT_TOP_EVIDENCE = 6
DEFAULT_ALPHA = 0.50


class Reranker:
    """
    Cross-encoder reranker using BAAI/bge-reranker-base with RRF rank blending.
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
            logger.info("Loading reranker %s on %s (max_length=512)...", self.model_name, self.device)
            kwargs: dict[str, Any] = {"max_length": 512}
            if self.device == "cuda":
                kwargs["model_kwargs"] = {"torch_dtype": torch.float16}
            try:
                self._model = CrossEncoder(self.model_name, device=self.device, local_files_only=True, **kwargs)
            except Exception:
                self._model = CrossEncoder(self.model_name, device=self.device, local_files_only=False, **kwargs)
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
        alpha: float | None = DEFAULT_ALPHA,
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
        alpha : float | None
            Blending weight: final = (1 - alpha) * rrf_minmax + alpha * sigmoid(rerank_logit).
            If None, sorts strictly by reranker score. Fixed default: 0.50.

        Returns
        -------
        list[dict]
            Top evidence chunks with scores, ranks, and evidence_id (E1..En).
        """
        pool = list(candidates[:candidate_pool])
        if not pool:
            return []

        model = self._get_model()
        pairs = []
        for c in pool:
            heading = (c.get("heading") or "").strip()
            text = (c.get("text") or "").strip()
            if heading and not text.startswith(heading):
                passage = f"{heading}\n{text}"
            else:
                passage = text
            pairs.append([query, passage])

        scores = model.predict(pairs, show_progress_bar=False, convert_to_numpy=True)

        # RRF min-max normalization
        rrf_raw = [float(c.get("rrf_score", 0.0)) for c in pool]
        min_rrf = min(rrf_raw) if rrf_raw else 0.0
        max_rrf = max(rrf_raw) if rrf_raw else 1.0
        rrf_range = (max_rrf - min_rrf) if (max_rrf - min_rrf) > 1e-9 else 1.0

        scored_candidates = []
        for c, score in zip(pool, scores):
            item = dict(c)
            logit = float(score)
            sig = 1.0 / (1.0 + math.exp(-logit))
            rrf_val = float(c.get("rrf_score", 0.0))
            rrf_norm = (rrf_val - min_rrf) / rrf_range

            item["rerank_logit"] = logit
            item["rerank_score"] = logit
            item["rerank_sigmoid"] = sig
            item["rrf_minmax"] = rrf_norm

            if alpha is not None:
                final_score = (1.0 - alpha) * rrf_norm + alpha * sig
            else:
                final_score = logit

            item["final_score"] = float(final_score)
            item["blend_score"] = float(final_score)
            scored_candidates.append(item)

        scored_candidates.sort(key=lambda x: x["final_score"], reverse=True)

        final_evidence = []
        for idx, item in enumerate(scored_candidates[:top_k], 1):
            item["rerank_rank"] = idx
            item["evidence_id"] = f"E{idx}"
            final_evidence.append(item)

        return final_evidence
