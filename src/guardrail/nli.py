"""
NLI Guardrail Provider with Dynamic Label Resolution
====================================================
Uses cross-encoder/nli-deberta-v3-base to verify claims against evidence premises.
Per RepoTriage_SPEC.md Section 7 & Rule 2:
- Dynamic label mapping: reads directly from model.config.id2label.
- FP16 execution on CUDA; CPU fallback.
- Lazy model loading and explicit unload_model() for GPU memory isolation.
"""

from __future__ import annotations

import gc
import logging
from typing import Any, Sequence

import torch

logger = logging.getLogger(__name__)

NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-base"
DEFAULT_BATCH_SIZE = 16


class NLIGuardrail:
    """Natural Language Inference cross-encoder guardrail."""

    def __init__(
        self,
        model_name: str = NLI_MODEL_NAME,
        device: str | None = None,
    ):
        self.model_name = model_name
        self._model = None
        self._tokenizer = None

        if device is not None:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Label indices resolved dynamically
        self.entailment_idx: int = -1
        self.contradiction_idx: int = -1
        self.neutral_idx: int = -1

    def _ensure_model_loaded(self) -> None:
        """Lazy load tokenizer and sequence classification model."""
        if self._model is not None and self._tokenizer is not None:
            return

        import src.config  # ensures HF_TOKEN & env loaded
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        logger.info("Loading NLI model %s on %s...", self.model_name, self.device)

        model_kwargs: dict[str, Any] = {}
        if self.device == "cuda":
            model_kwargs["torch_dtype"] = torch.float16

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=True)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                local_files_only=True,
                **model_kwargs,
            ).to(self.device)
        except Exception:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=False)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                local_files_only=False,
                **model_kwargs,
            ).to(self.device)

        self._model.eval()

        # Dynamic label mapping from config.id2label
        id2label = self._model.config.id2label
        logger.info("NLI dynamic id2label mapping: %s", id2label)

        for idx_raw, label_raw in id2label.items():
            idx = int(idx_raw)
            label = str(label_raw).lower()
            if "entail" in label:
                self.entailment_idx = idx
            elif "contra" in label:
                self.contradiction_idx = idx
            elif "neut" in label:
                self.neutral_idx = idx

        if self.entailment_idx == -1 or self.contradiction_idx == -1:
            raise RuntimeError(f"Failed to resolve entailment/contradiction in id2label: {id2label}")

    def unload_model(self) -> None:
        """Free model and clear PyTorch CUDA memory."""
        if self._model is not None:
            logger.info("Unloading NLI model %s from %s", self.model_name, self.device)
            del self._model
            del self._tokenizer
            self._model = None
            self._tokenizer = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def predict_pairs(
        self,
        premises: Sequence[str],
        hypotheses: Sequence[str],
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> list[dict[str, float]]:
        """
        Score (premise, hypothesis) pairs.

        Returns list of dicts:
        [{'entailment': float, 'contradiction': float, 'neutral': float}, ...]
        """
        if len(premises) != len(hypotheses):
            raise ValueError(
                f"Length mismatch: {len(premises)} premises vs {len(hypotheses)} hypotheses"
            )

        if not premises:
            return []

        self._ensure_model_loaded()

        results: list[dict[str, float]] = []

        for i in range(0, len(premises), batch_size):
            batch_prem = list(premises[i:i + batch_size])
            batch_hypo = list(hypotheses[i:i + batch_size])

            encoded = self._tokenizer(
                batch_prem,
                batch_hypo,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                logits = self._model(**encoded).logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()

            for row in probs:
                p_e = float(row[self.entailment_idx])
                p_c = float(row[self.contradiction_idx])
                p_n = float(row[self.neutral_idx]) if self.neutral_idx != -1 else float(1.0 - p_e - p_c)
                results.append({
                    "entailment": p_e,
                    "contradiction": p_c,
                    "neutral": p_n,
                })

        return results
