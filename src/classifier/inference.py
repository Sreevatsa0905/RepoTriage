"""
Production inference interface for GitHub Issue Classification.

Designed for standalone use and seamless integration into FastAPI or CLI.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.classifier.config import EXPECTED_LABELS, MODELS_DIR
from src.classifier.preprocessing import combine_issue_text, validate_issue_input

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IssueClassificationResult:
    """Standardized output schema for issue classification."""
    category: str
    scores: dict[str, float]
    model_identifier: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class IssueClassifierService:
    """
    Decoupled inference service for classifying GitHub issues.

    Accepts any backend model supporting `.predict(texts)` and `.predict_proba(texts)`.
    """

    def __init__(self, model_wrapper: Any, model_identifier: str = "RepoTriage-Classifier"):
        self.model_wrapper = model_wrapper
        self.model_identifier = model_identifier

    @classmethod
    def from_saved_model(
        cls,
        model_path: Path | str,
        model_type: str = "classical",
    ) -> IssueClassifierService:
        """
        Factory method to instantiate the inference service from a saved model artifact.
        """
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model artifact not found at {path}")

        if model_type == "classical":
            from src.classifier.classical import ClassicalClassifierWrapper
            wrapper = ClassicalClassifierWrapper.load(path)
            return cls(model_wrapper=wrapper, model_identifier=wrapper.model_id)
        elif model_type == "setfit":
            from src.classifier.setfit_model import SetFitClassifierWrapper
            wrapper = SetFitClassifierWrapper.load(path)
            return cls(model_wrapper=wrapper, model_identifier=wrapper.model_id)
        else:
            raise ValueError(f"Unknown model_type '{model_type}'. Choose 'classical' or 'setfit'.")

    def classify_issue(self, title: Any, body: Any) -> IssueClassificationResult:
        """
        Classify a single issue by title and body.

        Validates and handles missing, empty, or non-string inputs safely.
        """
        valid_title, valid_body = validate_issue_input(title, body)
        combined_text = combine_issue_text(valid_title, valid_body)

        # Fallback for completely empty texts: return empty string or default prior
        if not combined_text:
            combined_text = "issue report"

        # Predictions
        preds = self.model_wrapper.predict([combined_text])
        predicted_category = preds[0]

        # Probabilities / scores
        if hasattr(self.model_wrapper, "predict_proba"):
            try:
                probs_list = self.model_wrapper.predict_proba([combined_text])
                scores = probs_list[0]
            except Exception as e:
                logger.warning("predict_proba failed: %s, falling back to 1.0 for predicted class", e)
                scores = {lbl: (1.0 if lbl == predicted_category else 0.0) for lbl in EXPECTED_LABELS}
        else:
            scores = {lbl: (1.0 if lbl == predicted_category else 0.0) for lbl in EXPECTED_LABELS}

        # Normalize score values to floats with clean precision and standard str keys
        scores = {str(k): round(float(v), 4) for k, v in scores.items()}

        return IssueClassificationResult(
            category=predicted_category,
            scores=scores,
            model_identifier=self.model_identifier,
        )

    def classify_batch(
        self, issues: list[tuple[Any, Any]]
    ) -> list[IssueClassificationResult]:
        """Classify a batch of (title, body) tuples efficiently."""
        return [self.classify_issue(t, b) for t, b in issues]
