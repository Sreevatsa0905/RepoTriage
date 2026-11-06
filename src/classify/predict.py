"""
Frozen Issue Classification Wrapper with Uncertain Threshold
==============================================================
Wraps the trained pooled TF-IDF + Logistic Regression artifact
(`models/pooled_classifier.joblib`) as specified in RepoTriage_SPEC.md (sections 1b & 5).

No retraining or modifications to model weights are performed.
Threshold selection is derived strictly from Leave-One-Repository-Out (LOGO)
cross-validation on the official training split (never from the test split).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.classifier.config import MODELS_DIR
from src.classifier.inference import IssueClassifierService

logger = logging.getLogger(__name__)

# Default artifact path and empirically derived threshold
DEFAULT_MODEL_PATH = MODELS_DIR / "pooled_classifier.joblib"
# Threshold chosen from LOGO validation on the training split:
# At tau=0.50, coverage is 60.1% and accuracy of retained predictions rises from 59.4% to 67.4%.
DEFAULT_UNCERTAIN_THRESHOLD = 0.50

# Benchmark reference numbers per Section 1b
BENCHMARK_METRICS = {
    "pooled_test_macro_f1": 0.7605,  # 0.76 (repositories seen in training)
    "cold_start_macro_f1": 0.5941,   # 0.594 (leave-one-repository-out cross-validation)
}

_SERVICE_INSTANCE: IssueClassifierService | None = None


def get_classifier_service(model_path: Path | str = DEFAULT_MODEL_PATH) -> IssueClassifierService:
    """Lazy-load and cache the singleton IssueClassifierService instance."""
    global _SERVICE_INSTANCE
    if _SERVICE_INSTANCE is None:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Frozen classifier model not found at {path}. "
                "Ensure models/pooled_classifier.joblib exists."
            )
        _SERVICE_INSTANCE = IssueClassifierService.from_saved_model(path, model_type="classical")
    return _SERVICE_INSTANCE


def predict_issue_type(
    title: str,
    body: str = "",
    threshold: float = DEFAULT_UNCERTAIN_THRESHOLD,
    model_path: Path | str = DEFAULT_MODEL_PATH,
) -> dict[str, Any]:
    """
    Classify an issue report with an uncertainty threshold.

    Parameters
    ----------
    title : str
        Issue title or raw text.
    body : str, optional
        Issue description or body text.
    threshold : float, default 0.50
        Confidence cutoff below which the prediction is deemed 'uncertain'.
    model_path : Path | str, optional
        Path to the frozen joblib classifier artifact.

    Returns
    -------
    dict[str, Any]
        {
            "label": "bug" | "feature" | "question" | "uncertain",
            "predicted_category": str,
            "scores": dict[str, float],
            "confidence": float,
            "threshold": float,
            "needs_review": bool,
            "model_version": str,
            "benchmark_metrics": {
                "pooled_test_macro_f1": 0.76,
                "cold_start_macro_f1": 0.594
            }
        }
    """
    # If only title is passed and it contains newlines, split title and body
    if not body and "\n" in title:
        lines = title.strip().split("\n", 1)
        clean_title = lines[0].strip()
        clean_body = lines[1].strip() if len(lines) > 1 else ""
    else:
        clean_title = title.strip()
        clean_body = body.strip()

    service = get_classifier_service(model_path)
    result = service.classify_issue(clean_title, clean_body)

    scores = result.scores
    predicted_category = result.category
    top_score = scores.get(predicted_category, max(scores.values()) if scores else 0.0)

    # Uncertainty check
    if top_score < threshold:
        label = "uncertain"
        needs_review = True
    else:
        label = predicted_category
        needs_review = False

    return {
        "label": label,
        "predicted_category": predicted_category,
        "scores": scores,
        "confidence": round(float(top_score), 4),
        "threshold": threshold,
        "needs_review": needs_review,
        "model_version": result.model_identifier,
        "benchmark_metrics": {
            "pooled_test_macro_f1": 0.76,
            "cold_start_macro_f1": 0.594,
        },
    }
