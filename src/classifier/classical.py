"""
Classical Machine Learning Baseline (Model A)
==============================================
TF-IDF + Logistic Regression issue classifier and DummyClassifier baseline.

Uses scikit-learn Pipeline and FeatureUnion to eliminate data leakage.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

import joblib
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from src.classifier.config import CLASSICAL_CONFIG, RANDOM_SEED

logger = logging.getLogger(__name__)


def build_tfidf_pipeline(
    feature_mode: Literal["word", "word_char"] = "word",
    C: float = 1.0,
    random_state: int = RANDOM_SEED,
) -> Pipeline:
    """
    Construct a scikit-learn Pipeline with TF-IDF vectorization and Logistic Regression.

    Parameters
    ----------
    feature_mode : 'word' or 'word_char'
        If 'word', uses word unigrams + bigrams.
        If 'word_char', combines word n-grams with character n-grams via FeatureUnion.
    C : float
        Inverse regularization strength for Logistic Regression.
    random_state : int
        Seed for reproducibility.
    """
    word_cfg = CLASSICAL_CONFIG["tfidf_word"]
    word_vec = TfidfVectorizer(
        analyzer=word_cfg["analyzer"],
        ngram_range=word_cfg["ngram_range"],
        max_features=word_cfg["max_features"],
        sublinear_tf=word_cfg["sublinear_tf"],
        min_df=word_cfg["min_df"],
        max_df=word_cfg["max_df"],
    )

    if feature_mode == "word":
        features = word_vec
    elif feature_mode == "word_char":
        char_cfg = CLASSICAL_CONFIG["tfidf_char"]
        char_vec = TfidfVectorizer(
            analyzer=char_cfg["analyzer"],
            ngram_range=char_cfg["ngram_range"],
            max_features=char_cfg["max_features"],
            sublinear_tf=char_cfg["sublinear_tf"],
            min_df=char_cfg["min_df"],
            max_df=char_cfg["max_df"],
        )
        features = FeatureUnion([
            ("word", word_vec),
            ("char", char_vec),
        ])
    else:
        raise ValueError(f"Unknown feature_mode '{feature_mode}'. Use 'word' or 'word_char'.")

    logreg = LogisticRegression(
        C=C,
        max_iter=CLASSICAL_CONFIG["logreg"]["max_iter"],
        solver=CLASSICAL_CONFIG["logreg"]["solver"],
        class_weight=CLASSICAL_CONFIG["logreg"]["class_weight"],
        random_state=random_state,
    )

    pipeline = Pipeline([
        ("features", features),
        ("classifier", logreg),
    ])
    return pipeline


def build_dummy_baseline(strategy: str = "most_frequent") -> DummyClassifier:
    """Majority-class or prior baseline for sanity checking."""
    return DummyClassifier(strategy=strategy, random_state=RANDOM_SEED)


class ClassicalClassifierWrapper:
    """
    Serializable wrapper for classical models with standardized prediction interface.
    """

    def __init__(self, model: Any, model_id: str, feature_mode: str = "word", C: float = 1.0):
        self.model = model
        self.model_id = model_id
        self.feature_mode = feature_mode
        self.C = C

    def fit(self, texts: list[str], labels: list[str]) -> ClassicalClassifierWrapper:
        self.model.fit(texts, labels)
        return self

    def predict(self, texts: list[str]) -> list[str]:
        return list(self.model.predict(texts))

    def predict_proba(self, texts: list[str]) -> list[dict[str, float]]:
        if not hasattr(self.model, "predict_proba"):
            raise AttributeError(f"{self.model.__class__.__name__} does not support predict_proba")
        probs = self.model.predict_proba(texts)
        classes = list(self.model.classes_)
        results = []
        for row in probs:
            results.append({str(cls_name): float(prob) for cls_name, prob in zip(classes, row)})
        return results

    def save(self, filepath: Path) -> Path:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)
        logger.info("Saved classical model to %s", filepath)
        return filepath

    @classmethod
    def load(cls, filepath: Path) -> ClassicalClassifierWrapper:
        wrapper = joblib.load(filepath)
        if not isinstance(wrapper, cls):
            raise TypeError(f"Loaded object is {type(wrapper)}, expected {cls}")
        return wrapper
