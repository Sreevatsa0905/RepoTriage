"""
Training routines for Issue Report Classification.

Supports:
- Model A: Repository-specific TF-IDF + Logistic Regression with validation HP tuning.
- Model A Baseline: DummyClassifier sanity check.
- Model C: Pooled classifier with Leave-One-Group-Out (LOGO) cross-validation across repositories.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

from src.classifier.classical import (
    ClassicalClassifierWrapper,
    build_dummy_baseline,
    build_tfidf_pipeline,
)
from src.classifier.config import (
    CLASSICAL_CONFIG,
    EXPECTED_REPOS,
    MODELS_DIR,
    RANDOM_SEED,
)
from src.classifier.evaluate import compute_metrics
from src.classifier.preprocessing import combine_issue_text

logger = logging.getLogger(__name__)


def prepare_repo_data(
    records: list[dict[str, str]], repo: str
) -> tuple[list[str], list[str]]:
    """Extract combined texts and labels for a specific repository."""
    repo_records = [r for r in records if r["repo"] == repo]
    texts = [combine_issue_text(r.get("title", ""), r.get("body", "")) for r in repo_records]
    labels = [r["label"].strip().lower() for r in repo_records]
    return texts, labels


def tune_and_train_classical_repo(
    repo: str,
    train_records: list[dict[str, str]],
    feature_modes: list[Literal["word", "word_char"]] | None = None,
    C_values: list[float] | None = None,
    random_state: int = RANDOM_SEED,
) -> tuple[ClassicalClassifierWrapper, dict[str, Any]]:
    """
    Tune hyperparameters on a stratified validation split of the repo's training data,
    then retrain on the full repository training subset.
    """
    if feature_modes is None:
        feature_modes = ["word", "word_char"]
    if C_values is None:
        C_values = CLASSICAL_CONFIG["hp_search"]["C_values"]

    texts, labels = prepare_repo_data(train_records, repo)

    # 80/20 Stratified Validation split from training data ONLY
    val_fraction = CLASSICAL_CONFIG.get("val_fraction", 0.2)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=random_state)
    train_idx, val_idx = next(sss.split(texts, labels))

    X_train_sub = [texts[i] for i in train_idx]
    y_train_sub = [labels[i] for i in train_idx]
    X_val = [texts[i] for i in val_idx]
    y_val = [labels[i] for i in val_idx]

    best_score = -1.0
    best_config: dict[str, Any] = {"feature_mode": "word", "C": 1.0}
    tuning_history: list[dict[str, Any]] = []

    for f_mode in feature_modes:
        for c_val in C_values:
            pipe = build_tfidf_pipeline(feature_mode=f_mode, C=c_val, random_state=random_state)
            pipe.fit(X_train_sub, y_train_sub)
            val_preds = pipe.predict(X_val)
            val_metrics = compute_metrics(y_val, list(val_preds))
            macro_f1 = val_metrics["macro_f1"]

            tuning_history.append({
                "feature_mode": f_mode,
                "C": c_val,
                "val_macro_f1": macro_f1,
                "val_accuracy": val_metrics["accuracy"],
            })

            if macro_f1 > best_score:
                best_score = macro_f1
                best_config = {"feature_mode": f_mode, "C": c_val}

    logger.info(
        "Repo '%s' HP selection: best config=%s, val_macro_f1=%.4f",
        repo, best_config, best_score,
    )

    # Retrain on the FULL repository training split (300 records)
    final_pipe = build_tfidf_pipeline(
        feature_mode=best_config["feature_mode"],
        C=best_config["C"],
        random_state=random_state,
    )
    final_pipe.fit(texts, labels)

    safe_name = repo.replace("/", "-")
    model_id = f"tfidf-logreg-{safe_name}"
    wrapper = ClassicalClassifierWrapper(
        model=final_pipe,
        model_id=model_id,
        feature_mode=best_config["feature_mode"],
        C=best_config["C"],
    )

    save_path = MODELS_DIR / f"classical_{safe_name}.joblib"
    wrapper.save(save_path)

    metadata = {
        "repo": repo,
        "best_config": best_config,
        "best_val_macro_f1": best_score,
        "tuning_history": tuning_history,
        "model_path": str(save_path),
        "train_samples": len(texts),
    }
    return wrapper, metadata


def train_dummy_repo_baseline(
    repo: str,
    train_records: list[dict[str, str]],
) -> ClassicalClassifierWrapper:
    """Train a majority-class DummyClassifier baseline on repo training records."""
    texts, labels = prepare_repo_data(train_records, repo)
    dummy = build_dummy_baseline(strategy="most_frequent")
    dummy.fit(texts, labels)

    safe_name = repo.replace("/", "-")
    wrapper = ClassicalClassifierWrapper(
        model=dummy,
        model_id=f"dummy-{safe_name}",
        feature_mode="none",
        C=0.0,
    )
    save_path = MODELS_DIR / f"dummy_{safe_name}.joblib"
    wrapper.save(save_path)
    return wrapper


def run_logo_cross_validation_on_train(
    train_records: list[dict[str, str]],
    feature_mode: Literal["word", "word_char"] = "word",
    C: float = 1.0,
    random_state: int = RANDOM_SEED,
) -> dict[str, Any]:
    """
    Leave-One-Group-Out (LOGO) cross validation on official train split,
    holding out one repository per fold.
    """
    fold_results: dict[str, dict[str, Any]] = {}

    for held_out_repo in EXPECTED_REPOS:
        train_fold_records = [r for r in train_records if r["repo"] != held_out_repo]
        val_fold_records = [r for r in train_records if r["repo"] == held_out_repo]

        X_train_fold = [combine_issue_text(r.get("title", ""), r.get("body", "")) for r in train_fold_records]
        y_train_fold = [r["label"].strip().lower() for r in train_fold_records]

        X_val_fold = [combine_issue_text(r.get("title", ""), r.get("body", "")) for r in val_fold_records]
        y_val_fold = [r["label"].strip().lower() for r in val_fold_records]

        pipe = build_tfidf_pipeline(feature_mode=feature_mode, C=C, random_state=random_state)
        pipe.fit(X_train_fold, y_train_fold)

        preds = pipe.predict(X_val_fold)
        metrics = compute_metrics(y_val_fold, list(preds))
        fold_results[held_out_repo] = metrics

    macro_f1s = [fold_results[r]["macro_f1"] for r in EXPECTED_REPOS]
    mean_logo_macro_f1 = float(np.mean(macro_f1s))

    return {
        "fold_results": fold_results,
        "mean_logo_macro_f1": mean_logo_macro_f1,
        "feature_mode": feature_mode,
        "C": C,
    }


def train_pooled_classifier(
    train_records: list[dict[str, str]],
    feature_modes: list[Literal["word", "word_char"]] | None = None,
    C_values: list[float] | None = None,
    random_state: int = RANDOM_SEED,
) -> tuple[ClassicalClassifierWrapper, dict[str, Any]]:
    """
    Select pooled model configuration using LOGO cross-validation on train data,
    then retrain on all 1,500 training records.
    """
    if feature_modes is None:
        feature_modes = ["word", "word_char"]
    if C_values is None:
        C_values = [0.1, 0.5, 1.0, 2.0, 5.0]

    best_score = -1.0
    best_config: dict[str, Any] = {"feature_mode": "word", "C": 1.0}
    cv_search_results: list[dict[str, Any]] = []

    logger.info("Starting Leave-One-Repository-Out CV search for Pooled Classifier...")
    for f_mode in feature_modes:
        for c_val in C_values:
            cv_res = run_logo_cross_validation_on_train(
                train_records, feature_mode=f_mode, C=c_val, random_state=random_state
            )
            mean_f1 = cv_res["mean_logo_macro_f1"]
            cv_search_results.append({
                "feature_mode": f_mode,
                "C": c_val,
                "mean_logo_macro_f1": mean_f1,
                "fold_scores": {r: round(cv_res["fold_results"][r]["macro_f1"], 4) for r in EXPECTED_REPOS},
            })
            if mean_f1 > best_score:
                best_score = mean_f1
                best_config = {"feature_mode": f_mode, "C": c_val}

    logger.info(
        "Pooled Model Selected Config: %s with LOGO CV Macro F1: %.4f",
        best_config, best_score,
    )

    # Retrain on the FULL 1,500 training records
    all_train_texts = [combine_issue_text(r.get("title", ""), r.get("body", "")) for r in train_records]
    all_train_labels = [r["label"].strip().lower() for r in train_records]

    final_pipe = build_tfidf_pipeline(
        feature_mode=best_config["feature_mode"],
        C=best_config["C"],
        random_state=random_state,
    )
    final_pipe.fit(all_train_texts, all_train_labels)

    wrapper = ClassicalClassifierWrapper(
        model=final_pipe,
        model_id="RepoTriage-Pooled-TFIDF-LogReg",
        feature_mode=best_config["feature_mode"],
        C=best_config["C"],
    )
    save_path = MODELS_DIR / "pooled_classifier.joblib"
    wrapper.save(save_path)

    metadata = {
        "best_config": best_config,
        "best_logo_cv_macro_f1": best_score,
        "cv_search_results": cv_search_results,
        "total_train_samples": len(all_train_texts),
        "model_path": str(save_path),
    }
    return wrapper, metadata
