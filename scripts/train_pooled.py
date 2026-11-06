#!/usr/bin/env python3
"""
Model C: RepoTriage Pooled Classifier Pipeline.

Requirements:
1. Leave-One-Group-Out (LOGO) cross-validation across repositories on the official train split.
2. Reports F1 separately for each held-out repository and the arithmetic mean across folds.
3. Selects best configuration using training-only LOGO CV.
4. Retrains on the complete official training split (1,500 records).
5. Evaluates on the official test split (1,500 records) once.
6. Evaluates per-repository breakdown on the test split.
7. Saves model to models/pooled_classifier.joblib and results to results/pooled_results.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

csv.field_size_limit(2147483647)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.classifier.config import (
    EXPECTED_LABELS,
    EXPECTED_REPOS,
    FIGURES_DIR,
    MODELS_DIR,
    RESULTS_DIR,
    TEST_CSV,
    TRAIN_CSV,
)
from src.classifier.evaluate import compute_metrics, plot_confusion_matrix
from src.classifier.preprocessing import combine_issue_text
from src.classifier.train import prepare_repo_data, train_pooled_classifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_dataset(path: Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))


def run_pooled_experiment() -> dict[str, Any]:
    logger.info("Loading official datasets...")
    train_records = load_dataset(TRAIN_CSV)
    test_records = load_dataset(TEST_CSV)

    print("=" * 75)
    print("  MODEL C: REPOTRIAGE POOLED CLASSIFIER")
    print("  Leave-One-Repository-Out (LOGO) Cross-Validation & Final Evaluation")
    print("=" * 75)

    # 1. LOGO Cross-Validation & Training
    wrapper, cv_metadata = train_pooled_classifier(train_records)

    # 2. Evaluation on the Official Test Split
    logger.info("Evaluating Pooled Model on the Official Test Split (1,500 records)...")
    all_test_texts = [combine_issue_text(r.get("title", ""), r.get("body", "")) for r in test_records]
    all_test_labels = [r["label"].strip().lower() for r in test_records]

    overall_test_preds = wrapper.predict(all_test_texts)
    overall_test_metrics = compute_metrics(all_test_labels, overall_test_preds)

    # 3. Per-Repository breakdown on the Test Split
    per_repo_test_metrics: dict[str, dict[str, Any]] = {}
    for repo in EXPECTED_REPOS:
        test_texts, test_labels = prepare_repo_data(test_records, repo)
        preds = wrapper.predict(test_texts)
        metrics = compute_metrics(test_labels, preds)
        per_repo_test_metrics[repo] = metrics

    # Plot confusion matrix for overall pooled test set
    cm_path = FIGURES_DIR / "cm_pooled_overall_test.png"
    plot_confusion_matrix(
        overall_test_metrics["confusion_matrix"],
        EXPECTED_LABELS,
        "Pooled Model: Overall Test Set",
        cm_path,
    )

    results = {
        "model_id": wrapper.model_id,
        "best_config": cv_metadata["best_config"],
        "logo_cv_training_validation": {
            "mean_logo_macro_f1": cv_metadata["best_logo_cv_macro_f1"],
            "all_cv_configs_evaluated": cv_metadata["cv_search_results"],
        },
        "official_test_evaluation": {
            "overall": overall_test_metrics,
            "per_repository": per_repo_test_metrics,
            "mean_per_repo_test_macro_f1": float(np.mean([per_repo_test_metrics[r]["macro_f1"] for r in EXPECTED_REPOS])),
        },
    }

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = RESULTS_DIR / "pooled_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved pooled results to %s", out_json)

    # Print Report
    print("\n" + "=" * 75)
    print("  POOLED CLASSIFIER RESULTS SUMMARY")
    print("=" * 75)
    print(f"  Selected Config : {cv_metadata['best_config']}")
    print(f"  Training LOGO CV Mean Macro F1 : {cv_metadata['best_logo_cv_macro_f1']:.4f}")
    print("\n  Per-Repository Held-Out Fold Macro F1 (LOGO CV on Train Set):")
    # Find best config details
    best_res = next(r for r in cv_metadata["cv_search_results"] if r["C"] == cv_metadata["best_config"]["C"] and r["feature_mode"] == cv_metadata["best_config"]["feature_mode"])
    for repo, score in best_res["fold_scores"].items():
        print(f"    Held-out {repo:<25} : {score:.4f}")

    print("\n  Official Test Split Performance (Trained on full 1,500 train records):")
    print(f"    Overall Test Accuracy  : {overall_test_metrics['accuracy']:.4f}")
    print(f"    Overall Test Macro F1  : {overall_test_metrics['macro_f1']:.4f}")
    print(f"    Overall Test Weighted F1: {overall_test_metrics['weighted_f1']:.4f}")
    print("\n    Per-Class F1 on Test Set:")
    for lbl in EXPECTED_LABELS:
        cls_m = overall_test_metrics["per_class"][lbl]
        print(f"      {lbl:<10} - Prec: {cls_m['precision']:.4f}, Rec: {cls_m['recall']:.4f}, F1: {cls_m['f1']:.4f}")

    print("\n  Per-Repository Breakdown on Official Test Split:")
    for repo in EXPECTED_REPOS:
        rm = per_repo_test_metrics[repo]
        print(f"    {repo:<25} - Acc: {rm['accuracy']:.4f}, Macro F1: {rm['macro_f1']:.4f}")

    mean_per_repo = results["official_test_evaluation"]["mean_per_repo_test_macro_f1"]
    print(f"\n  Arithmetic Mean of Repo Macro F1s on Test Set: {mean_per_repo:.4f}")
    print("=" * 75 + "\n")

    return results


def main() -> None:
    run_pooled_experiment()


if __name__ == "__main__":
    main()
