"""
Evaluation metrics, reporting, and visualizations for Issue Report Classification.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.classifier.config import EXPECTED_LABELS, EXPECTED_REPOS, FIGURES_DIR


def compute_metrics(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str] | None = None,
) -> dict[str, Any]:
    """
    Compute accuracy, per-class precision/recall/F1, macro F1, and weighted F1.
    """
    if labels is None:
        labels = EXPECTED_LABELS

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0))
    macro_prec = float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))

    prec_per_class = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    rec_per_class = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    per_class_dict: dict[str, dict[str, float]] = {}
    for i, label in enumerate(labels):
        per_class_dict[label] = {
            "precision": float(prec_per_class[i]),
            "recall": float(rec_per_class[i]),
            "f1": float(f1_per_class[i]),
        }

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "per_class": per_class_dict,
        "confusion_matrix": cm.tolist(),
        "labels": labels,
    }


def compute_cross_repo_score(
    repo_results: dict[str, dict[str, Any]],
    repos: list[str] | None = None,
) -> dict[str, Any]:
    """
    Compute official cross-repository scores as arithmetic mean of repository-level metrics.
    """
    if repos is None:
        repos = EXPECTED_REPOS

    macro_f1s = [repo_results[r]["macro_f1"] for r in repos if r in repo_results]
    weighted_f1s = [repo_results[r]["weighted_f1"] for r in repos if r in repo_results]
    accuracies = [repo_results[r]["accuracy"] for r in repos if r in repo_results]
    macro_precs = [repo_results[r]["macro_precision"] for r in repos if r in repo_results]
    macro_recs = [repo_results[r]["macro_recall"] for r in repos if r in repo_results]

    per_class_avg: dict[str, dict[str, float]] = {}
    for lbl in EXPECTED_LABELS:
        per_class_avg[lbl] = {
            "precision": float(np.mean([repo_results[r]["per_class"][lbl]["precision"] for r in repos])),
            "recall": float(np.mean([repo_results[r]["per_class"][lbl]["recall"] for r in repos])),
            "f1": float(np.mean([repo_results[r]["per_class"][lbl]["f1"] for r in repos])),
        }

    return {
        "mean_macro_f1": float(np.mean(macro_f1s)),
        "mean_weighted_f1": float(np.mean(weighted_f1s)),
        "mean_accuracy": float(np.mean(accuracies)),
        "mean_macro_precision": float(np.mean(macro_precs)),
        "mean_macro_recall": float(np.mean(macro_recs)),
        "per_class_mean": per_class_avg,
    }


def plot_confusion_matrix(
    cm: list[list[int]] | np.ndarray,
    labels: list[str],
    title: str,
    output_path: Path,
) -> None:
    """Plot and save a confusion matrix figure."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.array(cm)

    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.matshow(matrix, cmap="Blues", alpha=0.85)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            ax.text(j, i, str(val), ha="center", va="center", color="black" if val < matrix.max() / 2 else "white")

    fig.colorbar(cax)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(title, pad=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def export_benchmark_results(
    results_dict: dict[str, Any],
    output_json: Path,
    output_csv: Path | None = None,
) -> None:
    """Save evaluation results to JSON and flat CSV format."""
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=2)

    if output_csv is not None:
        rows = []
        for model_name, model_data in results_dict.items():
            for repo, r_data in model_data.get("repositories", {}).items():
                rows.append({
                    "model": model_name,
                    "repo": repo,
                    "accuracy": r_data.get("accuracy"),
                    "macro_f1": r_data.get("macro_f1"),
                    "weighted_f1": r_data.get("weighted_f1"),
                    "bug_f1": r_data.get("per_class", {}).get("bug", {}).get("f1"),
                    "feature_f1": r_data.get("per_class", {}).get("feature", {}).get("f1"),
                    "question_f1": r_data.get("per_class", {}).get("question", {}).get("f1"),
                })
            # Add cross-repo row
            cross = model_data.get("cross_repository", {})
            rows.append({
                "model": model_name,
                "repo": "OVERALL_MEAN",
                "accuracy": cross.get("mean_accuracy"),
                "macro_f1": cross.get("mean_macro_f1"),
                "weighted_f1": cross.get("mean_weighted_f1"),
                "bug_f1": cross.get("per_class_mean", {}).get("bug", {}).get("f1"),
                "feature_f1": cross.get("per_class_mean", {}).get("feature", {}).get("f1"),
                "question_f1": cross.get("per_class_mean", {}).get("question", {}).get("f1"),
            })
        df = pd.DataFrame(rows)
        df.to_csv(output_csv, index=False)
