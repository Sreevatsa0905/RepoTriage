"""
Stage 6.6-F: Frozen Classifier Live Check on Ingested Issues
============================================================
Evaluates the frozen TF-IDF + Logistic Regression classifier on real issues
from the 4 indexed repositories (encode/httpx, pydantic/pydantic, psf/requests, pallets/flask)
that carry unambiguous ground-truth type labels:
  - 'bug' -> bug
  - 'enhancement', 'feature', 'feature request' -> feature
  - 'question' -> question

Reports:
  - Label mapping & sample size per repo
  - Coverage (share of predictions >= 0.50 threshold, not marked uncertain)
  - Macro-F1 & Accuracy on covered predictions
  - Macro-F1 & Accuracy on all predictions (uncertain counted as wrong)
  - Per-class precision, recall, F1, and confusion matrices
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from src.classify.predict import predict_issue_type

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "processed" / "repotriage.db"
REPORTS_DIR = PROJECT_ROOT / "eval" / "reports"

LABEL_MAPPING = {
    "bug": "bug",
    "enhancement": "feature",
    "feature": "feature",
    "feature request": "feature",
    "question": "question",
}


def load_labeled_issues() -> tuple[list[dict[str, Any]], dict[str, dict[str, int]]]:
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute("SELECT repo, number, title, body, labels FROM issues WHERE labels IS NOT NULL AND labels != ''")
    rows = cursor.fetchall()
    conn.close()

    mapped_issues: list[dict[str, Any]] = []
    repo_stats: dict[str, dict[str, int]] = {}

    for repo, num, title, body, l_str in rows:
        try:
            raw_labels = json.loads(l_str) if l_str.startswith("[") else [l_str]
        except Exception:
            raw_labels = [l_str]

        names: list[str] = []
        for l in raw_labels:
            n = l.get("name", "") if isinstance(l, dict) else str(l)
            if n:
                names.append(n.strip().lower())

        matched_cats = set(LABEL_MAPPING[n] for n in names if n in LABEL_MAPPING)

        # Unambiguous match: exactly one category matched
        if len(matched_cats) == 1:
            cat = list(matched_cats)[0]
            repo_stats.setdefault(repo, {}).setdefault(cat, 0)
            repo_stats[repo][cat] += 1
            mapped_issues.append({
                "repo": repo,
                "number": num,
                "title": title or "",
                "body": body or "",
                "ground_truth": cat,
            })

    return mapped_issues, repo_stats


def compute_metrics(
    y_true: list[str],
    y_pred: list[str],
    classes: list[str],
) -> dict[str, Any]:
    n = len(y_true)
    if n == 0:
        return {}

    acc = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp) / n
    cm = {c: {c2: 0 for c2 in classes} for c in classes}
    for yt, yp in zip(y_true, y_pred):
        if yt in cm and yp in cm[yt]:
            cm[yt][yp] += 1

    per_class = {}
    f1s = []
    for c in classes:
        tp = cm[c][c]
        fp = sum(cm[r][c] for r in classes if r != c)
        fn = sum(cm[c][col] for col in classes if col != c)
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
        f1s.append(f1)
        per_class[c] = {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
            "support": sum(cm[c].values()),
        }

    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0
    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": per_class,
        "confusion_matrix": cm,
    }


def main() -> dict[str, Any]:
    mapped_issues, repo_stats = load_labeled_issues()
    n = len(mapped_issues)
    classes = ["bug", "feature", "question"]

    y_true = [it["ground_truth"] for it in mapped_issues]
    y_pred_covered = []
    is_covered = []
    y_pred_all_strict = []

    for it in mapped_issues:
        res = predict_issue_type(title=it["title"], body=it["body"], threshold=0.50)
        lbl = res["label"]
        raw_cat = res["predicted_category"]
        if lbl == "uncertain":
            is_covered.append(False)
            y_pred_all_strict.append("uncertain")
        else:
            is_covered.append(True)
            y_pred_covered.append((it["ground_truth"], lbl))
            y_pred_all_strict.append(lbl)

    coverage = sum(is_covered) / n if n else 0.0

    # 1. Metrics on covered subset (predictions >= 0.50 threshold)
    yt_cov = [x[0] for x in y_pred_covered]
    yp_cov = [x[1] for x in y_pred_covered]
    cov_metrics = compute_metrics(yt_cov, yp_cov, classes)

    # 2. Metrics on all predictions where 'uncertain' is counted as incorrect
    all_classes = ["bug", "feature", "question", "uncertain"]
    strict_metrics = compute_metrics(y_true, y_pred_all_strict, all_classes)

    # Macro-F1 across the 3 target classes on all predictions (with recall penalised by uncertain)
    f1s_all = []
    per_class_all = {}
    for c in classes:
        tp = sum(1 for yt, yp in zip(y_true, y_pred_all_strict) if yt == c and yp == c)
        fp = sum(1 for yt, yp in zip(y_true, y_pred_all_strict) if yt != c and yp == c)
        fn = sum(1 for yt, yp in zip(y_true, y_pred_all_strict) if yt == c and yp != c)
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
        f1s_all.append(f1)
        per_class_all[c] = {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
            "support": sum(1 for yt in y_true if yt == c),
        }
    macro_f1_all_3classes = round(sum(f1s_all) / len(f1s_all), 4)

    results = {
        "dataset": "Ingested unambiguous GitHub issues from seed repos",
        "sample_size": n,
        "label_mapping": LABEL_MAPPING,
        "repo_distribution": repo_stats,
        "coverage_threshold": 0.50,
        "coverage_rate": round(coverage, 4),
        "covered_count": sum(is_covered),
        "uncertain_count": n - sum(is_covered),
        "covered_predictions": {
            "n": len(yt_cov),
            "accuracy": cov_metrics.get("accuracy", 0.0),
            "macro_f1": cov_metrics.get("macro_f1", 0.0),
            "per_class": cov_metrics.get("per_class", {}),
            "confusion_matrix": cov_metrics.get("confusion_matrix", {}),
        },
        "all_predictions_uncertain_as_wrong": {
            "n": n,
            "accuracy": round(sum(1 for yt, yp in zip(y_true, y_pred_all_strict) if yt == yp) / n, 4),
            "macro_f1_3classes": macro_f1_all_3classes,
            "per_class": per_class_all,
            "confusion_matrix": strict_metrics.get("confusion_matrix", {}),
        },
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = REPORTS_DIR / "classifier_live_check.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("  STAGE 6.6-F: CLASSIFIER LIVE CHECK ON INGESTED ISSUES")
    print("=" * 70)
    print(f"Total unambiguous labeled issues (n): {n}")
    print(f"Per-repo breakdown:")
    for r, d in repo_stats.items():
        print(f"  • {r}: {d} (Total: {sum(d.values())})")
    print(f"\nCoverage at threshold 0.50: {coverage*100:.2f}% ({sum(is_covered)}/{n})")
    print(f"Uncertain (flagged for review): {n - sum(is_covered)} ({ (1-coverage)*100:.2f}%)")
    print("\n[A] Performance on Covered Predictions (Non-Uncertain Subset):")
    print(f"  • Accuracy: {cov_metrics.get('accuracy', 0.0)*100:.2f}%")
    print(f"  • Macro-F1: {cov_metrics.get('macro_f1', 0.0):.4f}")
    for c, pc in cov_metrics.get("per_class", {}).items():
        print(f"    - {c:8s}: P={pc['precision']:.4f}, R={pc['recall']:.4f}, F1={pc['f1']:.4f} (n={pc['support']})")

    print("\n[B] Performance on All Predictions (Uncertain Counted as Wrong):")
    print(f"  • Overall Accuracy: {results['all_predictions_uncertain_as_wrong']['accuracy']*100:.2f}%")
    print(f"  • Macro-F1 (3-class): {macro_f1_all_3classes:.4f}")
    for c, pc in per_class_all.items():
        print(f"    - {c:8s}: P={pc['precision']:.4f}, R={pc['recall']:.4f}, F1={pc['f1']:.4f} (n={pc['support']})")

    print(f"\nSaved report to {out_file}\n")
    return results


if __name__ == "__main__":
    main()
