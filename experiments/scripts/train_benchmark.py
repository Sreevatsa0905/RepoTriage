#!/usr/bin/env python3
"""
Official 5-Repository Benchmark Pipeline.

Trains and evaluates:
1. DummyClassifier (sanity check)
2. Model A: TF-IDF + Logistic Regression (with HP selection on train validation fold)
3. Model B: SetFit few-shot classifier (following official NLBSE 2024 notebook)

Follows the official protocol:
- 5 independent models per approach
- Trained exclusively on repository's training records (300 per repo)
- Evaluated exclusively on repository's test records (300 per repo)
- Cross-repository score is the arithmetic mean of the five repository macro F1 scores.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Increase CSV limit
import csv
csv.field_size_limit(2147483647)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.classifier.config import (
    EXPECTED_LABELS,
    EXPECTED_REPOS,
    FIGURES_DIR,
    RESULTS_DIR,
    TEST_CSV,
    TRAIN_CSV,
)
from src.classifier.evaluate import (
    compute_cross_repo_score,
    compute_metrics,
    export_benchmark_results,
    plot_confusion_matrix,
)
from src.classifier.preprocessing import combine_issue_text
from src.classifier.train import (
    prepare_repo_data,
    train_dummy_repo_baseline,
    tune_and_train_classical_repo,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Published reference from NLBSE 2024 README
PUBLISHED_SETFIT_MACRO_F1 = 0.8270


def load_dataset(path: Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))


def run_benchmark(
    run_dummy: bool = True,
    run_classical: bool = True,
    run_setfit: bool = True,
    setfit_iterations: int = 1,
) -> dict[str, Any]:
    logger.info("Loading official datasets...")
    train_records = load_dataset(TRAIN_CSV)
    test_records = load_dataset(TEST_CSV)

    all_results: dict[str, Any] = {}
    out_json = RESULTS_DIR / "benchmark_results.json"
    if out_json.exists():
        try:
            with open(out_json, "r", encoding="utf-8") as f:
                all_results = json.load(f)
        except Exception as e:
            logger.warning("Could not load existing benchmark results: %s", e)

    # -------------------------------------------------------------------------
    # 1. Dummy Baseline
    # -------------------------------------------------------------------------
    if run_dummy or "DummyClassifier" not in all_results:
        logger.info("\n" + "=" * 60 + "\n  RUNNING DUMMY CLASSIFIER BASELINE\n" + "=" * 60)
        dummy_repo_metrics: dict[str, Any] = {}
        for repo in EXPECTED_REPOS:
            wrapper = train_dummy_repo_baseline(repo, train_records)
            test_texts, test_labels = prepare_repo_data(test_records, repo)
            preds = wrapper.predict(test_texts)
            metrics = compute_metrics(test_labels, preds)
            dummy_repo_metrics[repo] = metrics

        dummy_cross = compute_cross_repo_score(dummy_repo_metrics)
        all_results["DummyClassifier"] = {
            "repositories": dummy_repo_metrics,
            "cross_repository": dummy_cross,
        }

    # -------------------------------------------------------------------------
    # 2. Model A: TF-IDF + Logistic Regression
    # -------------------------------------------------------------------------
    if run_classical or "TF-IDF_LogReg" not in all_results:
        logger.info("\n" + "=" * 60 + "\n  RUNNING MODEL A: TF-IDF + LOGISTIC REGRESSION\n" + "=" * 60)
        classical_repo_metrics: dict[str, Any] = {}
        classical_metadata: dict[str, Any] = {}

        for repo in EXPECTED_REPOS:
            logger.info("Training classical model for %s...", repo)
            wrapper, meta = tune_and_train_classical_repo(repo, train_records)
            classical_metadata[repo] = meta

            test_texts, test_labels = prepare_repo_data(test_records, repo)
            preds = wrapper.predict(test_texts)
            metrics = compute_metrics(test_labels, preds)
            classical_repo_metrics[repo] = metrics

            # Plot confusion matrix
            safe_name = repo.replace("/", "-")
            cm_path = FIGURES_DIR / f"cm_classical_{safe_name}.png"
            plot_confusion_matrix(
                metrics["confusion_matrix"],
                EXPECTED_LABELS,
                f"TF-IDF + LogReg: {repo}",
                cm_path,
            )

        classical_cross = compute_cross_repo_score(classical_repo_metrics)
        all_results["TF-IDF_LogReg"] = {
            "repositories": classical_repo_metrics,
            "cross_repository": classical_cross,
            "hyperparameter_selection": classical_metadata,
        }

    # -------------------------------------------------------------------------
    # 3. Model B: SetFit Few-Shot Classifier
    # -------------------------------------------------------------------------
    if run_setfit:
        logger.info("\n" + "=" * 60 + f"\n  RUNNING MODEL B: SETFIT (all-mpnet-base-v2, iters={setfit_iterations})\n" + "=" * 60)
        from src.classifier.setfit_model import train_setfit_repo_model

        setfit_repo_metrics: dict[str, Any] = {}
        setfit_metadata: dict[str, Any] = {}

        for repo in EXPECTED_REPOS:
            logger.info("Training SetFit model for %s (num_iterations=%d)...", repo, setfit_iterations)
            wrapper, meta = train_setfit_repo_model(
                repo=repo,
                train_records=train_records,
                num_iterations=setfit_iterations,
                num_epochs=1,
            )
            setfit_metadata[repo] = meta

            test_texts, test_labels = prepare_repo_data(test_records, repo)
            preds = wrapper.predict(test_texts)
            metrics = compute_metrics(test_labels, preds)
            setfit_repo_metrics[repo] = metrics

            safe_name = repo.replace("/", "-")
            cm_path = FIGURES_DIR / f"cm_setfit_{safe_name}.png"
            plot_confusion_matrix(
                metrics["confusion_matrix"],
                EXPECTED_LABELS,
                f"SetFit: {repo}",
                cm_path,
            )

        setfit_cross = compute_cross_repo_score(setfit_repo_metrics)
        all_results["SetFit"] = {
            "repositories": setfit_repo_metrics,
            "cross_repository": setfit_cross,
            "metadata": setfit_metadata,
            "published_reference_macro_f1": PUBLISHED_SETFIT_MACRO_F1,
        }

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = RESULTS_DIR / "benchmark_results.csv"
    export_benchmark_results(all_results, out_json, out_csv)
    logger.info("Saved benchmark results to %s and %s", out_json, out_csv)

    return all_results


def print_comparison_table(results: dict[str, Any]) -> None:
    print("\n" + "=" * 80)
    print("  OFFICIAL BENCHMARK EVALUATION RESULTS (5 REPOSITORIES)")
    print("=" * 80)

    header = f"{'Model':<20} | {'Repo':<24} | {'Accuracy':<9} | {'Macro F1':<9} | {'Weighted F1':<11}"
    print(header)
    print("-" * len(header))

    for model_name, model_data in results.items():
        for repo in EXPECTED_REPOS:
            r = model_data["repositories"].get(repo, {})
            print(
                f"{model_name:<20} | {repo:<24} | "
                f"{r.get('accuracy', 0.0):<9.4f} | "
                f"{r.get('macro_f1', 0.0):<9.4f} | "
                f"{r.get('weighted_f1', 0.0):<11.4f}"
            )
        cross = model_data["cross_repository"]
        print(
            f"{model_name:<20} | {'* OVERALL MEAN *':<24} | "
            f"{cross['mean_accuracy']:<9.4f} | "
            f"{cross['mean_macro_f1']:<9.4f} | "
            f"{cross['mean_weighted_f1']:<11.4f}"
        )
        print("-" * len(header))

    if "SetFit" in results:
        our_setfit = results["SetFit"]["cross_repository"]["mean_macro_f1"]
        print(f"\nPublished SetFit Baseline Reference : {PUBLISHED_SETFIT_MACRO_F1:.4f}")
        print(f"Our Independently Reproduced SetFit : {our_setfit:.4f}")
        delta = our_setfit - PUBLISHED_SETFIT_MACRO_F1
        print(f"Delta                               : {delta:+.4f}")
    print("=" * 80 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run official NLBSE 2024 benchmark")
    parser.add_argument("--skip-dummy", action="store_true", help="Skip Dummy baseline")
    parser.add_argument("--skip-classical", action="store_true", help="Skip Classical TF-IDF+LogReg")
    parser.add_argument("--skip-setfit", action="store_true", help="Skip SetFit training")
    parser.add_argument("--setfit-iterations", type=int, default=1, help="Contrastive pairs per sample for SetFit (default: 1)")
    args = parser.parse_args()

    results = run_benchmark(
        run_dummy=not args.skip_dummy,
        run_classical=not args.skip_classical,
        run_setfit=not args.skip_setfit,
        setfit_iterations=args.setfit_iterations,
    )
    print_comparison_table(results)


if __name__ == "__main__":
    main()
