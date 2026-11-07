"""
Unit tests for evaluation metrics, cross-repository scoring, and matrix calculations.
"""

import unittest

from src.classifier.config import EXPECTED_LABELS
from src.classifier.evaluate import compute_cross_repo_score, compute_metrics


class TestEvaluation(unittest.TestCase):

    def test_compute_metrics_perfect_predictions(self):
        y_true = ["bug", "feature", "question", "bug", "feature", "question"]
        y_pred = ["bug", "feature", "question", "bug", "feature", "question"]

        metrics = compute_metrics(y_true, y_pred)
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["macro_f1"], 1.0)
        self.assertEqual(metrics["weighted_f1"], 1.0)
        for lbl in EXPECTED_LABELS:
            self.assertEqual(metrics["per_class"][lbl]["precision"], 1.0)
            self.assertEqual(metrics["per_class"][lbl]["recall"], 1.0)
            self.assertEqual(metrics["per_class"][lbl]["f1"], 1.0)

    def test_compute_metrics_known_imperfect(self):
        y_true = ["bug", "bug", "feature", "question"]
        y_pred = ["bug", "feature", "feature", "question"]

        metrics = compute_metrics(y_true, y_pred)
        self.assertEqual(metrics["accuracy"], 0.75)
        # bug: 1 TP, 1 FN -> recall = 0.5, prec = 1.0 -> f1 = 2/3 = 0.6667
        self.assertAlmostEqual(metrics["per_class"]["bug"]["recall"], 0.5, places=3)
        self.assertAlmostEqual(metrics["per_class"]["bug"]["precision"], 1.0, places=3)

    def test_compute_cross_repo_score(self):
        repo_results = {
            "facebook/react": {"macro_f1": 0.90, "weighted_f1": 0.90, "accuracy": 0.90, "macro_precision": 0.90, "macro_recall": 0.90, "per_class": {l: {"precision": 0.9, "recall": 0.9, "f1": 0.9} for l in EXPECTED_LABELS}},
            "tensorflow/tensorflow": {"macro_f1": 0.80, "weighted_f1": 0.80, "accuracy": 0.80, "macro_precision": 0.80, "macro_recall": 0.80, "per_class": {l: {"precision": 0.8, "recall": 0.8, "f1": 0.8} for l in EXPECTED_LABELS}},
            "microsoft/vscode": {"macro_f1": 0.70, "weighted_f1": 0.70, "accuracy": 0.70, "macro_precision": 0.70, "macro_recall": 0.70, "per_class": {l: {"precision": 0.7, "recall": 0.7, "f1": 0.7} for l in EXPECTED_LABELS}},
            "bitcoin/bitcoin": {"macro_f1": 0.60, "weighted_f1": 0.60, "accuracy": 0.60, "macro_precision": 0.60, "macro_recall": 0.60, "per_class": {l: {"precision": 0.6, "recall": 0.6, "f1": 0.6} for l in EXPECTED_LABELS}},
            "opencv/opencv": {"macro_f1": 0.50, "weighted_f1": 0.50, "accuracy": 0.50, "macro_precision": 0.50, "macro_recall": 0.50, "per_class": {l: {"precision": 0.5, "recall": 0.5, "f1": 0.5} for l in EXPECTED_LABELS}},
        }

        cross = compute_cross_repo_score(repo_results)
        # Arithmetic mean: (0.9 + 0.8 + 0.7 + 0.6 + 0.5) / 5 = 0.70
        self.assertAlmostEqual(cross["mean_macro_f1"], 0.70, places=4)
        self.assertAlmostEqual(cross["mean_accuracy"], 0.70, places=4)

    def test_guardrail_metrics_computation(self):
        from eval.run_guardrail_eval import compute_classification_metrics
        y_true = ["SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE", "SUPPORTED"]
        y_pred = ["SUPPORTED", "CONTRADICTED", "SUPPORTED", "SUPPORTED"]
        classes = ["SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE"]

        metrics = compute_classification_metrics(y_true, y_pred, classes)
        self.assertEqual(metrics["accuracy"], 0.75)
        self.assertIn("macro_f1", metrics)
        self.assertIn("confusion_matrix", metrics)
        self.assertEqual(len(metrics["confusion_matrix"]), 3)



if __name__ == "__main__":
    unittest.main()
