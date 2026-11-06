"""
Unit tests for the frozen classifier wrapper and CLI.
"""

from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout

from src.classify.predict import predict_issue_type
from src.cli import main


class TestClassifyAndCLI(unittest.TestCase):

    def test_predict_issue_type_schema(self):
        result = predict_issue_type("Crash on startup", "Traceback NullPointerException")
        self.assertIn("label", result)
        self.assertIn("predicted_category", result)
        self.assertIn("scores", result)
        self.assertIn("confidence", result)
        self.assertIn("threshold", result)
        self.assertIn("needs_review", result)
        self.assertIn("model_version", result)
        self.assertIn("benchmark_metrics", result)

        self.assertIn(result["label"], ["bug", "feature", "question", "uncertain"])
        self.assertEqual(set(result["scores"].keys()), {"bug", "feature", "question"})
        self.assertAlmostEqual(sum(result["scores"].values()), 1.0, places=2)

        # Check benchmark numbers
        benchmarks = result["benchmark_metrics"]
        self.assertEqual(benchmarks["pooled_test_macro_f1"], 0.76)
        self.assertEqual(benchmarks["cold_start_macro_f1"], 0.594)

    def test_uncertain_threshold_behavior(self):
        # With threshold 0.99, almost every prediction becomes uncertain
        result_uncertain = predict_issue_type(
            "Some ambiguous text",
            threshold=0.99,
        )
        self.assertEqual(result_uncertain["label"], "uncertain")
        self.assertTrue(result_uncertain["needs_review"])

        # With threshold 0.10, prediction is accepted
        result_certain = predict_issue_type(
            "Some ambiguous text",
            threshold=0.10,
        )
        self.assertNotEqual(result_certain["label"], "uncertain")
        self.assertFalse(result_certain["needs_review"])

    def test_cli_classify_command(self):
        f = io.StringIO()
        with redirect_stdout(f):
            exit_code = main(["classify", "Feature request: add dark mode support"])
        output = f.getvalue()

        self.assertEqual(exit_code, 0)
        self.assertIn("REPOTRIAGE ISSUE CLASSIFICATION", output)
        self.assertIn("Predicted Type:", output)
        self.assertIn("Pooled Test Macro-F1", output)
        self.assertIn("Cold-Start Macro-F1", output)

    def test_cli_empty_text_fails(self):
        exit_code = main(["classify", "   "])
        self.assertEqual(exit_code, 1)


if __name__ == "__main__":
    unittest.main()
