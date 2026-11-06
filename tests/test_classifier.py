"""
Unit tests for model building, training pipelines, serialization, and inference.
All tests use small synthetic fixtures without downloading external weights.
"""

import tempfile
import unittest
from pathlib import Path

from src.classifier.classical import (
    ClassicalClassifierWrapper,
    build_dummy_baseline,
    build_tfidf_pipeline,
)
from src.classifier.config import EXPECTED_LABELS
from src.classifier.inference import IssueClassifierService, IssueClassificationResult
from src.classifier.train import prepare_repo_data, tune_and_train_classical_repo


class TestClassifier(unittest.TestCase):

    def setUp(self):
        # Small synthetic 3-class dataset for 2 repositories
        self.synthetic_train = [
            {"repo": "repo/a", "title": "Crash on click", "body": "Traceback null pointer", "label": "bug"},
            {"repo": "repo/a", "title": "500 server error", "body": "Fails to return response", "label": "bug"},
            {"repo": "repo/a", "title": "Add dark mode", "body": "Please add dark theme support", "label": "feature"},
            {"repo": "repo/a", "title": "Export to CSV", "body": "New feature request to export data", "label": "feature"},
            {"repo": "repo/a", "title": "How to deploy?", "body": "Where is the documentation for deployment?", "label": "question"},
            {"repo": "repo/a", "title": "Can I use custom auth?", "body": "Is it possible to configure LDAP?", "label": "question"},
            # repo/b
            {"repo": "repo/b", "title": "Memory leak", "body": "OOM crash after 2 hours", "label": "bug"},
            {"repo": "repo/b", "title": "Add webhook support", "body": "Allow webhooks for events", "label": "feature"},
            {"repo": "repo/b", "title": "How to configure port?", "body": "What environment variable sets the port?", "label": "question"},
        ]

    def test_pipeline_word_features(self):
        pipe = build_tfidf_pipeline(feature_mode="word", C=1.0)
        texts = [f"{r['title']} {r['body']}" for r in self.synthetic_train]
        labels = [r["label"] for r in self.synthetic_train]
        pipe.fit(texts, labels)

        preds = pipe.predict(["Server crashed with traceback error", "Please add new export button"])
        self.assertEqual(len(preds), 2)
        for p in preds:
            self.assertIn(p, EXPECTED_LABELS)

    def test_pipeline_word_and_char_features(self):
        pipe = build_tfidf_pipeline(feature_mode="word_char", C=1.0)
        texts = [f"{r['title']} {r['body']}" for r in self.synthetic_train]
        labels = [r["label"] for r in self.synthetic_train]
        pipe.fit(texts, labels)

        probs = pipe.predict_proba(["How to configure the database?"])
        self.assertEqual(probs.shape, (1, 3))
        self.assertAlmostEqual(float(probs.sum()), 1.0, places=5)

    def test_dummy_baseline(self):
        dummy = build_dummy_baseline(strategy="most_frequent")
        dummy.fit(["text 1", "text 2"], ["bug", "bug"])
        preds = dummy.predict(["any new text"])
        self.assertEqual(preds[0], "bug")

    def test_repository_separation(self):
        texts_a, labels_a = prepare_repo_data(self.synthetic_train, "repo/a")
        self.assertEqual(len(texts_a), 6)
        texts_b, labels_b = prepare_repo_data(self.synthetic_train, "repo/b")
        self.assertEqual(len(texts_b), 3)

    def test_leakage_prevention_vocab_fitted_on_train_only(self):
        train_texts = [
            "error crash bug fail exception",
            "feature request add new button support",
            "another error bug traceback crash",
        ]
        train_labels = ["bug", "feature", "bug"]
        test_texts = ["unseen_word_xyz feature request"]

        pipe = build_tfidf_pipeline(feature_mode="word", C=1.0)
        pipe.fit(train_texts, train_labels)

        # Inspect fitted vocabulary
        vocab = pipe.named_steps["features"].vocabulary_
        self.assertNotIn("unseen_word_xyz", vocab)
        self.assertIn("crash", vocab)

    def test_model_serialization_and_inference_service(self):
        pipe = build_tfidf_pipeline(feature_mode="word", C=1.0)
        texts = [f"{r['title']} {r['body']}" for r in self.synthetic_train]
        labels = [r["label"] for r in self.synthetic_train]
        pipe.fit(texts, labels)

        wrapper = ClassicalClassifierWrapper(pipe, model_id="test-model", feature_mode="word", C=1.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test_model.joblib"
            wrapper.save(model_path)
            self.assertTrue(model_path.exists())

            # Load into production inference service
            service = IssueClassifierService.from_saved_model(model_path, model_type="classical")
            self.assertEqual(service.model_identifier, "test-model")

            # Inference test with valid issue
            result = service.classify_issue("App crashes on startup", "Traceback shows ZeroDivisionError")
            self.assertIsInstance(result, IssueClassificationResult)
            self.assertIn(result.category, EXPECTED_LABELS)
            self.assertEqual(set(result.scores.keys()), set(EXPECTED_LABELS))
            self.assertAlmostEqual(sum(result.scores.values()), 1.0, places=2)

            # Inference test with empty / edge-case inputs
            empty_result = service.classify_issue("", "")
            self.assertIn(empty_result.category, EXPECTED_LABELS)

            none_result = service.classify_issue(None, None)
            self.assertIn(none_result.category, EXPECTED_LABELS)


if __name__ == "__main__":
    unittest.main()
