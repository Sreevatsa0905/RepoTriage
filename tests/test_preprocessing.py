"""
Unit tests for text preprocessing, input validation, and label mapping.
"""

import unittest

from src.classifier.config import EXPECTED_LABELS
from src.classifier.preprocessing import (
    clean_markdown_and_noise,
    combine_issue_text,
    decode_labels,
    encode_labels,
    validate_issue_input,
)


class TestPreprocessing(unittest.TestCase):

    def test_validate_issue_input_normal(self):
        title, body = validate_issue_input("  Crash on startup  ", " Traceback in main.py ")
        self.assertEqual(title, "Crash on startup")
        self.assertEqual(body, "Traceback in main.py")

    def test_validate_issue_input_none_and_empty(self):
        title, body = validate_issue_input(None, "")
        self.assertEqual(title, "")
        self.assertEqual(body, "")

    def test_validate_issue_input_non_string_types(self):
        title, body = validate_issue_input(12345, True)
        self.assertEqual(title, "12345")
        self.assertEqual(body, "True")

    def test_combine_issue_text(self):
        combined = combine_issue_text("Login fails", "Getting 401 unauthorized")
        self.assertEqual(combined, "Login fails Getting 401 unauthorized")

    def test_combine_issue_text_empty_cases(self):
        self.assertEqual(combine_issue_text("Only title", ""), "Only title")
        self.assertEqual(combine_issue_text("", "Only body"), "Only body")
        self.assertEqual(combine_issue_text("", ""), "")
        self.assertEqual(combine_issue_text(None, None), "")

    def test_clean_markdown_and_noise(self):
        raw = "Line 1\n\n\nLine 2     with   spaces"
        cleaned = clean_markdown_and_noise(raw)
        self.assertEqual(cleaned, "Line 1\nLine 2 with spaces")

    def test_label_encoding_and_decoding(self):
        encoded = encode_labels(["bug", "feature", "question", "BUG", "Feature "])
        self.assertEqual(encoded, [0, 1, 2, 0, 1])
        decoded = decode_labels(encoded[:3])
        self.assertEqual(decoded, ["bug", "feature", "question"])

    def test_unknown_label_raises(self):
        with self.assertRaises(ValueError):
            encode_labels(["unknown_category"])

    def test_unknown_id_raises(self):
        with self.assertRaises(ValueError):
            decode_labels([999])


if __name__ == "__main__":
    unittest.main()
