"""
Offline Unit Tests for Stage 4: Structured LLM Generation
=========================================================
Tests Pydantic schemas, prompt formatting, 1-retry handling,
and evidence ID validation using mocked HTTP responses.
"""

from __future__ import annotations

import io
import json
import unittest
from unittest.mock import MagicMock, patch

from src.generate.llm import OllamaClient
from src.generate.prompts import build_analysis_prompt, format_evidence_block
from src.generate.schema import AnalysisResult, Claim


class TestGenerationSchemas(unittest.TestCase):
    """Test Pydantic schema validation for AnalysisResult and Claim."""

    def test_valid_schema_instance(self):
        data = {
            "issue_summary": "Pydantic validator fails on Optional fields.",
            "issue_type": "bug",
            "affected_components": ["pydantic.validator", "pydantic.main"],
            "observed_symptoms": ["ValidationError raised unexpectedly"],
            "environment_details": "pydantic 2.0, python 3.12",
            "reproduction_steps": ["field: Optional[int] = None", "@field_validator('field')"],
            "missing_information": [],
            "likely_causes": ["Validator runs before default None is assigned"],
            "recommended_diagnostics": ["Add print statements in pre-validator"],
            "recommended_next_steps": ["Use mode='before' with None check"],
            "claims": [
                {"claim_id": "C1", "text": "Validators run on default values if validate_default=True", "evidence_ids": ["E1"]},
                {"claim_id": "C2", "text": "In v2, field_validator replaces validator", "evidence_ids": ["E2"]},
            ],
            "limitations": "Tested against local corpus.",
        }
        res = AnalysisResult.model_validate(data)
        self.assertEqual(res.issue_type, "bug")
        self.assertEqual(len(res.claims), 2)
        self.assertEqual(res.claims[0].evidence_ids, ["E1"])

    def test_invalid_issue_type_rejected(self):
        data = {
            "issue_summary": "Some summary",
            "issue_type": "not_a_valid_type",
            "claims": [],
        }
        with self.assertRaises(Exception):
            AnalysisResult.model_validate(data)


class TestPromptBuilder(unittest.TestCase):
    """Test prompt building and evidence formatting."""

    def test_format_evidence_block(self):
        items = [
            {
                "evidence_id": "E1",
                "source_type": "issue",
                "repo": "pydantic/pydantic",
                "heading": "Issue #123: Optional field error",
                "url": "https://github.com/pydantic/pydantic/issues/123",
                "text": "This is the body text of issue 123.",
            }
        ]
        ev_str = format_evidence_block(items)
        self.assertIn("EVIDENCE [E1]", ev_str)
        self.assertIn("Issue #123", ev_str)
        self.assertIn("This is the body text", ev_str)

    def test_build_analysis_prompt_contains_critical_tokens(self):
        prompt = build_analysis_prompt(
            issue_text="My app crashed with KeyError",
            repo="pallets/flask",
            evidence_items=[{"evidence_id": "E1", "text": "Fix flask key error"}],
            predicted_type="bug",
        )
        self.assertIn("pallets/flask", prompt)
        self.assertIn("KeyError", prompt)
        self.assertIn("E1", prompt)
        self.assertIn("Classifier Preliminary Type: bug", prompt)


class TestOllamaClientMocked(unittest.TestCase):
    """Test OllamaClient with mocked urllib responses (offline)."""

    def setUp(self):
        self.client = OllamaClient(base_url="http://mock-ollama:11434")

    @patch("urllib.request.urlopen")
    def test_successful_generation_and_evidence_id_sanitization(self, mock_urlopen):
        raw_output = {
            "message": {
                "content": json.dumps({
                    "issue_summary": "Flask routing error on trailing slash.",
                    "issue_type": "bug",
                    "affected_components": ["flask.app", "flask.routing"],
                    "observed_symptoms": ["404 Not Found instead of 308 redirect"],
                    "environment_details": "Flask 3.0",
                    "reproduction_steps": ["@app.route('/path/')"],
                    "missing_information": [],
                    "likely_causes": ["strict_slashes configuration"],
                    "recommended_diagnostics": ["Check url_map rules"],
                    "recommended_next_steps": ["Set strict_slashes=False"],
                    "claims": [
                        {"claim_id": "C1", "text": "strict_slashes defaults to True", "evidence_ids": ["E1", "E999_FAKE"]},
                        {"claim_id": "C2", "text": "Redirects issue 308 response", "evidence_ids": ["E2"]},
                    ],
                    "limitations": "Local triage.",
                })
            },
            "prompt_eval_count": 150,
            "eval_count": 200,
        }
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(raw_output).encode("utf-8")
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response

        evidence = [
            {"evidence_id": "E1", "text": "strict_slashes doc"},
            {"evidence_id": "E2", "text": "redirect logic"},
        ]

        result, trace = self.client.generate_analysis(
            issue_text="Flask trailing slash error",
            repo="pallets/flask",
            evidence_items=evidence,
        )

        self.assertEqual(result.issue_type, "bug")
        self.assertEqual(len(result.claims), 2)
        # Verify fake evidence ID E999_FAKE was discarded!
        self.assertEqual(result.claims[0].evidence_ids, ["E1"])
        self.assertEqual(result.claims[1].evidence_ids, ["E2"])
        self.assertFalse(trace["retried"])
        self.assertEqual(trace["num_claims"], 2)

    @patch("urllib.request.urlopen")
    def test_retry_on_first_failure(self, mock_urlopen):
        # 1st call returns invalid json string, 2nd call returns valid json
        resp1 = MagicMock()
        resp1.read.return_value = b"Not valid json"
        resp1.__enter__.return_value = resp1

        valid_json = {
            "issue_summary": "Recovered summary",
            "issue_type": "question",
            "affected_components": [],
            "observed_symptoms": [],
            "environment_details": "not established",
            "reproduction_steps": [],
            "missing_information": [],
            "likely_causes": [],
            "recommended_diagnostics": [],
            "recommended_next_steps": [],
            "claims": [],
            "limitations": "Retried successfully.",
        }
        resp2 = MagicMock()
        resp2.read.return_value = json.dumps({"message": {"content": json.dumps(valid_json)}}).encode("utf-8")
        resp2.__enter__.return_value = resp2

        mock_urlopen.side_effect = [resp1, resp2]

        result, trace = self.client.generate_analysis(
            issue_text="How to do async in httpx?",
            repo="encode/httpx",
            evidence_items=[],
        )

        self.assertEqual(result.issue_type, "question")
        self.assertTrue(trace["retried"])
