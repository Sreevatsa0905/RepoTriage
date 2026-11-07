"""
Unit Tests for RepoTriage FastAPI Backend (Stage 7)
===================================================
Tests all REST endpoints using FastAPI TestClient with offline mocks.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from src.api.main import app, _analysis_lock


class TestRepoTriageAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_health_endpoint(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("status"), "healthy")
        self.assertIn("gpu", data)
        self.assertIn("database", data)
        self.assertIn("models", data)

    def test_repositories_endpoint(self):
        response = self.client.get("/api/repositories")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("repositories", data)
        self.assertIsInstance(data["repositories"], list)

    def test_evaluation_endpoint(self):
        response = self.client.get("/api/evaluation")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("timestamp", data)
        self.assertIn("retrieval_ablation", data)
        self.assertIn("guardrail_evaluation", data)
        self.assertEqual(data.get("human_check_status"), "not performed")

    def test_trace_id_validation_path_traversal(self):
        # Invalid trace_ids with forbidden characters or traversal attempts
        invalid_ids = [
            "..secret",
            "trace..secret",
            "trace$123",
            "trace;evil",
            "trace!special",
            "trace space",
            "trace@domain",
        ]
        for tid in invalid_ids:
            response = self.client.get(f"/api/analysis/{tid}")
            self.assertEqual(response.status_code, 400, f"Expected 400 for '{tid}', got {response.status_code}")
            self.assertIn("Invalid trace_id", response.json()["detail"])

    def test_trace_id_validation_not_found(self):
        # Valid format should return 404 when file does not exist
        valid_id = "trace_nonexistent_99999"
        response = self.client.get(f"/api/analysis/{valid_id}")
        self.assertEqual(response.status_code, 404)
        self.assertIn("not found", response.json()["detail"].lower())

    @patch("src.api.main.get_pipeline")
    def test_analyze_mocked(self, mock_get_pipeline):
        mock_pipeline = MagicMock()
        mock_pipeline.analyze.return_value = {
            "query": "mock test issue",
            "repo": "pydantic/pydantic",
            "classifier": {"predicted_label": "bug", "confidence": 0.88, "needs_review": False},
            "synthesis": {"title": "Test Title", "claims": []},
            "verification": {"claims": []},
            "timings": {"total_pipeline_ms": 120.0},
        }
        mock_get_pipeline.return_value = mock_pipeline

        payload = {"text": "mock issue text", "repo": "pydantic/pydantic"}
        response = self.client.post("/api/analyze", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["repo"], "pydantic/pydantic")
        self.assertEqual(data["classifier"]["predicted_label"], "bug")

    def test_analyze_empty_request_rejected(self):
        payload = {"text": "   "}
        response = self.client.post("/api/analyze", json=payload)
        self.assertEqual(response.status_code, 400)

    def test_analyze_429_when_locked(self):
        # Simulate lock already acquired by another request
        with patch.object(_analysis_lock, "locked", return_value=True):
            payload = {"text": "some issue query", "repo": "pydantic/pydantic"}
            response = self.client.post("/api/analyze", json=payload)
            self.assertEqual(response.status_code, 429)
            self.assertIn("Pipeline busy", response.json()["detail"])

    @patch("src.api.main.get_retriever")
    def test_search_mocked(self, mock_get_retriever):
        mock_retriever = MagicMock()
        mock_retriever.search.return_value = [
            {"chunk_id": "c1", "score": 0.95, "title": "Result 1"}
        ]
        mock_get_retriever.return_value = mock_retriever

        payload = {"query": "authentication error", "repo": "encode/httpx", "top_k": 3}
        response = self.client.post("/api/search", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["count"], 1)
        self.assertEqual(data["results"][0]["chunk_id"], "c1")


if __name__ == "__main__":
    unittest.main()
