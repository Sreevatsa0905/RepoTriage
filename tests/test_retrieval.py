"""
Unit tests for Stage 2 Retrieval Components:
BM25 code tokenizer, RRF fusion, and retrieval ordering.
Offline tests - no real network or external model downloads required.
"""

from __future__ import annotations

import unittest
import numpy as np

from src.retrieval.bm25 import BM25Retriever, tokenize_code_text
from src.retrieval.fusion import DEFAULT_RRF_K, reciprocal_rank_fusion


class TestBM25Tokenizer(unittest.TestCase):
    """Test the code-aware BM25 tokenizer."""

    def test_camel_case_identifiers(self):
        tokens = tokenize_code_text("ValidationError in BaseModel while parsing")
        self.assertIn("validationerror", tokens)
        self.assertIn("validation", tokens)
        self.assertIn("error", tokens)
        self.assertIn("basemodel", tokens)
        self.assertIn("base", tokens)
        self.assertIn("model", tokens)
        self.assertNotIn("in", tokens)  # stopword

    def test_snake_case_identifiers(self):
        tokens = tokenize_code_text("calling custom_validator with check_fields=True")
        self.assertIn("custom_validator", tokens)
        self.assertIn("custom", tokens)
        self.assertIn("validator", tokens)
        self.assertIn("check_fields", tokens)
        self.assertIn("check", tokens)
        self.assertIn("fields", tokens)

    def test_version_numbers_and_paths(self):
        tokens = tokenize_code_text("In pydantic v2.0 and python 3.12, check src/models/user.py")
        self.assertIn("v2.0", tokens)
        self.assertIn("3.12", tokens)
        self.assertIn("src/models/user.py", tokens)

    def test_urls(self):
        tokens = tokenize_code_text("See https://github.com/encode/httpx/issues/1042 for details")
        self.assertTrue(any("https://github.com/encode/httpx/issues/1042" in t for t in tokens))

    def test_empty_and_whitespace(self):
        self.assertEqual(tokenize_code_text(""), [])
        self.assertEqual(tokenize_code_text("   \n\t  "), [])


class TestReciprocalRankFusion(unittest.TestCase):
    """Test RRF score computation, ranking, and deduplication."""

    def test_rrf_scoring_and_ordering(self):
        bm25_results = [
            {"chunk_id": "chunk_A", "text": "Doc A", "bm25_score": 10.0, "bm25_rank": 1},
            {"chunk_id": "chunk_B", "text": "Doc B", "bm25_score": 8.0, "bm25_rank": 2},
            {"chunk_id": "chunk_C", "text": "Doc C", "bm25_score": 5.0, "bm25_rank": 3},
        ]
        dense_results = [
            {"chunk_id": "chunk_B", "text": "Doc B", "dense_score": 0.95, "dense_rank": 1},
            {"chunk_id": "chunk_D", "text": "Doc D", "dense_score": 0.85, "dense_rank": 2},
            {"chunk_id": "chunk_A", "text": "Doc A", "dense_score": 0.75, "dense_rank": 3},
        ]

        fused = reciprocal_rank_fusion(bm25_results, dense_results, k=60, top_k=10)

        # Mathematical RRF scores:
        # chunk_A: 1/(60+1) + 1/(60+3) = 1/61 + 1/63 = 0.016393 + 0.015873 = 0.032266
        # chunk_B: 1/(60+2) + 1/(60+1) = 1/62 + 1/61 = 0.016129 + 0.016393 = 0.032522
        # chunk_D: 1/(60+2) = 0.016129
        # chunk_C: 1/(60+3) = 0.015873
        # chunk_B has highest score (0.032522) -> rank 1!
        # chunk_A has 2nd score (0.032266) -> rank 2!

        self.assertEqual(len(fused), 4)
        self.assertEqual(fused[0]["chunk_id"], "chunk_B")
        self.assertEqual(fused[0]["rrf_rank"], 1)
        self.assertEqual(fused[1]["chunk_id"], "chunk_A")
        self.assertEqual(fused[1]["rrf_rank"], 2)

        # Check preserved method ranks
        self.assertEqual(fused[0]["bm25_rank"], 2)
        self.assertEqual(fused[0]["dense_rank"], 1)

    def test_rrf_deduplication(self):
        bm25 = [{"chunk_id": "X", "text": "X text", "bm25_score": 5.0, "bm25_rank": 1}]
        dense = [{"chunk_id": "X", "text": "X text", "dense_score": 0.9, "dense_rank": 1}]
        fused = reciprocal_rank_fusion(bm25, dense, k=60)
        self.assertEqual(len(fused), 1)
        self.assertEqual(fused[0]["chunk_id"], "X")
        expected_score = (1.0 / 61) + (1.0 / 61)
        self.assertAlmostEqual(fused[0]["rrf_score"], expected_score, places=6)


class TestBM25Retriever(unittest.TestCase):
    """Test BM25 search over synthetic chunks."""

    def test_bm25_search_exact_match(self):
        chunks = [
            {"chunk_id": "c1", "text": "FastAPI dependency injection in route handler"},
            {"chunk_id": "c2", "text": "Pydantic custom validator on Optional field"},
            {"chunk_id": "c3", "text": "HTTPX connection pool keepalive timeout error"},
        ]
        retriever = BM25Retriever(chunks)
        results = retriever.search("keepalive timeout in httpx", top_k=2)

        self.assertGreaterEqual(len(results), 1)
        self.assertEqual(results[0]["chunk_id"], "c3")
        self.assertGreater(results[0]["bm25_score"], 0.0)
        self.assertEqual(results[0]["bm25_rank"], 1)


if __name__ == "__main__":
    unittest.main()
