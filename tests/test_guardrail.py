"""
Offline Unit Tests for Stage 5: NLI Guardrail & Verification Engine
===================================================================
Tests dynamic id2label resolution, citation enforcement, premise window
selection, and thresholding without requiring live model downloads.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from src.generate.schema import Claim
from src.guardrail.nli import NLIGuardrail
from src.guardrail.verify import (
    ClaimVerificationEngine,
    apply_dominance_rule,
    build_candidate_premises,
    build_multi_evidence_premise,
    select_best_evidence_window,
    split_into_sentences,
)


class TestNLIComponents(unittest.TestCase):
    """Test dynamic id2label mapping and verification safeguards."""

    def test_dynamic_id2label_mapping(self):
        """Verify dynamic label mapping correctly resolves indices regardless of ordering."""
        nli = NLIGuardrail(device="cpu")

        # Mock model with reversed id2label order
        mock_model = MagicMock()
        mock_model.config.id2label = {
            0: "CONTRADICTION",
            1: "ENTAILMENT",
            2: "NEUTRAL",
        }
        mock_tok = MagicMock()
        nli._model = mock_model
        nli._tokenizer = mock_tok

        # Simulate _ensure_model_loaded logic
        id2label = mock_model.config.id2label
        for idx_raw, label_raw in id2label.items():
            idx = int(idx_raw)
            label = str(label_raw).lower()
            if "entail" in label:
                nli.entailment_idx = idx
            elif "contra" in label:
                nli.contradiction_idx = idx
            elif "neut" in label:
                nli.neutral_idx = idx

        self.assertEqual(nli.contradiction_idx, 0)
        self.assertEqual(nli.entailment_idx, 1)
        self.assertEqual(nli.neutral_idx, 2)

    def test_missing_or_invalid_citation_never_supported(self):
        """CRITICAL: Any claim without valid evidence IDs can NEVER be SUPPORTED."""
        mock_nli = MagicMock()
        # Even if NLI somehow returns 100% entailment
        mock_nli.predict_pairs.return_value = [{"entailment": 0.99, "contradiction": 0.01, "neutral": 0.0}]

        engine = ClaimVerificationEngine(nli_guardrail=mock_nli, tau_s=0.50, tau_c=0.50)

        # Claim with no citations
        claims = [
            Claim(claim_id="C1", text="Python was created in 1991", evidence_ids=[]),
            Claim(claim_id="C2", text="Flask supports async handlers", evidence_ids=["E_INVALID_999"]),
        ]
        evidence = [{"evidence_id": "E1", "text": "Flask routing guide"}]

        verifications = engine.verify_claims(claims, evidence)

        self.assertEqual(len(verifications), 2)
        for v in verifications:
            self.assertEqual(v["verdict"], "INSUFFICIENT_EVIDENCE")
            self.assertEqual(v["entailment_score"], 0.0)
            self.assertEqual(v["contradiction_score"], 0.0)

        # Model should never have been queried for ungrounded claims!
        mock_nli.predict_pairs.assert_not_called()

    def test_supported_verdict_when_entailment_exceeds_threshold(self):
        mock_nli = MagicMock()
        mock_nli.predict_pairs.return_value = [
            {"entailment": 0.92, "contradiction": 0.02, "neutral": 0.06}
        ]

        engine = ClaimVerificationEngine(nli_guardrail=mock_nli, tau_s=0.50, tau_c=0.50)

        claims = [
            Claim(claim_id="C1", text="Field validator replaces validator in v2", evidence_ids=["E1"])
        ]
        evidence = [{"evidence_id": "E1", "text": "In v2, field_validator replaces validator."}]

        verifications = engine.verify_claims(claims, evidence)
        self.assertEqual(len(verifications), 1)
        self.assertEqual(verifications[0]["verdict"], "SUPPORTED")
        self.assertAlmostEqual(verifications[0]["entailment_score"], 0.92, places=2)

    def test_contradicted_verdict_when_contradiction_exceeds_threshold(self):
        mock_nli = MagicMock()
        mock_nli.predict_pairs.return_value = [
            {"entailment": 0.05, "contradiction": 0.88, "neutral": 0.07}
        ]

        engine = ClaimVerificationEngine(nli_guardrail=mock_nli, tau_s=0.50, tau_c=0.50)

        claims = [
            Claim(claim_id="C1", text="Flask requires Python 2.7", evidence_ids=["E1"])
        ]
        evidence = [{"evidence_id": "E1", "text": "Flask requires Python 3.8 or newer."}]

        verifications = engine.verify_claims(claims, evidence)
        self.assertEqual(len(verifications), 1)
        self.assertEqual(verifications[0]["verdict"], "CONTRADICTED")
        self.assertAlmostEqual(verifications[0]["contradiction_score"], 0.88, places=2)

    def test_strict_dominance_rule(self):
        """Test dominance decision rule edge cases."""
        # 1. Entailment dominates
        self.assertEqual(
            apply_dominance_rule(entailment=0.75, contradiction=0.10, tau_s=0.30, tau_c=0.30),
            "SUPPORTED",
        )
        # 2. Contradiction dominates
        self.assertEqual(
            apply_dominance_rule(entailment=0.10, contradiction=0.75, tau_s=0.30, tau_c=0.30),
            "CONTRADICTED",
        )
        # 3. Both exceed threshold, but contradiction > entailment -> CONTRADICTED
        self.assertEqual(
            apply_dominance_rule(entailment=0.35, contradiction=0.60, tau_s=0.30, tau_c=0.30),
            "CONTRADICTED",
        )
        # 4. Both exceed threshold, but entailment > contradiction -> SUPPORTED
        self.assertEqual(
            apply_dominance_rule(entailment=0.60, contradiction=0.35, tau_s=0.30, tau_c=0.30),
            "SUPPORTED",
        )
        # 5. Tie when both exceed threshold -> INSUFFICIENT_EVIDENCE (neither strictly strictly dominates)
        self.assertEqual(
            apply_dominance_rule(entailment=0.50, contradiction=0.50, tau_s=0.30, tau_c=0.30),
            "INSUFFICIENT_EVIDENCE",
        )
        # 6. Below thresholds -> INSUFFICIENT_EVIDENCE
        self.assertEqual(
            apply_dominance_rule(entailment=0.20, contradiction=0.25, tau_s=0.30, tau_c=0.30),
            "INSUFFICIENT_EVIDENCE",
        )


class TestCandidatePremises(unittest.TestCase):
    """Test candidate premise construction with heading prefixes and windowing."""

    def test_build_candidate_premises_with_heading(self):
        chunk = {
            "heading": "Custom Validation in v2",
            "text": "In Pydantic v2, custom validation is achieved using @field_validator. It replaces the v1 validator decorator. Extra options are available.",
        }
        candidates = build_candidate_premises("custom validation with @field_validator", chunk)
        self.assertGreaterEqual(len(candidates), 1)

        # Full chunk premise
        full_chunk_pair = next(c for c in candidates if c[0] == "full_chunk")
        self.assertTrue(full_chunk_pair[1].startswith("[Custom Validation in v2]"))

        # Sentence window premise (if distinct)
        window_pairs = [c for c in candidates if c[0] == "sentence_window"]
        if window_pairs:
            self.assertTrue(window_pairs[0][1].startswith("[Custom Validation in v2]"))
            self.assertIn("@field_validator", window_pairs[0][1])

    def test_build_multi_evidence_premise(self):
        chunks = [
            {"heading": "Part 1", "text": "Feature A is enabled by passing flag=True."},
            {"heading": "Part 2", "text": "Feature B is required when Feature A is active."},
        ]
        multi = build_multi_evidence_premise("Flag enables Feature A and Feature B", chunks)
        self.assertIsNotNone(multi)
        self.assertIn("[Part 1]", multi)
        self.assertIn("[Part 2]", multi)
        self.assertIn("Feature A", multi)
        self.assertIn("Feature B", multi)


class TestSentenceWindowSelection(unittest.TestCase):
    """Test text window extraction."""

    def test_split_into_sentences(self):
        text = "First sentence. Second sentence! Third sentence? Fourth sentence."
        sents = split_into_sentences(text)
        self.assertEqual(len(sents), 4)

    def test_select_best_window(self):
        text = (
            "Introduction section. Setting up dependencies with pip. "
            "To enable SSL verification pass verify=True to the client. "
            "Finally call client.get with timeout."
        )
        claim = "Pass verify=True to enable SSL verification"
        window = select_best_evidence_window(claim, text, max_sentences=2)
        self.assertIn("verify=True", window)
