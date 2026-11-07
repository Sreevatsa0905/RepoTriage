"""
Claim Verification Engine using Multi-Premise NLI Guardrail
===========================================================
Per RepoTriage_SPEC.md & Stage 6.5 Refinements:
- Dominance rule:
    * SUPPORTED if entailment >= tau_s AND entailment > contradiction
    * CONTRADICTED if contradiction >= tau_c AND contradiction > entailment
    * otherwise INSUFFICIENT_EVIDENCE
- Multi-premise evaluation (pipeline and B2):
    (a) Full chunk prefixed with [heading], truncated cleanly to ~512 tokens
    (b) Top-2 sentence window prefixed with [heading]
- Selects the single premise with highest max(entailment, contradiction)
- Tracks winning premise type ('full_chunk' vs 'sentence_window')
- Hard rule: Missing or invalid citations can NEVER be SUPPORTED.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Literal, Sequence

from src.generate.schema import Claim
from src.guardrail.nli import NLIGuardrail

logger = logging.getLogger(__name__)

VerdictType = Literal["SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE"]
# Stage 6.6-B: Calibrated in-domain production thresholds
# Raised from 0.30 to 0.80 to prevent spurious support on tangential overlap
DEFAULT_TAU_S = 0.80
DEFAULT_TAU_C = 0.80

# Dedicated thresholds for the out-of-domain adversarial ANLI benchmark
ANLI_TAU_S = 0.30
ANLI_TAU_C = 0.30


def apply_dominance_rule(
    entailment: float,
    contradiction: float,
    tau_s: float,
    tau_c: float,
) -> VerdictType:
    """
    Apply strict dominance decision rule.
    - SUPPORTED if entailment >= tau_s AND entailment > contradiction
    - CONTRADICTED if contradiction >= tau_c AND contradiction > entailment
    - Otherwise INSUFFICIENT_EVIDENCE
    """
    if entailment >= tau_s and entailment > contradiction:
        return "SUPPORTED"
    elif contradiction >= tau_c and contradiction > entailment:
        return "CONTRADICTED"
    else:
        return "INSUFFICIENT_EVIDENCE"


def split_into_sentences(text: str) -> list[str]:
    """Split text into individual sentences on standard boundaries."""
    if not text or not text.strip():
        return []
    raw_sentences = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    cleaned = [s.strip() for s in raw_sentences if len(s.strip()) > 3]
    return cleaned if cleaned else [text.strip()]


def select_best_evidence_window(
    claim_text: str,
    evidence_text: str,
    max_sentences: int = 2,
) -> str:
    """
    Select the most relevant 1-2 sentence window from evidence text
    based on token overlap with the claim statement.
    """
    sentences = split_into_sentences(evidence_text)
    if len(sentences) <= max_sentences:
        return evidence_text.strip()

    claim_tokens = set(re.findall(r"\w+", claim_text.lower()))
    if not claim_tokens:
        return " ".join(sentences[:max_sentences])

    scores = []
    for s in sentences:
        s_tokens = set(re.findall(r"\w+", s.lower()))
        overlap = len(claim_tokens.intersection(s_tokens))
        scores.append(overlap)

    best_idx = int(scores.index(max(scores))) if scores else 0

    start_idx = max(0, best_idx)
    end_idx = min(len(sentences), start_idx + max_sentences)
    # If at end, grab preceding sentence to maintain window size
    if end_idx - start_idx < max_sentences and start_idx > 0:
        start_idx = max(0, end_idx - max_sentences)

    window = " ".join(sentences[start_idx:end_idx])
    return window.strip()


def build_candidate_premises(
    claim_text: str,
    evidence_chunk: dict[str, Any],
    max_tokens: int = 512,
) -> list[tuple[str, str]]:
    """
    Build candidate premises for in-domain NLI verification:
    (a) Full chunk prefixed with [heading], truncated cleanly to max_tokens (~400 words)
    (b) Top-2 sentence window prefixed with [heading]

    Returns list of tuples: [('full_chunk', premise_a), ('sentence_window', premise_b)]
    """
    heading = (evidence_chunk.get("heading") or "").strip()
    prefix = f"[{heading}] " if heading else ""
    raw_text = (evidence_chunk.get("text") or "").strip()

    # (a) Full chunk (cleanly word-truncated)
    words = raw_text.split()
    truncated_words = words[:400]
    full_chunk_text = prefix + " ".join(truncated_words)

    # (b) Top-2 sentence window
    window_body = select_best_evidence_window(claim_text, raw_text, max_sentences=2)
    window_text = prefix + window_body

    candidates = [
        ("full_chunk", full_chunk_text),
    ]
    # Only add window if meaningfully distinct from full chunk
    if window_body and window_body != raw_text:
        candidates.append(("sentence_window", window_text))

    return candidates


def build_multi_evidence_premise(
    claim_text: str,
    evidence_chunks: Sequence[dict[str, Any]],
    max_tokens: int = 512,
) -> str | None:
    """
    Stage 6.6-I: Build concatenated multi-evidence premise for claims citing 2+ items.
    Takes best window (or full chunk if short) from each cited item (at most 3),
    prefixed with its heading, concatenated and cleanly truncated to ~400 words.
    """
    if len(evidence_chunks) < 2:
        return None

    parts = []
    for chunk in evidence_chunks[:3]:
        heading = (chunk.get("heading") or "").strip()
        prefix = f"[{heading}] " if heading else ""
        raw_text = (chunk.get("text") or "").strip()
        words = raw_text.split()
        if len(words) <= 50:
            part_body = raw_text
        else:
            part_body = select_best_evidence_window(claim_text, raw_text, max_sentences=2)
        parts.append(prefix + part_body)

    combined_text = "\n\n".join(parts)
    combined_words = combined_text.split()
    if len(combined_words) > 400:
        combined_text = " ".join(combined_words[:400])
    return combined_text.strip()


class ClaimVerificationEngine:
    """Orchestrates NLI premise-hypothesis verification for generated claims."""

    def __init__(
        self,
        nli_guardrail: NLIGuardrail | None = None,
        tau_s: float = DEFAULT_TAU_S,
        tau_c: float = DEFAULT_TAU_C,
    ):
        self.nli = nli_guardrail or NLIGuardrail()
        self.tau_s = tau_s
        self.tau_c = tau_c

    def verify_claims(
        self,
        claims: Sequence[Claim],
        evidence_items: Sequence[dict[str, Any]],
        tau_s: float | None = None,
        tau_c: float | None = None,
    ) -> list[dict[str, Any]]:
        """
        Verify each claim against cited evidence items using multi-premise evaluation
        and strict dominance thresholding.
        """
        th_s = tau_s if tau_s is not None else self.tau_s
        th_c = tau_c if tau_c is not None else self.tau_c

        evidence_map: dict[str, dict[str, Any]] = {}
        for idx, item in enumerate(evidence_items, 1):
            eid = item.get("evidence_id") or f"E{idx}"
            evidence_map[eid] = item

        verification_results: list[dict[str, Any]] = []

        # List of items to evaluate: (premise, hypothesis, claim_idx, eid, premise_type, premise_text)
        nli_queue: list[tuple[str, str, int, str, str, str]] = []

        for c_idx, claim in enumerate(claims):
            clean_text = claim.text.strip()
            valid_cites = [eid for eid in claim.evidence_ids if eid in evidence_map]

            # Hard Rule: Missing or invalid citations can NEVER be SUPPORTED
            if not valid_cites:
                verification_results.append({
                    "claim_id": claim.claim_id,
                    "text": clean_text,
                    "evidence_ids": [],
                    "verdict": "INSUFFICIENT_EVIDENCE",
                    "reason": "No valid evidence cited",
                    "entailment_score": 0.0,
                    "contradiction_score": 0.0,
                    "neutral_score": 1.0,
                    "best_evidence_id": None,
                    "best_excerpt": "",
                    "best_premise_type": "none",
                })
                continue

            # For each cited evidence item, construct (full_chunk, sentence_window) premises
            for eid in valid_cites:
                ev_chunk = evidence_map[eid]
                cand_premises = build_candidate_premises(clean_text, ev_chunk)
                for p_type, p_text in cand_premises:
                    nli_queue.append((p_text, clean_text, c_idx, eid, p_type, p_text))

            # Stage 6.6-I: Add multi-evidence combined premise if 2+ cited items
            if len(valid_cites) >= 2:
                multi_premise = build_multi_evidence_premise(
                    clean_text, [evidence_map[eid] for eid in valid_cites[:3]]
                )
                if multi_premise:
                    multi_eid = "+".join(valid_cites[:3])
                    nli_queue.append((multi_premise, clean_text, c_idx, multi_eid, "multi_evidence", multi_premise))

            verification_results.append({
                "claim_id": claim.claim_id,
                "text": clean_text,
                "evidence_ids": valid_cites,
                "verdict": "INSUFFICIENT_EVIDENCE",
                "entailment_score": 0.0,
                "contradiction_score": 0.0,
                "neutral_score": 1.0,
                "best_evidence_id": None,
                "best_excerpt": "",
                "best_premise_type": "none",
            })

        if nli_queue:
            premises = [item[0] for item in nli_queue]
            hypotheses = [item[1] for item in nli_queue]

            scores = self.nli.predict_pairs(premises, hypotheses)

            # Map scores back to each claim: {c_idx: [(score, eid, p_type, p_text), ...]}
            claim_scores_map: dict[int, list[tuple[dict[str, float], str, str, str]]] = {}
            for (p_text, hyp, c_idx, eid, p_type, _), score in zip(nli_queue, scores):
                if c_idx not in claim_scores_map:
                    claim_scores_map[c_idx] = []
                claim_scores_map[c_idx].append((score, eid, p_type, p_text))

            for c_idx, items in claim_scores_map.items():
                # Choose single premise with highest max(entailment, contradiction)
                winning_item = max(
                    items,
                    key=lambda x: max(x[0]["entailment"], x[0]["contradiction"]),
                )
                win_score, win_eid, win_ptype, win_ptext = winning_item

                p_e = win_score["entailment"]
                p_c = win_score["contradiction"]
                p_n = win_score["neutral"]

                # Apply dominance rule to the winning premise ONLY
                verdict = apply_dominance_rule(p_e, p_c, th_s, th_c)

                verification_results[c_idx].update({
                    "verdict": verdict,
                    "entailment_score": round(p_e, 4),
                    "contradiction_score": round(p_c, 4),
                    "neutral_score": round(p_n, 4),
                    "best_evidence_id": win_eid,
                    "best_excerpt": win_ptext,
                    "best_premise_type": win_ptype,
                })

        return verification_results
