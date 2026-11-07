"""
Claim Verification Engine using Evidence Windows and NLI Guardrail
=================================================================
Per RepoTriage_SPEC.md Section 7 & Stage 5 requirements:
- Splits compound statements into atomic sentences.
- Hard safety rule: any claim with missing or invalid citations is NEVER SUPPORTED.
- Extracts top 1-3 sentence premise windows from cited evidence items.
- Evaluates NLI (premise=evidence, hypothesis=claim).
- Emits verdicts: SUPPORTED (>= tau_s), CONTRADICTED (>= tau_c), else INSUFFICIENT_EVIDENCE.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Literal, Sequence

from src.generate.schema import Claim
from src.guardrail.nli import NLIGuardrail

logger = logging.getLogger(__name__)

VerdictType = Literal["SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE"]
DEFAULT_TAU_S = 0.50
DEFAULT_TAU_C = 0.50


def split_into_sentences(text: str) -> list[str]:
    """Split text into individual sentences on standard boundaries."""
    if not text or not text.strip():
        return []
    # Split on periods, exclamation marks, question marks, or newlines followed by space or capital
    raw_sentences = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    cleaned = [s.strip() for s in raw_sentences if len(s.strip()) > 3]
    return cleaned if cleaned else [text.strip()]


def select_best_evidence_window(
    claim_text: str,
    evidence_text: str,
    max_sentences: int = 3,
) -> str:
    """
    Select the most relevant 1-3 sentence window from evidence text
    based on token overlap with the claim statement.
    """
    sentences = split_into_sentences(evidence_text)
    if len(sentences) <= max_sentences:
        return evidence_text.strip()

    claim_tokens = set(re.findall(r"\w+", claim_text.lower()))
    if not claim_tokens:
        return " ".join(sentences[:max_sentences])

    # Score each sentence by token overlap
    scores = []
    for s in sentences:
        s_tokens = set(re.findall(r"\w+", s.lower()))
        overlap = len(claim_tokens.intersection(s_tokens))
        scores.append(overlap)

    best_idx = int(scores.index(max(scores))) if scores else 0

    # Expand window around best sentence (up to max_sentences contiguous sentences)
    start_idx = max(0, best_idx - 1)
    end_idx = min(len(sentences), start_idx + max_sentences)
    window = " ".join(sentences[start_idx:end_idx])
    return window.strip()


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
        Verify each claim against cited evidence items.

        Parameters
        ----------
        claims : Sequence[Claim]
            List of generated claims.
        evidence_items : Sequence[dict]
            Retrieved evidence chunks with 'evidence_id' (e.g. 'E1'..'En').
        tau_s : float | None
            Entailment threshold for SUPPORTED verdict (defaults to self.tau_s).
        tau_c : float | None
            Contradiction threshold for CONTRADICTED verdict (defaults to self.tau_c).

        Returns
        -------
        list[dict]
            Detailed verification reports for each claim.
        """
        th_s = tau_s if tau_s is not None else self.tau_s
        th_c = tau_c if tau_c is not None else self.tau_c

        evidence_map: dict[str, dict[str, Any]] = {}
        for idx, item in enumerate(evidence_items, 1):
            eid = item.get("evidence_id") or f"E{idx}"
            evidence_map[eid] = item

        verification_results: list[dict[str, Any]] = []

        # Pairs to evaluate via NLI: (premise, hypothesis, claim_idx, eid, window)
        nli_queue: list[tuple[str, str, int, str, str]] = []

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
                })
                continue

            # For each cited evidence item, extract best premise window
            for eid in valid_cites:
                ev_chunk = evidence_map[eid]
                ev_text = ev_chunk.get("text", "")
                window = select_best_evidence_window(clean_text, ev_text, max_sentences=3)
                nli_queue.append((window, clean_text, c_idx, eid, window))

            # Placeholder to be filled after NLI batch scoring
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
            })

        if nli_queue:
            premises = [item[0] for item in nli_queue]
            hypotheses = [item[1] for item in nli_queue]

            scores = self.nli.predict_pairs(premises, hypotheses)

            # Map scores back to each claim, taking the maximum entailment or contradiction
            claim_scores_map: dict[int, list[tuple[dict[str, float], str, str]]] = {}
            for (prem, hypo, c_idx, eid, window), score in zip(nli_queue, scores):
                if c_idx not in claim_scores_map:
                    claim_scores_map[c_idx] = []
                claim_scores_map[c_idx].append((score, eid, window))

            for c_idx, items in claim_scores_map.items():
                # Find maximum contradiction and maximum entailment across cited windows
                best_item = max(items, key=lambda x: (x[0]["contradiction"], x[0]["entailment"]))
                max_contra_item = max(items, key=lambda x: x[0]["contradiction"])
                max_entail_item = max(items, key=lambda x: x[0]["entailment"])

                p_c = max_contra_item[0]["contradiction"]
                p_e = max_entail_item[0]["entailment"]

                if p_c >= th_c:
                    verdict: VerdictType = "CONTRADICTED"
                    winning = max_contra_item
                elif p_e >= th_s:
                    verdict: VerdictType = "SUPPORTED"
                    winning = max_entail_item
                else:
                    verdict: VerdictType = "INSUFFICIENT_EVIDENCE"
                    winning = max_entail_item

                verification_results[c_idx].update({
                    "verdict": verdict,
                    "entailment_score": round(p_e, 4),
                    "contradiction_score": round(p_c, 4),
                    "neutral_score": round(winning[0]["neutral"], 4),
                    "best_evidence_id": winning[1],
                    "best_excerpt": winning[2],
                })

        return verification_results
