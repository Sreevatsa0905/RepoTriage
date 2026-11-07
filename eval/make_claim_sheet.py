"""
Generate Sample Claims Sheet for Optional Human Annotation (B3)
===============================================================
Per RepoTriage_SPEC.md Section 8B:
Generates ~30 claims with cited excerpts, URLs, model verdicts,
and a blank `human_label` column, saving to eval/reports/claims_to_label.csv.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

from src.pipeline import RepoTriagePipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GROUND_TRUTH_FILE = PROJECT_ROOT / "eval" / "retrieval_ground_truth.json"
OUTPUT_CSV = PROJECT_ROOT / "eval" / "reports" / "claims_to_label.csv"


def generate_claim_sheet(
    sample_size: int = 6,
    target_claims: int = 30,
) -> Path:
    """Run pipeline across sample issues and export generated claims to CSV."""
    if not GROUND_TRUTH_FILE.exists():
        raise FileNotFoundError(f"Ground truth file {GROUND_TRUTH_FILE} not found.")

    with open(GROUND_TRUTH_FILE, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)

    # Pick sample evenly across available repositories
    repos = list({item["repo"] for item in ground_truth})
    sampled_queries: list[dict[str, Any]] = []

    for r in repos:
        repo_items = [item for item in ground_truth if item["repo"] == r]
        sampled_queries.extend(repo_items[:2])

    logger.info("Sampled %d queries across %d repos for claim generation", len(sampled_queries), len(repos))

    pipeline = RepoTriagePipeline()
    extracted_rows: list[dict[str, Any]] = []

    for item in sampled_queries:
        if len(extracted_rows) >= target_claims:
            break

        repo = item["repo"]
        query_text = item["query_text"]
        q_num = item["query_number"]

        logger.info("Analyzing %s#%d for claim extraction...", repo, q_num)
        try:
            report = pipeline.analyze(text=query_text, repo=repo, save_trace=False)
            claims = report.get("verified_claims", [])
            evidence_items = {
                e.get("evidence_id"): e for e in report.get("evidence_items", [])
            }

            for c in claims:
                cid = c["claim_id"]
                ctext = c["text"]
                verdict = c["verdict"]
                p_e = c.get("entailment_score", 0.0)
                p_c = c.get("contradiction_score", 0.0)
                best_eid = c.get("best_evidence_id") or ""
                best_excerpt = c.get("best_excerpt", "")

                ev_url = ""
                if best_eid in evidence_items:
                    ev_url = evidence_items[best_eid].get("url") or ""

                extracted_rows.append({
                    "claim_id": f"{repo}#{q_num}_{cid}",
                    "repo": repo,
                    "issue_number": q_num,
                    "claim_text": ctext,
                    "cited_evidence_id": best_eid,
                    "cited_excerpt": best_excerpt.replace("\n", " "),
                    "evidence_url": ev_url,
                    "model_verdict": verdict,
                    "entailment_score": f"{p_e:.4f}",
                    "contradiction_score": f"{p_c:.4f}",
                    "human_label": "",  # Blank for optional manual annotation: SUPPORTED / CONTRADICTED / INSUFFICIENT
                })

                if len(extracted_rows) >= target_claims:
                    break
        except Exception as e:
            logger.warning("Failed analyzing %s#%d: %s", repo, q_num, e)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "claim_id",
        "repo",
        "issue_number",
        "claim_text",
        "cited_evidence_id",
        "cited_excerpt",
        "evidence_url",
        "model_verdict",
        "entailment_score",
        "contradiction_score",
        "human_label",
    ]

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(extracted_rows)

    logger.info("Successfully exported %d claims to %s", len(extracted_rows), OUTPUT_CSV)
    return OUTPUT_CSV


if __name__ == "__main__":
    generate_claim_sheet()
