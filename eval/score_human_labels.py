"""
Stage 6.6-D: Human Label Scoring Harness
========================================
Scores eval/reports/claims_to_label.csv against eval/reports/claims_key.csv
if human labels are ever provided.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "eval" / "reports"
CLAIMS_CSV = REPORTS_DIR / "claims_to_label.csv"
KEY_CSV = REPORTS_DIR / "claims_key.csv"


def score_human_labels() -> dict[str, Any]:
    if not CLAIMS_CSV.exists() or not KEY_CSV.exists():
        return {"status": "human check: not performed", "reason": "CSV files not found"}

    keys: dict[str, str] = {}
    with open(KEY_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            keys[row["id"]] = row["model_verdict"]

    labeled_rows: list[dict[str, str]] = []
    with open(CLAIMS_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lbl = (row.get("human_label") or "").strip().upper()
            if lbl:
                labeled_rows.append({"id": row["id"], "human_label": lbl, "model_verdict": keys.get(row["id"], "")})

    n_labeled = len(labeled_rows)
    if n_labeled == 0:
        return {
            "status": "human check: not performed",
            "labeled_count": 0,
            "total_count": len(keys),
            "note": "All documents must state 'human check: not performed' unless the CSV actually contains labels.",
        }

    correct = sum(1 for r in labeled_rows if r["human_label"] == r["model_verdict"])
    accuracy = correct / n_labeled

    return {
        "status": "performed",
        "labeled_count": n_labeled,
        "total_count": len(keys),
        "accuracy": round(accuracy, 4),
        "labeled_rows": labeled_rows,
    }


if __name__ == "__main__":
    res = score_human_labels()
    print(json.dumps(res, indent=2))
