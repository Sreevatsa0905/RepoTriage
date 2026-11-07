"""
Stage 6.6-D: Automated Synthetic Perturbation Test
==================================================
Constructs 3 balanced classes of test pairs whose ground-truth labels
are known strictly by rule/construction (zero human or LLM labeling):
  (i)   SUPPORTED: Verbatim sentence extracted from evidence chunk.
  (ii)  CONTRADICTED: Same sentence altered by rule (negation, polarity inversion,
        or numeric/version substitution).
  (iii) INSUFFICIENT_EVIDENCE: Sentence from an unrelated issue chunk within the same repo.

Reports:
  - Sample size n per class (balanced ~100 items per class)
  - Confusion matrix and per-class precision / recall / F1
  - Threshold sweep over tau_s in {0.30, 0.50, 0.70, 0.80, 0.90} and tau_c in {0.50, 0.70, 0.80, 0.90}
  - Generates eval/reports/claims_to_label.csv (30 sampled claims) & claims_key.csv
  - Outputs to eval/reports/perturbation_test.json and eval/reports/perturbation_test.md

IMPORTANT LABEL:
"synthetic perturbation test; labels are known by construction; claims are easier
and more templated than real LLM output; not a substitute for human review."
"""

from __future__ import annotations

import csv
import json
import logging
import random
import re
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import src.config
from src.guardrail.nli import NLIGuardrail
from src.guardrail.verify import apply_dominance_rule, split_into_sentences

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "processed" / "repotriage.db"
REPORTS_DIR = PROJECT_ROOT / "eval" / "reports"

PERTURBATION_RULES = [
    (r"\bis not\b", "is"),
    (r"\bis\b", "is not"),
    (r"\bcannot\b", "can"),
    (r"\bcan\b", "cannot"),
    (r"\bshould not\b", "should"),
    (r"\bshould\b", "should not"),
    (r"\bmust not\b", "must"),
    (r"\bmust\b", "must not"),
    (r"\benables\b", "disables"),
    (r"\bdisables\b", "enables"),
    (r"\bsupported\b", "unsupported"),
    (r"\bunsupported\b", "supported"),
    (r"\braises\b", "does not raise"),
    (r"\bdoes not raise\b", "raises"),
    (r"\brequired\b", "optional"),
    (r"\boptional\b", "required"),
    (r"\bTrue\b", "False"),
    (r"\bFalse\b", "True"),
    (r"\bv2\b", "v1"),
    (r"\bv1\b", "v2"),
    (r"\b2\.0\b", "1.0"),
    (r"\b3\.10\b", "2.7"),
]


def perturb_sentence(sentence: str) -> str | None:
    """Alter a sentence by applying the first matching deterministic rule."""
    for pattern, replacement in PERTURBATION_RULES:
        if re.search(pattern, sentence, flags=re.IGNORECASE):
            # Substitute preserving match
            altered = re.sub(pattern, replacement, sentence, count=1, flags=re.IGNORECASE)
            if altered != sentence:
                return altered
    return None


def build_perturbation_dataset(target_count: int = 100) -> list[dict[str, Any]]:
    """Build dataset from real SQLite chunks across repositories."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("""
        SELECT chunk_id, repo, issue_number, heading, url, text
        FROM chunks
        WHERE length(text) > 100
        ORDER BY chunk_id
    """)
    rows = cursor.fetchall()
    conn.close()

    by_repo: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        d = dict(r)
        by_repo.setdefault(d["repo"], []).append(d)

    random.seed(42)
    shuffled_rows = list(rows)
    random.shuffle(shuffled_rows)

    dataset: list[dict[str, Any]] = []
    seen_supported = 0
    seen_contradicted = 0
    seen_nei = 0

    for r in shuffled_rows:
        ch = dict(r)
        if seen_supported >= target_count and seen_contradicted >= target_count and seen_nei >= target_count:
            break

        repo = ch["repo"]
        heading = ch.get("heading") or ""
        prefix = f"[{heading}] " if heading else ""
        premise_text = prefix + ch["text"][:1200]

        sentences = split_into_sentences(ch["text"])
        candidates = [s for s in sentences if 35 <= len(s) <= 200 and not s.startswith("http")]
        if not candidates:
            continue

        base_sent = candidates[0]

        # 1. SUPPORTED
        if seen_supported < target_count:
            dataset.append({
                "pair_id": f"P_SUP_{seen_supported+1}",
                "premise": premise_text,
                "hypothesis": base_sent,
                "true_label": "SUPPORTED",
                "repo": repo,
                "url": ch.get("url", ""),
                "rule_type": "verbatim_extraction",
            })
            seen_supported += 1

        # 2. CONTRADICTED
        if seen_contradicted < target_count:
            for s in candidates:
                perturbed = perturb_sentence(s)
                if perturbed:
                    dataset.append({
                        "pair_id": f"P_CON_{seen_contradicted+1}",
                        "premise": premise_text,
                        "hypothesis": perturbed,
                        "true_label": "CONTRADICTED",
                        "repo": repo,
                        "url": ch.get("url", ""),
                        "rule_type": "rule_inversion",
                    })
                    seen_contradicted += 1
                    break

        # 3. INSUFFICIENT_EVIDENCE (unrelated chunk in same repo)
        if seen_nei < target_count:
            other_chunks = [
                c for c in by_repo.get(repo, [])
                if c["issue_number"] != ch.get("issue_number")
            ]
            if other_chunks:
                other_ch = random.choice(other_chunks)
                other_sents = split_into_sentences(other_ch["text"])
                other_candidates = [s for s in other_sents if 35 <= len(s) <= 200 and not s.startswith("http")]
                if other_candidates:
                    dataset.append({
                        "pair_id": f"P_NEI_{seen_nei+1}",
                        "premise": premise_text,
                        "hypothesis": other_candidates[0],
                        "true_label": "INSUFFICIENT_EVIDENCE",
                        "repo": repo,
                        "url": ch.get("url", ""),
                        "rule_type": "unrelated_same_repo_issue",
                    })
                    seen_nei += 1

    logger.info("Constructed perturbation dataset: %d SUPPORTED, %d CONTRADICTED, %d INSUFFICIENT_EVIDENCE",
                seen_supported, seen_contradicted, seen_nei)
    return dataset


def compute_metrics(
    y_true: list[str],
    y_pred: list[str],
    classes: list[str],
) -> dict[str, Any]:
    n = len(y_true)
    if n == 0:
        return {}

    acc = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp) / n
    cm = {c: {c2: 0 for c2 in classes} for c in classes}
    for yt, yp in zip(y_true, y_pred):
        if yt in cm and yp in cm[yt]:
            cm[yt][yp] += 1

    per_class = {}
    f1s = []
    for c in classes:
        tp = cm[c][c]
        fp = sum(cm[r][c] for r in classes if r != c)
        fn = sum(cm[c][col] for col in classes if col != c)
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
        f1s.append(f1)
        per_class[c] = {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
            "support": sum(cm[c].values()),
        }

    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0
    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": per_class,
        "confusion_matrix": cm,
    }


def main() -> None:
    t0 = time.perf_counter()
    logger.info("Building automated synthetic perturbation dataset...")
    dataset = build_perturbation_dataset(target_count=100)

    # Score with NLI model
    logger.info("Evaluating %d perturbation pairs with DeBERTa-v3...", len(dataset))
    nli = NLIGuardrail()
    premises = [d["premise"] for d in dataset]
    hypotheses = [d["hypothesis"] for d in dataset]
    scores = nli.predict_pairs(premises, hypotheses, batch_size=16)
    nli.unload_model()

    # Store raw scores on dataset items
    for item, sc in zip(dataset, scores):
        item["entailment"] = sc["entailment"]
        item["contradiction"] = sc["contradiction"]
        item["neutral"] = sc["neutral"]

    classes = ["SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE"]
    y_true = [d["true_label"] for d in dataset]

    # Evaluate at default production threshold (0.80 / 0.80) and ANLI threshold (0.30 / 0.30)
    y_pred_prod = [
        apply_dominance_rule(sc["entailment"], sc["contradiction"], tau_s=0.80, tau_c=0.80)
        for sc in scores
    ]
    metrics_prod = compute_metrics(y_true, y_pred_prod, classes)

    y_pred_030 = [
        apply_dominance_rule(sc["entailment"], sc["contradiction"], tau_s=0.30, tau_c=0.30)
        for sc in scores
    ]
    metrics_030 = compute_metrics(y_true, y_pred_030, classes)

    # Threshold sweep
    tau_s_list = [0.30, 0.50, 0.70, 0.80, 0.90]
    tau_c_list = [0.50, 0.70, 0.80, 0.90]
    sweep_results: list[dict[str, Any]] = []

    for ts in tau_s_list:
        for tc in tau_c_list:
            preds = [apply_dominance_rule(sc["entailment"], sc["contradiction"], ts, tc) for sc in scores]
            m = compute_metrics(y_true, preds, classes)
            # Easy/hard false support proxies from dataset:
            # How many true NEI were marked SUPPORTED?
            nei_false_support = sum(1 for yt, yp in zip(y_true, preds) if yt == "INSUFFICIENT_EVIDENCE" and yp == "SUPPORTED")
            con_false_support = sum(1 for yt, yp in zip(y_true, preds) if yt == "CONTRADICTED" and yp == "SUPPORTED")
            sup_pass = sum(1 for yt, yp in zip(y_true, preds) if yt == "SUPPORTED" and yp == "SUPPORTED")

            sweep_results.append({
                "tau_s": ts,
                "tau_c": tc,
                "accuracy": m["accuracy"],
                "macro_f1": m["macro_f1"],
                "supported_pass_rate": round(sup_pass / 100.0, 4),
                "nei_false_support_rate": round(nei_false_support / 100.0, 4),
                "contradicted_false_support_rate": round(con_false_support / 100.0, 4),
            })

    total_time = time.perf_counter() - t0

    # Write claims_to_label.csv (30 sampled claims) & claims_key.csv
    # Sample 12 SUPPORTED, 9 CONTRADICTED, 9 INSUFFICIENT_EVIDENCE based on model verdict
    by_verdict: dict[str, list[dict[str, Any]]] = {}
    for d, p in zip(dataset, y_pred_prod):
        by_verdict.setdefault(p, []).append({**d, "model_verdict": p})

    random.seed(42)
    sheet_sample = []
    sheet_sample.extend(random.sample(by_verdict.get("SUPPORTED", []), min(12, len(by_verdict.get("SUPPORTED", [])))))
    sheet_sample.extend(random.sample(by_verdict.get("CONTRADICTED", []), min(9, len(by_verdict.get("CONTRADICTED", [])))))
    sheet_sample.extend(random.sample(by_verdict.get("INSUFFICIENT_EVIDENCE", []), min(9, len(by_verdict.get("INSUFFICIENT_EVIDENCE", [])))))
    random.shuffle(sheet_sample)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    claims_csv_path = REPORTS_DIR / "claims_to_label.csv"
    key_csv_path = REPORTS_DIR / "claims_key.csv"

    with open(claims_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "claim", "cited_excerpt", "source_url", "human_label"])
        writer.writeheader()
        for idx, row in enumerate(sheet_sample, 1):
            writer.writerow({
                "id": f"CLM_{idx:02d}",
                "claim": row["hypothesis"],
                "cited_excerpt": row["premise"][:300].replace("\n", " "),
                "source_url": row.get("url", ""),
                "human_label": "",
            })

    with open(key_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "model_verdict", "synthetic_ground_truth"])
        writer.writeheader()
        for idx, row in enumerate(sheet_sample, 1):
            writer.writerow({
                "id": f"CLM_{idx:02d}",
                "model_verdict": row["model_verdict"],
                "synthetic_ground_truth": row["true_label"],
            })

    # Save JSON and Markdown reports
    results = {
        "benchmark": "Synthetic Automated Perturbation Test",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_time_seconds": round(total_time, 2),
        "caveat": "synthetic perturbation test; labels are known by construction; claims are easier and more templated than real LLM output; not a substitute for human review.",
        "sample_size": len(dataset),
        "production_thresholds": {"tau_s": 0.80, "tau_c": 0.80, "metrics": metrics_prod},
        "anli_thresholds": {"tau_s": 0.30, "tau_c": 0.30, "metrics": metrics_030},
        "threshold_sweep": sweep_results,
    }

    out_json = REPORTS_DIR / "perturbation_test.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Markdown format
    md_lines = [
        "# RepoTriage Automated Synthetic Perturbation Test Report",
        f"**Timestamp**: {datetime.now(timezone.utc).isoformat()}  ",
        "**Caveat**: *Synthetic perturbation test; labels are known by construction; claims are easier and more templated than real LLM output; not a substitute for human review.*  ",
        f"**Sample**: {len(dataset)} rule-constructed pairs (~100 per class)  ",
        f"**Runtime**: {total_time:.2f} s  ",
        "",
        "---",
        "",
        "## 1. Performance at Production Thresholds (tau_s=0.80, tau_c=0.80)",
        f"- **Accuracy**: **{metrics_prod['accuracy']*100:.2f}%**",
        f"- **Macro-F1**: **{metrics_prod['macro_f1']:.4f}**",
        "",
        "### Per-Class Breakdown (Production Thresholds)",
        "",
        "| Class | Precision | Recall | F1-Score | Support |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ]
    for c in classes:
        pc = metrics_prod["per_class"][c]
        md_lines.append(f"| **{c}** | {pc['precision']:.4f} | {pc['recall']:.4f} | {pc['f1']:.4f} | {pc['support']} |")

    md_lines.extend([
        "",
        "### Confusion Matrix (Rows: Ground Truth, Columns: Predicted)",
        "",
        "| True \\ Pred | SUPPORTED | CONTRADICTED | INSUFFICIENT_EVIDENCE |",
        "| :--- | :---: | :---: | :---: |",
    ])
    cm = metrics_prod["confusion_matrix"]
    for c in classes:
        md_lines.append(f"| **{c}** | {cm[c]['SUPPORTED']} | {cm[c]['CONTRADICTED']} | {cm[c]['INSUFFICIENT_EVIDENCE']} |")

    md_lines.extend([
        "",
        "---",
        "",
        "## 2. Threshold Calibration Sweep",
        "",
        "| tau_s | tau_c | Accuracy | Macro-F1 | Supported Pass Rate | NEI False-Support Rate | Contradicted False-Support Rate |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])
    for row in sweep_results:
        md_lines.append(
            f"| {row['tau_s']:.2f} | {row['tau_c']:.2f} | {row['accuracy']*100:.2f}% | {row['macro_f1']:.4f} | "
            f"{row['supported_pass_rate']*100:.2f}% | {row['nei_false_support_rate']*100:.2f}% | {row['contradicted_false_support_rate']*100:.2f}% |"
        )

    out_md = REPORTS_DIR / "perturbation_test.md"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print("\n" + "=" * 70)
    print("  STAGE 6.6-D: SYNTHETIC PERTURBATION TEST RESULTS")
    print("=" * 70)
    print(f"Total pairs evaluated: {len(dataset)}")
    print(f"Production Thresholds (0.80 / 0.80): Accuracy = {metrics_prod['accuracy']*100:.2f}%, Macro-F1 = {metrics_prod['macro_f1']:.4f}")
    print(f"ANLI Thresholds (0.30 / 0.30)      : Accuracy = {metrics_030['accuracy']*100:.2f}%, Macro-F1 = {metrics_030['macro_f1']:.4f}")
    print(f"Sampled 30 claims to {claims_csv_path}")
    print(f"Saved reports to {out_json} and {out_md}\n")


if __name__ == "__main__":
    main()
