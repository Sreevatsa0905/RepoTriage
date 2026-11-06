"""
4-Way Retrieval Ablation Evaluation Harness for RepoTriage
==========================================================
Evaluates and compares:
1. BM25 only
2. BGE dense only
3. BM25 + BGE with RRF (k=60)
4. RRF + BGE Reranker

Per RepoTriage_SPEC.md Section 8A:
- Temporal filtering: Excludes query issue itself and any chunks created after query created_at.
- Computes: Recall@5, Precision@5, MRR@10, nDCG@10, and mean latency (ms).
- Provides overall metrics and per-repository breakdown.
- Exports results to eval/reports/retrieval_ablation.json and .md.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.retrieval.index import HybridRetriever, RetrievalMethod

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = PROJECT_ROOT / "eval"
GROUND_TRUTH_FILE = EVAL_DIR / "retrieval_ground_truth.json"
REPORTS_DIR = EVAL_DIR / "reports"


def compute_metrics_for_query(
    ranked_chunks: list[dict[str, Any]],
    target_issue_number: int,
) -> dict[str, float]:
    """Compute Recall@5, Precision@5, MRR@10, and nDCG@10 for a single query."""
    # Find the rank of the target issue
    target_rank: int | None = None
    for idx, c in enumerate(ranked_chunks, 1):
        # A hit occurs if the retrieved chunk originates from the target issue or comments on it
        if c.get("issue_number") == target_issue_number:
            target_rank = idx
            break

    # Recall@5: Target retrieved within top 5
    recall_at_5 = 1.0 if (target_rank is not None and target_rank <= 5) else 0.0

    # Precision@5: Target in top 5 / 5
    precision_at_5 = recall_at_5 / 5.0

    # MRR@10: 1 / rank if rank <= 10 else 0.0
    if target_rank is not None and target_rank <= 10:
        mrr_at_10 = 1.0 / target_rank
    else:
        mrr_at_10 = 0.0

    # nDCG@10: 1 / log2(rank + 1) if rank <= 10 else 0.0
    if target_rank is not None and target_rank <= 10:
        ndcg_at_10 = 1.0 / math.log2(target_rank + 1)
    else:
        ndcg_at_10 = 0.0

    return {
        "hit_rank": target_rank if target_rank is not None else -1,
        "recall@5": recall_at_5,
        "precision@5": precision_at_5,
        "mrr@10": mrr_at_10,
        "ndcg@10": ndcg_at_10,
    }


def evaluate_method(
    method: RetrievalMethod,
    ground_truth: list[dict[str, Any]],
    retriever: HybridRetriever,
    top_k: int = 10,
) -> dict[str, Any]:
    """Run evaluation for a single retrieval method across all queries."""
    logger.info("Evaluating method: %s on %d queries...", method, len(ground_truth))

    query_results: list[dict[str, Any]] = []
    latencies: list[float] = []

    per_repo_stats: dict[str, dict[str, list[float]]] = {}

    for idx, item in enumerate(ground_truth, 1):
        repo = item["repo"]
        q_num = item["query_number"]
        q_created = item.get("query_created_at")
        target_num = item["target_issue_number"]
        query_text = item["query_text"]

        if repo not in per_repo_stats:
            per_repo_stats[repo] = {
                "recall@5": [],
                "precision@5": [],
                "mrr@10": [],
                "ndcg@10": [],
                "latency_ms": [],
            }

        # Filter corpus: exclude the query issue itself and any chunks created after query_created
        retriever.reload_corpus(
            repo=repo,
            exclude_issue_number=q_num,
            max_created_at=q_created,
        )

        t0 = time.perf_counter()
        try:
            results = retriever.search(
                query=query_text,
                repo=repo,
                top_k=top_k,
                method=method,
                return_trace=False,
            )
        except Exception as e:
            logger.error("Failed search for query %s#%d: %s", repo, q_num, e)
            results = []
        latency_ms = (time.perf_counter() - t0) * 1000.0

        metrics = compute_metrics_for_query(results, target_num)
        metrics["latency_ms"] = latency_ms

        latencies.append(latency_ms)
        for k in ["recall@5", "precision@5", "mrr@10", "ndcg@10", "latency_ms"]:
            per_repo_stats[repo][k].append(metrics[k])

        query_results.append({
            "query_id": item["query_id"],
            "repo": repo,
            "target": target_num,
            "metrics": metrics,
        })

        if idx % 10 == 0 or idx == len(ground_truth):
            logger.info("  Processed %d / %d queries", idx, len(ground_truth))

    # Aggregate overall metrics
    n = len(ground_truth)
    overall = {
        "recall@5": round(sum(q["metrics"]["recall@5"] for q in query_results) / n, 4),
        "precision@5": round(sum(q["metrics"]["precision@5"] for q in query_results) / n, 4),
        "mrr@10": round(sum(q["metrics"]["mrr@10"] for q in query_results) / n, 4),
        "ndcg@10": round(sum(q["metrics"]["ndcg@10"] for q in query_results) / n, 4),
        "mean_latency_ms": round(sum(latencies) / n, 2),
    }

    # Aggregate per-repo metrics
    by_repo = {}
    for r, stats in per_repo_stats.items():
        count = len(stats["recall@5"])
        by_repo[r] = {
            "query_count": count,
            "recall@5": round(sum(stats["recall@5"]) / count, 4),
            "precision@5": round(sum(stats["precision@5"]) / count, 4),
            "mrr@10": round(sum(stats["mrr@10"]) / count, 4),
            "ndcg@10": round(sum(stats["ndcg@10"]) / count, 4),
            "mean_latency_ms": round(sum(stats["latency_ms"]) / count, 2),
        }

    return {
        "method": method,
        "query_count": n,
        "overall": overall,
        "per_repo": by_repo,
        "queries": query_results,
    }


def run_full_ablation(
    methods: list[RetrievalMethod] | None = None,
    ground_truth_path: Path = GROUND_TRUTH_FILE,
    reports_dir: Path = REPORTS_DIR,
) -> dict[str, Any]:
    """Execute four-way retrieval ablation evaluation."""
    if not ground_truth_path.exists():
        from eval.build_retrieval_set import build_ground_truth
        logger.info("Ground truth file not found. Building it now...")
        build_ground_truth(output_path=ground_truth_path)

    with open(ground_truth_path, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)

    if methods is None:
        methods = ["bm25", "dense", "hybrid", "hybrid_rerank"]

    reports_dir.mkdir(parents=True, exist_ok=True)
    retriever = HybridRetriever()

    ablation_results: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query_count": len(ground_truth),
        "methods": {},
    }

    for m in methods:
        res = evaluate_method(m, ground_truth, retriever)
        ablation_results["methods"][m] = res

    # Save JSON report
    json_path = reports_dir / "retrieval_ablation.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(ablation_results, f, indent=2, ensure_ascii=False)
    logger.info("Saved JSON ablation report to %s", json_path)

    # Generate Markdown Table Report
    md_path = reports_dir / "retrieval_ablation.md"
    md_content = generate_markdown_report(ablation_results)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    logger.info("Saved Markdown ablation report to %s", md_path)

    print("\n" + md_content)
    return ablation_results


def generate_markdown_report(data: dict[str, Any]) -> str:
    """Format evaluation results into a Markdown report table."""
    lines = [
        "# Retrieval Ablation Evaluation Report",
        f"**Timestamp**: {data['timestamp']}  ",
        f"**Queries Evaluated**: {data['query_count']} verified duplicate/cross-reference pairs  ",
        "",
        "## Overall 4-Way Ablation Results",
        "",
        "| Retrieval Method | Recall@5 | Precision@5 | MRR@10 | nDCG@10 | Mean Latency (ms) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]

    method_labels = {
        "bm25": "1. BM25 Only",
        "dense": "2. BGE Dense Only",
        "hybrid": "3. BM25 + BGE (RRF k=60)",
        "hybrid_rerank": "4. RRF + BGE Reranker",
    }

    for m_key in ["bm25", "dense", "hybrid", "hybrid_rerank"]:
        if m_key not in data["methods"]:
            continue
        stats = data["methods"][m_key]["overall"]
        label = method_labels.get(m_key, m_key)
        lines.append(
            f"| **{label}** | {stats['recall@5']:.4f} | {stats['precision@5']:.4f} | {stats['mrr@10']:.4f} | {stats['ndcg@10']:.4f} | {stats['mean_latency_ms']:.2f} ms |"
        )

    lines.append("")
    lines.append("## Per-Repository Breakdown (Recall@5 / MRR@10)")
    lines.append("")
    lines.append("| Repository | Queries | BM25 Recall@5 | Dense Recall@5 | Hybrid Recall@5 | Rerank Recall@5 |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |")

    # Get repos from first available method
    first_m = next(iter(data["methods"].values()))
    repos = sorted(first_m.get("per_repo", {}).keys())

    for r in repos:
        q_count = first_m["per_repo"][r]["query_count"]
        r_bm25 = data["methods"].get("bm25", {}).get("per_repo", {}).get(r, {}).get("recall@5", 0.0)
        r_dense = data["methods"].get("dense", {}).get("per_repo", {}).get(r, {}).get("recall@5", 0.0)
        r_hybrid = data["methods"].get("hybrid", {}).get("per_repo", {}).get(r, {}).get("recall@5", 0.0)
        r_rerank = data["methods"].get("hybrid_rerank", {}).get("per_repo", {}).get(r, {}).get("recall@5", 0.0)
        lines.append(f"| **`{r}`** | {q_count} | {r_bm25:.4f} | {r_dense:.4f} | {r_hybrid:.4f} | {r_rerank:.4f} |")

    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run retrieval ablation evaluation.")
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["bm25", "dense", "hybrid", "hybrid_rerank"],
        default=None,
        help="Specific methods to evaluate (default: all four)",
    )
    args = parser.parse_args()

    run_full_ablation(methods=args.methods)
