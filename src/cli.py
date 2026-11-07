"""
RepoTriage CLI
==============
Main command-line interface for RepoTriage.

Stage 0 Deliverable:
    python -m src.cli classify "some issue text"
"""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from src.classify.predict import DEFAULT_UNCERTAIN_THRESHOLD, predict_issue_type


def handle_classify(args: argparse.Namespace) -> int:
    """Execute issue classification with uncertainty thresholding."""
    text = args.text.strip()
    if not text:
        print("Error: Empty issue text provided for classification.", file=sys.stderr)
        return 1

    body = args.body or ""
    threshold = args.threshold

    result = predict_issue_type(title=text, body=body, threshold=threshold)

    label = result["label"]
    confidence = result["confidence"]
    scores = result["scores"]
    needs_review = result["needs_review"]
    benchmarks = result["benchmark_metrics"]

    print("\n" + "=" * 65)
    print("  REPOTRIAGE ISSUE CLASSIFICATION")
    print("=" * 65)
    print(f"  Input Text   : {text[:80]}{'...' if len(text) > 80 else ''}")
    print(f"  Predicted Type: {label.upper()}")
    print(f"  Confidence   : {confidence:.4f} (Threshold: {threshold:.2f})")
    print(f"  Needs Review : {'YES (low confidence)' if needs_review else 'NO'}")
    print(f"  Model ID     : {result['model_version']}")

    print("\n  Class Confidence Scores:")
    for cat in ["bug", "feature", "question"]:
        score = scores.get(cat, 0.0)
        bar = "█" * int(score * 30)
        print(f"    {cat:<10} : {score:.4f}  {bar}")

    print("\n  Benchmark Reference Metrics:")
    print(f"    • Pooled Test Macro-F1   : {benchmarks['pooled_test_macro_f1']:.2f} (repositories seen in training)")
    print(f"    • Cold-Start Macro-F1    : {benchmarks['cold_start_macro_f1']:.3f} (leave-one-repository-out cross-validation)")
    print("=" * 65 + "\n")

    return 0


def handle_ingest(args: argparse.Namespace) -> int:
    """Execute ingestion of issues, comments, docs, and chunking."""
    from src.ingest import SEED_REPOSITORIES, ingest_repository

    repos_to_ingest: list[str] = []
    if args.all:
        repos_to_ingest = list(SEED_REPOSITORIES)
    elif args.repo:
        repos_to_ingest = [args.repo]
    else:
        print("Error: Specify --repo <owner/repo> or --all.", file=sys.stderr)
        return 1

    for repo_name in repos_to_ingest:
        try:
            ingest_repository(
                repo_name,
                target_issues=args.target_issues,
                force_refresh=args.force,
            )
        except Exception as e:
            print(f"Error ingesting repository '{repo_name}': {e}", file=sys.stderr)
            return 1

    return 0


def _normalize_repo_arg(repo_arg: str | None) -> str | None:
    if not repo_arg:
        return None
    repo_arg = repo_arg.strip()
    mapping = {
        "flask": "pallets/flask",
        "httpx": "encode/httpx",
        "pydantic": "pydantic/pydantic",
        "requests": "psf/requests",
        "pallets/flask": "pallets/flask",
        "encode/httpx": "encode/httpx",
        "pydantic/pydantic": "pydantic/pydantic",
        "psf/requests": "psf/requests",
    }
    return mapping.get(repo_arg.lower(), repo_arg)


def handle_search(args: argparse.Namespace) -> int:
    """Execute hybrid retrieval search and print ranked issues and docs."""
    from src.retrieval.index import HybridRetriever

    query = args.query.strip()
    if not query:
        print("Error: Empty search query provided.", file=sys.stderr)
        return 1

    repo = _normalize_repo_arg(args.repo)
    method = args.method
    top_k = args.top_k

    print("\n" + "=" * 75)
    print(f"  REPOTRIAGE RETRIEVAL SEARCH")
    print("=" * 75)
    print(f"  Query  : {query}")
    print(f"  Repo   : {repo or 'All Repositories'}")
    print(f"  Method : {method}")
    print("-" * 75)

    try:
        retriever = HybridRetriever(repo=repo)
        results = retriever.search(query, repo=repo, top_k=top_k, method=method, return_trace=True)
    except Exception as e:
        print(f"Search failed: {e}", file=sys.stderr)
        return 1

    hits = results["results"]
    issue_hits = [h for h in hits if h["source_type"] in ("issue", "comment")]
    doc_hits = [h for h in hits if h["source_type"] == "doc"]

    print("\n  TOP SIMILAR ISSUES & DISCUSSIONS:")
    if not issue_hits:
        print("    (No issue matches found)")
    else:
        for h in issue_hits:
            score = h.get("rerank_score") or h.get("rrf_score") or h.get("dense_score") or h.get("bm25_score") or 0.0
            eid = h.get("evidence_id", "Hit")
            num = h.get("issue_number", "?")
            heading = h.get("heading", "")
            stype = h.get("source_type", "issue").upper()
            url = h.get("url", "")
            preview = h.get("text", "").split("\n\n", 1)[-1].strip()[:140].replace("\n", " ")
            print(f"    • [{eid}] #{num} ({stype}) - {heading}")
            print(f"        Score: {score:.4f} | URL: {url}")
            print(f"        Excerpt: {preview}...\n")

    print("  TOP RELEVANT DOCUMENTATION EXCERPTS:")
    if not doc_hits:
        print("    (No doc matches found)")
    else:
        for h in doc_hits:
            score = h.get("rerank_score") or h.get("rrf_score") or h.get("dense_score") or h.get("bm25_score") or 0.0
            eid = h.get("evidence_id", "Doc")
            fpath = h.get("file_path", "")
            heading = h.get("heading", "")
            url = h.get("url", "")
            preview = h.get("text", "").split("\n\n", 1)[-1].strip()[:140].replace("\n", " ")
            print(f"    • [{eid}] {fpath} > {heading}")
            print(f"        Score: {score:.4f} | URL: {url}")
            print(f"        Excerpt: {preview}...\n")

    print("-" * 75)
    print(f"  Latency: {results['total_latency_ms']} ms  | Timings: {results['stage_timings_ms']}")
    print("=" * 75 + "\n")
    return 0


def handle_analyze(args: argparse.Namespace) -> int:
    """Execute end-to-end issue triage, structured generation, and NLI claim verification."""
    from src.pipeline import RepoTriagePipeline

    text = (args.text or "").strip()
    issue_url = (args.issue or "").strip() or None
    repo = _normalize_repo_arg(args.repo)

    if not text and not issue_url:
        print("Error: Provide issue text as a positional argument or use --issue <url>.", file=sys.stderr)
        return 1

    print("\n" + "=" * 80)
    print("  REPOTRIAGE: LOCAL GITHUB ISSUE INTELLIGENCE & RESOLUTION COPILOT")
    print("=" * 80)
    if issue_url:
        print(f"  Issue URL : {issue_url}")
    if repo:
        print(f"  Repository: {repo}")
    if text:
        preview = text.split("\n", 1)[0][:80]
        print(f"  Query Text: {preview}...")
    print("-" * 80)
    print("  Executing: Classification -> Hybrid Retrieval -> Ollama Synthesis -> NLI Guardrail...")

    pipeline = RepoTriagePipeline()
    try:
        report = pipeline.analyze(text=text, repo=repo, issue_url=issue_url, save_trace=True)
    except Exception as e:
        print(f"\nAnalysis failed: {e}", file=sys.stderr)
        return 1

    # 1. Classification Output
    clf = report["classification"]
    label = clf["label"].upper()
    conf = clf["confidence"]
    print("\n" + "─" * 80)
    print(f"  1. PREDICTED ISSUE TYPE: [{label}] (Confidence: {conf:.4f})")
    print("─" * 80)
    score_parts = [f"{k}: {v:.3f}" for k, v in clf["scores"].items()]
    print(f"     Scores: {' | '.join(score_parts)}")
    benchmarks = clf["benchmark_metrics"]
    print(f"     Benchmarks: Pooled Macro-F1: {benchmarks['pooled_test_macro_f1']} | Cold-Start: {benchmarks['cold_start_macro_f1']}")

    # 2. Similar Issues
    issues = report.get("similar_issues", [])
    print("\n" + "─" * 80)
    print(f"  2. TOP SIMILAR PAST ISSUES ({len(issues)} retrieved)")
    print("─" * 80)
    if not issues:
        print("     (No past issue evidence retrieved)")
    else:
        for it in issues:
            eid = it["evidence_id"]
            num = it.get("number", "?")
            head = it.get("heading", "")
            score = it.get("score", 0.0)
            url = it.get("url") or ""
            print(f"     • [{eid}] #{num}: {head}")
            print(f"       Score: {score:.4f} | URL: {url}")

    # 3. Relevant Docs
    docs = report.get("doc_excerpts", [])
    print("\n" + "─" * 80)
    print(f"  3. RELEVANT DOCUMENTATION EXCERPTS ({len(docs)} retrieved)")
    print("─" * 80)
    if not docs:
        print("     (No doc excerpts retrieved)")
    else:
        for d in docs:
            eid = d["evidence_id"]
            fpath = d.get("file_path", "")
            head = d.get("heading", "")
            score = d.get("score", 0.0)
            print(f"     • [{eid}] {fpath} > {head} (Score: {score:.4f})")

    # 4. Structured Analysis
    ana = report["analysis"]
    print("\n" + "─" * 80)
    print("  4. STRUCTURED ISSUE TRIAGE & RESOLUTION")
    print("─" * 80)
    print(f"  Summary: {ana.get('issue_summary', '')}")
    if ana.get("affected_components"):
        print(f"  Affected Components: {', '.join(ana['affected_components'])}")
    if ana.get("observed_symptoms"):
        print(f"  Observed Symptoms: {', '.join(ana['observed_symptoms'])}")
    print(f"  Environment: {ana.get('environment_details', 'not established')}")

    if ana.get("likely_causes"):
        print("\n  Likely Technical Causes:")
        for cause in ana["likely_causes"]:
            print(f"    • {cause}")

    if ana.get("recommended_diagnostics"):
        print("\n  Recommended Diagnostic Steps:")
        for diag in ana["recommended_diagnostics"]:
            print(f"    • {diag}")

    if ana.get("recommended_next_steps"):
        print("\n  Recommended Next Steps & Fix:")
        for nxt in ana["recommended_next_steps"]:
            print(f"    • {nxt}")

    if ana.get("limitations"):
        print(f"\n  Limitations: {ana['limitations']}")

    # 5. Verified Claims List with NLI Guardrail Badges
    claims = report.get("verified_claims", [])
    print("\n" + "─" * 80)
    print(f"  5. VERIFIED FACTUAL CLAIMS & NLI GUARDRAIL VERDICTS ({len(claims)} claims)")
    print("─" * 80)
    for c in claims:
        verdict = c["verdict"]
        cid = c["claim_id"]
        text_claim = c["text"]
        cites = ", ".join(c["evidence_ids"]) if c["evidence_ids"] else "None (ungrounded)"
        p_e = c.get("entailment_score", 0.0)
        p_c = c.get("contradiction_score", 0.0)

        if verdict == "SUPPORTED":
            badge = "[SUPPORTED]"
        elif verdict == "CONTRADICTED":
            badge = "[CONTRADICTED]"
        else:
            badge = "[INSUFFICIENT_EVIDENCE]"

        print(f"\n     {badge:<24} | {cid} (Cites: {cites})")
        print(f"     Statement : \"{text_claim}\"")
        print(f"     NLI Scores: Entailment={p_e:.4f}, Contradiction={p_c:.4f}")
        if c.get("best_excerpt"):
            print(f"     Evidence  : \"{c['best_excerpt'][:160]}...\"")

    timings = report.get("timings_ms", {})
    print("\n" + "─" * 80)
    print("  PERFORMANCE TIMINGS:")
    print(f"     Classifier: {timings.get('classifier_ms', 0)} ms | Retrieval: {timings.get('retrieval_ms', 0)} ms")
    print(f"     Ollama Gen: {timings.get('generation_ms', 0)} ms | NLI Guard: {timings.get('guardrail_ms', 0)} ms")
    print(f"     Total Pipeline Latency: {timings.get('total_ms', 0)} ms ({timings.get('total_ms', 0)/1000:.2f} s)")
    if report.get("trace_file"):
        print(f"  Execution Trace: {report['trace_file']}")
    print("=" * 80 + "\n")

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.cli",
        description="RepoTriage — Local-first GitHub Issue Intelligence & Resolution Copilot",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand: classify
    classify_parser = subparsers.add_parser(
        "classify",
        help="Classify an issue report into bug, feature, question, or uncertain.",
    )
    classify_parser.add_argument(
        "text",
        type=str,
        help="Issue title or description text to classify",
    )
    classify_parser.add_argument(
        "--body",
        type=str,
        default="",
        help="Optional separate body text",
    )
    classify_parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_UNCERTAIN_THRESHOLD,
        help=f"Uncertainty threshold for top probability (default: {DEFAULT_UNCERTAIN_THRESHOLD})",
    )

    # Subcommand: ingest
    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Ingest issues, comments, and documentation for a repository.",
    )
    ingest_parser.add_argument(
        "--repo",
        type=str,
        default=None,
        help="Repository in format owner/repo (e.g. pallets/flask)",
    )
    ingest_parser.add_argument(
        "--all",
        action="store_true",
        help="Ingest all seed repositories (pydantic/pydantic, encode/httpx, pallets/flask)",
    )
    ingest_parser.add_argument(
        "--target-issues",
        type=int,
        default=400,
        help="Target issue count to ingest (default: 400)",
    )
    ingest_parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-fetching and ignore cached responses",
    )

    # Subcommand: search
    search_parser = subparsers.add_parser(
        "search",
        help="Search repository corpus using hybrid retrieval (BM25 + BGE + RRF + Reranker).",
    )
    search_parser.add_argument(
        "query",
        type=str,
        help="Search query or issue description",
    )
    search_parser.add_argument(
        "--repo",
        type=str,
        default=None,
        help="Target repository filter (e.g. pallets/flask, httpx, pydantic)",
    )
    search_parser.add_argument(
        "--top-k",
        type=int,
        default=6,
        help="Number of evidence items to return (default: 6)",
    )
    search_parser.add_argument(
        "--method",
        choices=["bm25", "dense", "hybrid", "hybrid_rerank"],
        default="hybrid_rerank",
        help="Retrieval method (default: hybrid_rerank)",
    )

    # Subcommand: analyze
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Analyze an issue with classification, hybrid retrieval, LLM synthesis, and NLI verification.",
    )
    analyze_parser.add_argument(
        "text",
        type=str,
        nargs="?",
        default="",
        help="Issue title/description text to analyze",
    )
    analyze_parser.add_argument(
        "--repo",
        type=str,
        default=None,
        help="Target repository filter (e.g. pydantic/pydantic, encode/httpx, pallets/flask, psf/requests)",
    )
    analyze_parser.add_argument(
        "--issue",
        type=str,
        default=None,
        help="Direct GitHub issue URL to analyze (e.g. https://github.com/owner/repo/issues/123)",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "classify":
        return handle_classify(args)
    if args.command == "ingest":
        return handle_ingest(args)
    if args.command == "search":
        return handle_search(args)
    if args.command == "analyze":
        return handle_analyze(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
