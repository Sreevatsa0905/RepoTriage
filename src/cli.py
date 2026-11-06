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

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "classify":
        return handle_classify(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
