"""
CLI entry point for the RepoTriage issue classifier.

Usage:
  # Classify a single issue:
  python -m src.classifier --title "App crashes on startup" --body "Traceback: ..."

  # Classify issues from an ingested JSON file:
  python -m src.classifier --file data/raw/fastapi_fastapi_issues_20261106_145717.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.classifier.config import MODELS_DIR
from src.classifier.inference import IssueClassifierService


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RepoTriage — Issue Classification CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--title",
        type=str,
        default="",
        help="Issue title to classify",
    )
    parser.add_argument(
        "--body",
        type=str,
        default="",
        help="Issue body text to classify",
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=None,
        help="Path to an ingested JSON file from RepoTriage ingestion",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=MODELS_DIR / "pooled_classifier.joblib",
        help="Path to the trained model artifact (default: models/pooled_classifier.joblib)",
    )
    parser.add_argument(
        "--model-type",
        choices=["classical", "setfit"],
        default="classical",
        help="Model type (default: classical)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum issues to triage when reading from --file (default: 10, 0 for all)",
    )

    args = parser.parse_args()

    if not args.title and not args.file:
        parser.print_help()
        print("\nError: Please provide either --title (and optionally --body) or --file.")
        sys.exit(1)

    if not args.model_path.exists():
        print(f"Error: Model not found at '{args.model_path}'.")
        print("Please train a model first using: python scripts/train_pooled.py")
        sys.exit(1)

    service = IssueClassifierService.from_saved_model(args.model_path, model_type=args.model_type)

    if args.title or not args.file:
        res = service.classify_issue(args.title, args.body)
        print("\n" + "=" * 60)
        print("  REPOTRIAGE CLASSIFICATION RESULT")
        print("=" * 60)
        print(f"  Title     : {args.title}")
        print(f"  Category  : {res.category.upper()}")
        print(f"  Model     : {res.model_identifier}")
        print("  Confidence scores:")
        for label, score in sorted(res.scores.items(), key=lambda x: -x[1]):
            bar = "█" * int(score * 30)
            print(f"    {label:<10} : {score:.4f}  {bar}")
        print("=" * 60 + "\n")
        return

    # Batch file mode
    with open(args.file, "r", encoding="utf-8") as f:
        issues = json.load(f)

    if not isinstance(issues, list):
        print(f"Error: Expected a JSON list of issues in {args.file}")
        sys.exit(1)

    total_issues = len(issues)
    if args.limit > 0:
        issues = issues[:args.limit]

    print("\n" + "=" * 80)
    print(f"  TRIAGING {len(issues)} of {total_issues} ISSUES FROM {args.file.name}")
    print(f"  Model: {service.model_identifier}")
    print("=" * 80)

    category_counts: dict[str, int] = {}
    for item in issues:
        num = item.get("number", "?")
        title = item.get("title", "")
        body = item.get("body", "")
        res = service.classify_issue(title, body)

        category_counts[res.category] = category_counts.get(res.category, 0) + 1
        scores_str = " | ".join(f"{k}:{v:.2f}" for k, v in res.scores.items())
        print(f"  #{num:<6} [{res.category.upper():<8}] {title[:48]:<48} ({scores_str})")

    print("-" * 80)
    print("  Summary breakdown:")
    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        pct = (count / len(issues)) * 100
        print(f"    {cat:<10} : {count:3d} ({pct:5.1f}%)")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
