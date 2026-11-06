"""
GitHub Issue Ingestion Tool
============================
Fetches issues from a public GitHub repository via the REST API,
normalises them into a consistent schema, and saves the dataset as JSON.

Usage:
    python -m src.ingest                           # defaults: fastapi/fastapi
    python -m src.ingest --owner pallets --repo flask --per-page 50 --max-pages 3
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from http.client import HTTPResponse
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GITHUB_API_BASE = "https://api.github.com"
DEFAULT_OWNER = "fastapi"
DEFAULT_REPO = "fastapi"
DEFAULT_PER_PAGE = 30
DEFAULT_MAX_PAGES = 0  # 0 = unlimited
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2  # seconds

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

# Regex for GitHub Link header pagination
_LINK_NEXT_RE = re.compile(r'<([^>]+)>;\s*rel="next"')


# ---------------------------------------------------------------------------
# Networking helpers
# ---------------------------------------------------------------------------

def _build_headers() -> dict[str, str]:
    """Standard headers for GitHub API requests."""
    return {
        "Accept": "application/vnd.github+json",
        "User-Agent": "RepoTriage/0.1",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _parse_link_next(link_header: str | None) -> str | None:
    """Extract the 'next' URL from a GitHub Link header."""
    if not link_header:
        return None
    match = _LINK_NEXT_RE.search(link_header)
    return match.group(1) if match else None


def _handle_rate_limit(response: HTTPResponse) -> None:
    """If rate-limited, sleep until the reset timestamp."""
    remaining = response.headers.get("X-RateLimit-Remaining")
    if remaining is not None and int(remaining) == 0:
        reset_ts = int(response.headers.get("X-RateLimit-Reset", "0"))
        now = int(time.time())
        sleep_seconds = max(reset_ts - now, 1) + 1  # +1 safety margin
        logger.warning("Rate limit reached. Sleeping %d seconds …", sleep_seconds)
        time.sleep(sleep_seconds)


def _request_with_retry(url: str, retries: int = MAX_RETRIES) -> tuple[list[dict], str | None]:
    """
    GET a URL and return (parsed_json, next_url).

    Retries on transient failures (429, 403 rate-limit, network errors).
    """
    headers = _build_headers()
    last_exc: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            req = Request(url, headers=headers)
            with urlopen(req, timeout=30) as resp:
                _handle_rate_limit(resp)
                body = resp.read().decode("utf-8")
                data = json.loads(body)
                if not isinstance(data, list):
                    raise ValueError(f"Expected JSON array, got {type(data).__name__}")
                link_next = _parse_link_next(resp.headers.get("Link"))
                return data, link_next

        except HTTPError as exc:
            last_exc = exc
            if exc.code in (403, 429):
                # Try to read reset header from the error response
                reset_ts = exc.headers.get("X-RateLimit-Reset") if exc.headers else None
                if reset_ts:
                    sleep_for = max(int(reset_ts) - int(time.time()), 1) + 1
                else:
                    sleep_for = RETRY_BACKOFF_BASE ** attempt
                logger.warning(
                    "HTTP %d on attempt %d/%d. Retrying in %d s …",
                    exc.code, attempt, retries, sleep_for,
                )
                time.sleep(sleep_for)
            elif exc.code >= 500:
                sleep_for = RETRY_BACKOFF_BASE ** attempt
                logger.warning(
                    "HTTP %d (server error) on attempt %d/%d. Retrying in %d s …",
                    exc.code, attempt, retries, sleep_for,
                )
                time.sleep(sleep_for)
            else:
                raise  # 404, 422 etc. — not recoverable

        except (URLError, OSError, ValueError) as exc:
            last_exc = exc
            sleep_for = RETRY_BACKOFF_BASE ** attempt
            logger.warning(
                "Network/parse error on attempt %d/%d: %s. Retrying in %d s …",
                attempt, retries, exc, sleep_for,
            )
            time.sleep(sleep_for)

    raise RuntimeError(
        f"Failed after {retries} retries for {url}"
    ) from last_exc


# ---------------------------------------------------------------------------
# Issue fetching
# ---------------------------------------------------------------------------

def fetch_issues(
    owner: str = DEFAULT_OWNER,
    repo: str = DEFAULT_REPO,
    per_page: int = DEFAULT_PER_PAGE,
    max_pages: int = DEFAULT_MAX_PAGES,
    state: str = "all",
) -> list[dict[str, Any]]:
    """
    Fetch issues (excluding pull requests) from a GitHub repository.

    Parameters
    ----------
    owner : str
        Repository owner / organisation.
    repo : str
        Repository name.
    per_page : int
        Results per page (max 100).
    max_pages : int
        Stop after this many pages. 0 = fetch all pages.
    state : str
        Issue state filter: "open", "closed", or "all".

    Returns
    -------
    list[dict]
        Raw issue dicts from the API (PRs already excluded).
    """
    per_page = min(max(per_page, 1), 100)
    url: str | None = (
        f"{GITHUB_API_BASE}/repos/{owner}/{repo}/issues"
        f"?state={state}&per_page={per_page}&page=1&sort=created&direction=desc"
    )

    all_issues: list[dict] = []
    page = 0

    while url:
        page += 1
        if max_pages and page > max_pages:
            logger.info("Reached page limit (%d). Stopping.", max_pages)
            break

        logger.info("Fetching page %d: %s", page, url)
        items, next_url = _request_with_retry(url)

        # Exclude pull requests — GitHub's issues endpoint includes them
        issues_only = [i for i in items if "pull_request" not in i]
        all_issues.extend(issues_only)

        pr_count = len(items) - len(issues_only)
        logger.info(
            "  Page %d: %d items (%d issues, %d PRs excluded)",
            page, len(items), len(issues_only), pr_count,
        )

        if not items:
            break
        url = next_url

    return all_issues


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize_issue(raw: dict[str, Any], owner: str, repo: str) -> dict[str, Any]:
    """
    Map a raw GitHub issue dict to our canonical schema.

    Schema
    ------
    - repository   : str   — "owner/repo"
    - number       : int   — issue number
    - title        : str   — issue title
    - body         : str   — issue body (empty string if null)
    - state        : str   — "open" or "closed"
    - labels       : list[str] — label names
    - created_at   : str   — ISO 8601 timestamp
    - updated_at   : str   — ISO 8601 timestamp
    - html_url     : str   — link to the issue on GitHub
    - comment_count: int   — number of comments
    """
    return {
        "repository": f"{owner}/{repo}",
        "number": raw["number"],
        "title": raw.get("title", ""),
        "body": raw.get("body") or "",
        "state": raw.get("state", "unknown"),
        "labels": [lbl["name"] for lbl in raw.get("labels", []) if isinstance(lbl, dict)],
        "created_at": raw.get("created_at", ""),
        "updated_at": raw.get("updated_at", ""),
        "html_url": raw.get("html_url", ""),
        "comment_count": raw.get("comments", 0),
    }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_dataset(
    issues: list[dict[str, Any]],
    owner: str,
    repo: str,
    output_dir: Path = DATA_DIR,
) -> Path:
    """Save normalised issues to a JSON file. Returns the output path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    filename = f"{owner}_{repo}_issues_{timestamp}.json"
    filepath = output_dir / filename

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(issues, f, indent=2, ensure_ascii=False)

    logger.info("Saved %d issues to %s", len(issues), filepath)
    return filepath


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_stats(issues: list[dict[str, Any]]) -> None:
    """Print ingestion statistics and label frequencies to stdout."""
    total = len(issues)
    if total == 0:
        print("No issues ingested.")
        return

    open_count = sum(1 for i in issues if i["state"] == "open")
    closed_count = sum(1 for i in issues if i["state"] == "closed")
    with_body = sum(1 for i in issues if i["body"])
    with_labels = sum(1 for i in issues if i["labels"])
    total_comments = sum(i["comment_count"] for i in issues)

    print("\n" + "=" * 60)
    print("  INGESTION STATISTICS")
    print("=" * 60)
    print(f"  Total issues   : {total}")
    print(f"  Open           : {open_count}")
    print(f"  Closed         : {closed_count}")
    print(f"  With body text : {with_body} ({with_body * 100 // total}%)")
    print(f"  With labels    : {with_labels} ({with_labels * 100 // total}%)")
    print(f"  Total comments : {total_comments}")
    print(f"  Avg comments   : {total_comments / total:.1f}")

    # Label frequency
    label_counter: Counter[str] = Counter()
    for issue in issues:
        for label in issue["labels"]:
            label_counter[label] += 1

    if label_counter:
        print(f"\n  Unique labels  : {len(label_counter)}")
        print("\n  Top 20 labels:")
        for label, count in label_counter.most_common(20):
            bar = "█" * min(count * 40 // label_counter.most_common(1)[0][1], 40)
            print(f"    {label:<40s} {count:>5d}  {bar}")
    else:
        print("\n  No labels found on any issue.")

    # Data-quality flags
    print("\n  DATA QUALITY NOTES:")
    unlabeled = total - with_labels
    if unlabeled:
        print(f"    ⚠  {unlabeled} issues ({unlabeled * 100 // total}%) have no labels.")
    empty_body = total - with_body
    if empty_body:
        print(f"    ⚠  {empty_body} issues ({empty_body * 100 // total}%) have empty body text.")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch GitHub issues and save as a normalised JSON dataset.",
    )
    parser.add_argument("--owner", default=DEFAULT_OWNER, help="Repo owner (default: %(default)s)")
    parser.add_argument("--repo", default=DEFAULT_REPO, help="Repo name (default: %(default)s)")
    parser.add_argument("--per-page", type=int, default=DEFAULT_PER_PAGE, help="Results per page (default: %(default)s)")
    parser.add_argument("--max-pages", type=int, default=DEFAULT_MAX_PAGES, help="Max pages to fetch; 0 = all (default: %(default)s)")
    parser.add_argument("--state", choices=["open", "closed", "all"], default="all", help="Issue state filter (default: %(default)s)")
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR, help="Output directory (default: %(default)s)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    logger.info(
        "Starting ingestion: %s/%s (per_page=%d, max_pages=%s, state=%s)",
        args.owner, args.repo, args.per_page,
        args.max_pages or "unlimited", args.state,
    )

    raw_issues = fetch_issues(
        owner=args.owner,
        repo=args.repo,
        per_page=args.per_page,
        max_pages=args.max_pages,
        state=args.state,
    )

    normalised = [normalize_issue(i, args.owner, args.repo) for i in raw_issues]

    filepath = save_dataset(normalised, args.owner, args.repo, output_dir=args.output_dir)
    print(f"Dataset saved to: {filepath}")

    print_stats(normalised)


if __name__ == "__main__":
    main()
