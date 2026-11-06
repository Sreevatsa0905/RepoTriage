"""
RepoTriage Ingestion Package
============================
Contains GitHub API client, documentation loader, markdown/rst chunker,
and SQLite/JSONL persistence.

Maintains 100% backward-compatibility with Microproject 1 ingestion APIs and tests.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from http.client import HTTPResponse
import json
import logging
import os
from pathlib import Path
import re
import sys
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from src.ingest.chunker import Chunker
from src.ingest.docs_loader import DocsLoader
from src.ingest.github_client import GitHubClient
from src.ingest.storage import StorageEngine

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

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DATA_DIR = DATA_DIR

SEED_REPOSITORIES = [
    "pydantic/pydantic",
    "encode/httpx",
    "pallets/flask",
]

# Regex for GitHub Link header pagination
_LINK_NEXT_RE = re.compile(r'<([^>]+)>;\s*rel="next"')


# ---------------------------------------------------------------------------
# Networking helpers
# ---------------------------------------------------------------------------

def _load_env_token() -> None:
    """Load GITHUB_TOKEN from project root .env if not already set."""
    if "GITHUB_TOKEN" in os.environ and os.environ["GITHUB_TOKEN"]:
        return
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        try:
            with open(env_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        if k.strip() == "GITHUB_TOKEN":
                            os.environ["GITHUB_TOKEN"] = v.strip().strip("'\"")
                            break
        except Exception as e:
            logger.debug("Failed reading .env file: %s", e)


def _build_headers() -> dict[str, str]:
    """Standard headers for GitHub API requests."""
    _load_env_token()
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "RepoTriage/0.1",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


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


def _request_with_retry(url: str, retries: int = MAX_RETRIES, allow_dict: bool = False) -> tuple[Any, str | None]:
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
                if allow_dict:
                    if not isinstance(data, (list, dict)):
                        raise ValueError(f"Expected JSON array or dict, got {type(data).__name__}")
                else:
                    if not isinstance(data, list):
                        raise ValueError(f"Expected JSON array, got {type(data).__name__}")
                link_next = _parse_link_next(resp.headers.get("Link"))
                return data, link_next

        except HTTPError as exc:
            last_exc = exc
            if exc.code in (403, 429):
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
                raise

        except (URLError, OSError, ValueError) as exc:
            last_exc = exc
            sleep_for = RETRY_BACKOFF_BASE ** attempt
            logger.warning(
                "Network/parse error on attempt %d/%d: %s. Retrying in %d s …",
                attempt, retries, exc, sleep_for,
            )
            time.sleep(sleep_for)

    raise RuntimeError(f"Failed after {retries} retries for {url}") from last_exc


# Backwards compatibility aliases
build_headers = _build_headers
parse_link_next = _parse_link_next
handle_rate_limit = _handle_rate_limit
request_with_retry = _request_with_retry


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
        if not isinstance(items, list):
            break

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


def fetch_issue_comments(
    owner: str,
    repo: str,
    issue_number: int,
    max_comments: int = 10,
) -> list[dict[str, Any]]:
    """Fetch comments for an issue, capped at max_comments."""
    url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/issues/{issue_number}/comments?per_page={max_comments}"
    try:
        data, _ = _request_with_retry(url, allow_dict=True)
        return data[:max_comments] if isinstance(data, list) else []
    except Exception as exc:
        logger.warning("Failed fetching comments for %s/%s#%d: %s", owner, repo, issue_number, exc)
        return []


def fetch_single_issue(
    owner: str,
    repo: str,
    issue_number: int,
    fetch_comments: bool = True,
) -> dict[str, Any] | None:
    """
    Fetch a single issue by number on-demand with raw JSONL cache check.
    If the issue is closed with comments, also fetches comments.
    """
    cache_path = DATA_DIR / f"{owner}_{repo}_raw_issues.jsonl"
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        item = json.loads(line)
                        if item.get("number") == issue_number:
                            return normalize_issue(item, owner, repo)
                    except json.JSONDecodeError:
                        continue

    # Fetch from API
    url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/issues/{issue_number}"
    try:
        data, _ = _request_with_retry(url, allow_dict=True)
        if isinstance(data, dict) and "pull_request" not in data:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(data, ensure_ascii=False) + "\n")

            if fetch_comments and data.get("state") == "closed" and data.get("comments", 0) > 0:
                comments = fetch_issue_comments(owner, repo, issue_number)
                if comments:
                    comm_path = DATA_DIR / f"{owner}_{repo}_raw_comments.jsonl"
                    with open(comm_path, "a", encoding="utf-8") as f:
                        for c in comments:
                            c["issue_number"] = issue_number
                            c["repo"] = f"{owner}/{repo}"
                            f.write(json.dumps(c, ensure_ascii=False) + "\n")

            return normalize_issue(data, owner, repo)
        return None
    except Exception as exc:
        logger.warning("Failed fetching single issue %s/%s#%d: %s", owner, repo, issue_number, exc)
        return None


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize_issue(raw: dict[str, Any], owner: str, repo: str) -> dict[str, Any]:
    """Map a raw GitHub issue dict to our canonical schema."""
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

    print("\n  DATA QUALITY NOTES:")
    unlabeled = total - with_labels
    if unlabeled:
        print(f"    ⚠  {unlabeled} issues ({unlabeled * 100 // total}%) have no labels.")
    empty_body = total - with_body
    if empty_body:
        print(f"    ⚠  {empty_body} issues ({empty_body * 100 // total}%) have empty body text.")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Ingestion Orchestration for Stage 1
# ---------------------------------------------------------------------------

def ingest_repository(
    repo_identifier: str,
    target_issues: int = 400,
    force_refresh: bool = False,
    storage: StorageEngine | None = None,
) -> dict[str, int]:
    """
    Complete end-to-end ingestion pipeline for a single repository:
    1. Fetch up to target_issues (skipping PRs, prioritizing closed issues).
    2. Fetch comments for closed issues with discussions (capped at 10 each).
    3. Shallow-clone repository and load documentation (README, CONTRIBUTING, docs/**/*.{md,mdx,rst}).
    4. Chunk issues, comments, and doc files with heading and anchor preservation.
    5. Save everything to SQLite (data/processed/repotriage.db) and processed JSONL.
    """
    if "/" not in repo_identifier:
        raise ValueError(f"Invalid repository identifier '{repo_identifier}'. Expected 'owner/repo'.")

    owner, repo = repo_identifier.split("/", 1)
    if storage is None:
        storage = StorageEngine()

    gh_client = GitHubClient()
    docs_loader = DocsLoader()
    chunker = Chunker()

    print("\n" + "=" * 70)
    print(f"  INGESTING REPOSITORY: {owner}/{repo}")
    print("=" * 70)

    # 1. Issues
    print(f"--> Step 1/4: Fetching issues (target: {target_issues})...")
    raw_issues = gh_client.fetch_repo_issues(owner, repo, target_count=target_issues, force_refresh=force_refresh)
    normalized_issues = [normalize_issue(i, owner, repo) for i in raw_issues]
    storage.save_issues(normalized_issues)
    print(f"    Saved {len(normalized_issues)} issues to database.")

    # 2. Comments
    print("--> Step 2/4: Fetching comments for closed issues...")
    raw_comments = gh_client.fetch_all_comments_for_issues(owner, repo, raw_issues, force_refresh=force_refresh)
    normalized_comments = []
    for c in raw_comments:
        issue_num = c.get("issue_number")
        if issue_num is not None:
            user_login = c.get("user", {}).get("login", "unknown") if isinstance(c.get("user"), dict) else "unknown"
            normalized_comments.append({
                "repository": f"{owner}/{repo}",
                "issue_number": issue_num,
                "comment_id": c["id"],
                "user": user_login,
                "body": c.get("body", ""),
                "created_at": c.get("created_at", ""),
                "html_url": c.get("html_url", ""),
            })
    storage.save_comments(normalized_comments)
    print(f"    Saved {len(normalized_comments)} comments to database.")

    # 3. Documentation
    print("--> Step 3/4: Loading documentation files (README, CONTRIBUTING, docs/)...")
    doc_files = docs_loader.load_docs(owner, repo)
    storage.save_doc_files(doc_files)
    print(f"    Saved {len(doc_files)} documentation files to database.")

    # 4. Chunking
    print("--> Step 4/4: Chunking issues, comments, and docs into retrieval windows...")
    all_chunks: list[dict[str, Any]] = []

    for issue in normalized_issues:
        all_chunks.extend(chunker.chunk_issue(issue))

    for comment in normalized_comments:
        all_chunks.extend(chunker.chunk_comment(comment))

    for doc in doc_files:
        all_chunks.extend(chunker.chunk_doc_file(doc))

    storage.save_chunks(all_chunks, repo=f"{owner}/{repo}")
    print(f"    Generated and saved {len(all_chunks)} chunks to database.")

    counts = storage.get_repo_counts(f"{owner}/{repo}")
    print("\n  Summary Counts for " + f"{owner}/{repo}:")
    print(f"    Issues      : {counts['issues']}")
    print(f"    Comments    : {counts['comments']}")
    print(f"    Doc Files   : {counts['doc_files']}")
    print(f"    Chunks      : {counts['chunks']}")
    print("=" * 70 + "\n")

    return counts


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


__all__ = [
    "GitHubClient",
    "DocsLoader",
    "Chunker",
    "StorageEngine",
    "SEED_REPOSITORIES",
    "ingest_repository",
    "fetch_issues",
    "fetch_single_issue",
    "fetch_issue_comments",
    "normalize_issue",
    "save_dataset",
    "print_stats",
    "_parse_link_next",
    "_request_with_retry",
    "_build_headers",
    "_handle_rate_limit",
    "build_headers",
    "parse_link_next",
    "handle_rate_limit",
    "request_with_retry",
    "urlopen",
    "time",
    "HTTPError",
    "URLError",
    "Request",
    "main",
    "parse_args",
]
