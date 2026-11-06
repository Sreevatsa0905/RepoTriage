"""
GitHub REST API Client for RepoTriage
======================================
Implements resilient fetching of issues and issue comments using the standard library.
Supports pagination up to target issue count, pull-request filtering, rate-limit backoff,
and raw JSONL caching.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone
from http.client import HTTPResponse
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

GITHUB_API_BASE = "https://api.github.com"
DEFAULT_OWNER = "fastapi"
DEFAULT_REPO = "fastapi"
DEFAULT_PER_PAGE = 30
DEFAULT_MAX_PAGES = 0  # 0 = unlimited
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2  # seconds

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
DATA_DIR = RAW_DATA_DIR

_LINK_NEXT_RE = re.compile(r'<([^>]+)>;\s*rel="next"')


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
        sleep_seconds = max(reset_ts - now, 1) + 1
        logger.warning("Rate limit reached. Sleeping %d seconds …", sleep_seconds)
        time.sleep(sleep_seconds)


def _request_with_retry(url: str, retries: int = MAX_RETRIES) -> tuple[Any, str | None]:
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
                if not isinstance(data, (list, dict)):
                    raise ValueError(f"Expected JSON list or dict, got {type(data).__name__}")
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
# Core Issue Fetching Functions (matching test_ingest signatures)
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
    Retains exact original signature for test_ingest compatibility.
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


def save_dataset(
    issues: list[dict[str, Any]],
    owner: str,
    repo: str,
    output_dir: Path = RAW_DATA_DIR,
) -> Path:
    """Save normalised issues to a JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    filename = f"{owner}_{repo}_issues_{timestamp}.json"
    filepath = output_dir / filename

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(issues, f, indent=2, ensure_ascii=False)

    logger.info("Saved %d issues to %s", len(issues), filepath)
    return filepath


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
# Raw JSONL Cache Helpers
# ---------------------------------------------------------------------------

def get_issues_cache_path(owner: str, repo: str) -> Path:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    return RAW_DATA_DIR / f"{owner}_{repo}_raw_issues.jsonl"


def get_comments_cache_path(owner: str, repo: str) -> Path:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    return RAW_DATA_DIR / f"{owner}_{repo}_raw_comments.jsonl"


def load_cached_jsonl(filepath: Path) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of dicts."""
    if not filepath.exists():
        return []
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def append_to_jsonl(filepath: Path, records: list[dict[str, Any]]) -> None:
    """Append records to a JSONL file."""
    if not records:
        return
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def overwrite_jsonl(filepath: Path, records: list[dict[str, Any]]) -> None:
    """Write records to a JSONL file atomically."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    temp_path = filepath.with_suffix(".tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    temp_path.replace(filepath)


# ---------------------------------------------------------------------------
# GitHub Client Class
# ---------------------------------------------------------------------------

class GitHubClient:
    """
    Manages issue and comment acquisition with disk caching.
    """

    def __init__(self, raw_data_dir: Path = RAW_DATA_DIR):
        self.raw_data_dir = raw_data_dir
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)

    def fetch_repo_issues(
        self,
        owner: str,
        repo: str,
        target_count: int = 400,
        per_page: int = 100,
        force_refresh: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Fetch issues up to target_count, prioritizing closed issues.
        Paginates until target_count genuine issues (excluding PRs) are reached.
        """
        cache_path = get_issues_cache_path(owner, repo)
        if not force_refresh and cache_path.exists():
            cached = load_cached_jsonl(cache_path)
            if len(cached) >= target_count:
                logger.info(
                    "Reusing %d cached issues from %s for %s/%s",
                    len(cached), cache_path.name, owner, repo,
                )
                return cached[:target_count]

        logger.info(
            "Fetching issues for %s/%s (target=%d, per_page=%d)...",
            owner, repo, target_count, per_page,
        )

        per_page = min(max(per_page, 1), 100)
        collected_issues: list[dict[str, Any]] = []
        seen_numbers: set[int] = set()

        # Prioritize closed issues first to capture resolutions and comments, then open issues
        states_to_fetch = ["closed", "open"]

        for state in states_to_fetch:
            if len(collected_issues) >= target_count:
                break

            url: str | None = (
                f"{GITHUB_API_BASE}/repos/{owner}/{repo}/issues"
                f"?state={state}&per_page={per_page}&page=1&sort=created&direction=desc"
            )

            while url and len(collected_issues) < target_count:
                items, next_url = _request_with_retry(url)
                if not isinstance(items, list):
                    break

                for item in items:
                    if "pull_request" in item:
                        continue
                    num = item["number"]
                    if num not in seen_numbers:
                        seen_numbers.add(num)
                        collected_issues.append(item)
                        if len(collected_issues) >= target_count:
                            break

                logger.info(
                    "  [%s/%s state=%s] Collected %d / %d issues",
                    owner, repo, state, len(collected_issues), target_count,
                )

                if not items or len(collected_issues) >= target_count:
                    break
                url = next_url

        overwrite_jsonl(cache_path, collected_issues)
        logger.info("Saved %d raw issues to %s", len(collected_issues), cache_path)
        return collected_issues

    def fetch_issue_comments(
        self,
        owner: str,
        repo: str,
        issue_number: int,
        max_comments: int = 10,
    ) -> list[dict[str, Any]]:
        """Fetch comments for a single issue, capped at max_comments."""
        url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/issues/{issue_number}/comments?per_page={max_comments}"
        try:
            data, _ = _request_with_retry(url)
            if isinstance(data, list):
                return data[:max_comments]
            return []
        except Exception as e:
            logger.warning("Failed fetching comments for %s/%s#%d: %s", owner, repo, issue_number, e)
            return []

    def fetch_all_comments_for_issues(
        self,
        owner: str,
        repo: str,
        issues: list[dict[str, Any]],
        max_comments_per_issue: int = 10,
        force_refresh: bool = False,
    ) -> list[dict[str, Any]]:
        """Fetch comments for closed issues that have >= 1 comment, with JSONL caching."""
        cache_path = get_comments_cache_path(owner, repo)
        cached_comments = load_cached_jsonl(cache_path) if (cache_path.exists() and not force_refresh) else []

        seen_comment_ids = {c["id"] for c in cached_comments}
        already_fetched_issues = {c["issue_number"] for c in cached_comments if "issue_number" in c}

        candidates = [
            i for i in issues
            if i.get("state") == "closed" and i.get("comments", 0) > 0
            and i["number"] not in already_fetched_issues
        ]

        logger.info(
            "Found %d closed issues with comments to fetch for %s/%s (%d already cached)",
            len(candidates), owner, repo, len(already_fetched_issues),
        )

        new_comments: list[dict[str, Any]] = []
        for idx, issue in enumerate(candidates, 1):
            num = issue["number"]
            raw_comments = self.fetch_issue_comments(owner, repo, num, max_comments=max_comments_per_issue)
            for c in raw_comments:
                if c["id"] not in seen_comment_ids:
                    seen_comment_ids.add(c["id"])
                    c["issue_number"] = num
                    c["repo"] = f"{owner}/{repo}"
                    new_comments.append(c)

            if idx % 25 == 0 or idx == len(candidates):
                logger.info("  Fetched comments for %d / %d issues", idx, len(candidates))
                append_to_jsonl(cache_path, new_comments)
                cached_comments.extend(new_comments)
                new_comments = []

        if new_comments:
            append_to_jsonl(cache_path, new_comments)
            cached_comments.extend(new_comments)

        logger.info("Total comments cached for %s/%s: %d", owner, repo, len(cached_comments))
        return cached_comments

    def fetch_single_issue(
        self,
        owner: str,
        repo: str,
        issue_number: int,
    ) -> dict[str, Any] | None:
        """
        Fetch a single issue by number on-demand with JSONL cache lookup.
        """
        cache_path = get_issues_cache_path(owner, repo)
        cached = load_cached_jsonl(cache_path)
        for item in cached:
            if item.get("number") == issue_number:
                return item

        logger.info("Fetching single issue on-demand: %s/%s#%d", owner, repo, issue_number)
        url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/issues/{issue_number}"
        try:
            data, _ = _request_with_retry(url)
            if isinstance(data, dict) and "pull_request" not in data:
                append_to_jsonl(cache_path, [data])
                if data.get("state") == "closed" and data.get("comments", 0) > 0:
                    raw_comments = self.fetch_issue_comments(owner, repo, issue_number)
                    if raw_comments:
                        comments_cache = get_comments_cache_path(owner, repo)
                        for c in raw_comments:
                            c["issue_number"] = issue_number
                            c["repo"] = f"{owner}/{repo}"
                        append_to_jsonl(comments_cache, raw_comments)
                return data
            return None
        except Exception as e:
            logger.warning("Failed fetching issue %s/%s#%d: %s", owner, repo, issue_number, e)
            return None


# Standalone function exports matching requirements
_client_instance = GitHubClient()
fetch_issue_comments = _client_instance.fetch_issue_comments
fetch_single_issue = _client_instance.fetch_single_issue
