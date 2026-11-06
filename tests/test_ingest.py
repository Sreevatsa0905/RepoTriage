"""
Unit tests for src.ingest — GitHub Issue Ingestion.

All network calls are mocked; no real HTTP requests are made.
"""

from __future__ import annotations

import io
import json
import unittest
from http.client import HTTPResponse
from typing import Any
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError, URLError

from src.ingest import (
    _parse_link_next,
    fetch_issues,
    normalize_issue,
    print_stats,
    save_dataset,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_issue(number: int, *, is_pr: bool = False, labels: list[str] | None = None,
                state: str = "open", body: str = "body text") -> dict[str, Any]:
    """Build a realistic-looking raw GitHub issue dict."""
    issue: dict[str, Any] = {
        "number": number,
        "title": f"Issue #{number}",
        "body": body,
        "state": state,
        "labels": [{"name": lbl, "id": hash(lbl)} for lbl in (labels or [])],
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-06-01T00:00:00Z",
        "html_url": f"https://github.com/test/repo/issues/{number}",
        "comments": 3,
    }
    if is_pr:
        issue["pull_request"] = {
            "url": f"https://api.github.com/repos/test/repo/pulls/{number}",
        }
    return issue


def _mock_urlopen_factory(pages: list[tuple[list[dict], str | None]]):
    """
    Return a context-manager mock for urlopen that serves `pages`.

    Each page is a (items_list, next_link_url_or_None) tuple.
    Pages are consumed in order across successive calls.
    """
    call_count = {"n": 0}

    def fake_urlopen(request, *, timeout=None):
        idx = call_count["n"]
        call_count["n"] += 1
        items, next_url = pages[idx]

        body = json.dumps(items).encode()
        resp = MagicMock()
        resp.read.return_value = body
        resp.__enter__ = lambda s: s
        resp.__exit__ = lambda s, *a: None

        link_header = None
        if next_url:
            link_header = f'<{next_url}>; rel="next"'

        header_dict = {
            "Link": link_header,
            "X-RateLimit-Remaining": "50",
            "X-RateLimit-Reset": "9999999999",
        }
        resp.headers = MagicMock()
        resp.headers.get = lambda key, default=None: header_dict.get(key, default)
        return resp

    return fake_urlopen


# ---------------------------------------------------------------------------
# Tests: Link header parsing
# ---------------------------------------------------------------------------

class TestParseLinkNext(unittest.TestCase):

    def test_valid_next_link(self):
        header = '<https://api.github.com/repos/a/b/issues?page=2>; rel="next", <https://api.github.com/repos/a/b/issues?page=5>; rel="last"'
        self.assertEqual(
            _parse_link_next(header),
            "https://api.github.com/repos/a/b/issues?page=2",
        )

    def test_no_next(self):
        header = '<https://api.github.com/repos/a/b/issues?page=1>; rel="prev"'
        self.assertIsNone(_parse_link_next(header))

    def test_none_header(self):
        self.assertIsNone(_parse_link_next(None))

    def test_empty_header(self):
        self.assertIsNone(_parse_link_next(""))


# ---------------------------------------------------------------------------
# Tests: Normalization
# ---------------------------------------------------------------------------

class TestNormalizeIssue(unittest.TestCase):

    def test_basic_normalization(self):
        raw = _make_issue(42, labels=["bug", "help wanted"], state="open")
        result = normalize_issue(raw, "fastapi", "fastapi")

        self.assertEqual(result["repository"], "fastapi/fastapi")
        self.assertEqual(result["number"], 42)
        self.assertEqual(result["title"], "Issue #42")
        self.assertEqual(result["body"], "body text")
        self.assertEqual(result["state"], "open")
        self.assertEqual(result["labels"], ["bug", "help wanted"])
        self.assertEqual(result["created_at"], "2024-01-01T00:00:00Z")
        self.assertEqual(result["html_url"], "https://github.com/test/repo/issues/42")
        self.assertEqual(result["comment_count"], 3)

    def test_null_body_becomes_empty_string(self):
        raw = _make_issue(1, body=None)
        raw["body"] = None
        result = normalize_issue(raw, "o", "r")
        self.assertEqual(result["body"], "")

    def test_missing_labels(self):
        raw = _make_issue(1)
        del raw["labels"]
        result = normalize_issue(raw, "o", "r")
        self.assertEqual(result["labels"], [])

    def test_label_with_non_dict_skipped(self):
        raw = _make_issue(1)
        raw["labels"] = [{"name": "good"}, "bad-string-label", 42]
        result = normalize_issue(raw, "o", "r")
        self.assertEqual(result["labels"], ["good"])


# ---------------------------------------------------------------------------
# Tests: Fetching with pagination and PR filtering
# ---------------------------------------------------------------------------

class TestFetchIssues(unittest.TestCase):

    @patch("src.ingest.urlopen")
    def test_single_page_no_prs(self, mock_urlopen):
        issues = [_make_issue(1), _make_issue(2)]
        mock_urlopen.side_effect = _mock_urlopen_factory([
            (issues, None),
        ])

        result = fetch_issues("test", "repo", per_page=10, max_pages=1)
        self.assertEqual(len(result), 2)

    @patch("src.ingest.urlopen")
    def test_prs_are_excluded(self, mock_urlopen):
        items = [
            _make_issue(1),
            _make_issue(2, is_pr=True),
            _make_issue(3),
            _make_issue(4, is_pr=True),
        ]
        mock_urlopen.side_effect = _mock_urlopen_factory([
            (items, None),
        ])

        result = fetch_issues("test", "repo", per_page=10, max_pages=1)
        self.assertEqual(len(result), 2)
        numbers = {i["number"] for i in result}
        self.assertEqual(numbers, {1, 3})

    @patch("src.ingest.urlopen")
    def test_pagination_follows_next_link(self, mock_urlopen):
        page1 = [_make_issue(1), _make_issue(2)]
        page2 = [_make_issue(3)]
        mock_urlopen.side_effect = _mock_urlopen_factory([
            (page1, "https://api.github.com/repos/t/r/issues?page=2"),
            (page2, None),
        ])

        result = fetch_issues("t", "r", per_page=2, max_pages=0)
        self.assertEqual(len(result), 3)

    @patch("src.ingest.urlopen")
    def test_max_pages_limit_respected(self, mock_urlopen):
        page1 = [_make_issue(1)]
        page2 = [_make_issue(2)]
        mock_urlopen.side_effect = _mock_urlopen_factory([
            (page1, "https://api.github.com/repos/t/r/issues?page=2"),
            (page2, None),
        ])

        result = fetch_issues("t", "r", per_page=1, max_pages=1)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["number"], 1)

    @patch("src.ingest.urlopen")
    def test_empty_page_stops_pagination(self, mock_urlopen):
        mock_urlopen.side_effect = _mock_urlopen_factory([
            ([], None),
        ])

        result = fetch_issues("t", "r", per_page=10, max_pages=0)
        self.assertEqual(len(result), 0)


# ---------------------------------------------------------------------------
# Tests: Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling(unittest.TestCase):

    @patch("src.ingest.urlopen")
    @patch("src.ingest.time.sleep")  # Don't actually sleep in tests
    def test_http_404_raises_immediately(self, mock_sleep, mock_urlopen):
        mock_urlopen.side_effect = HTTPError(
            "https://api.github.com/repos/x/y/issues",
            404, "Not Found", {}, None,
        )
        with self.assertRaises(HTTPError):
            fetch_issues("x", "y", max_pages=1)

    @patch("src.ingest.urlopen")
    @patch("src.ingest.time.sleep")
    def test_http_403_retries_then_fails(self, mock_sleep, mock_urlopen):
        error_headers = MagicMock()
        error_headers.get = lambda key, default=None: None
        mock_urlopen.side_effect = HTTPError(
            "https://api.github.com/repos/x/y/issues",
            403, "Forbidden", error_headers, None,
        )
        with self.assertRaises(RuntimeError):
            fetch_issues("x", "y", max_pages=1)
        self.assertEqual(mock_urlopen.call_count, 3)  # MAX_RETRIES

    @patch("src.ingest.urlopen")
    @patch("src.ingest.time.sleep")
    def test_network_error_retries(self, mock_sleep, mock_urlopen):
        mock_urlopen.side_effect = URLError("Connection refused")
        with self.assertRaises(RuntimeError):
            fetch_issues("x", "y", max_pages=1)
        self.assertEqual(mock_urlopen.call_count, 3)

    @patch("src.ingest.urlopen")
    @patch("src.ingest.time.sleep")
    def test_malformed_json_retries(self, mock_sleep, mock_urlopen):
        resp = MagicMock()
        resp.read.return_value = b"not valid json"
        resp.__enter__ = lambda s: s
        resp.__exit__ = lambda s, *a: None
        resp.headers = MagicMock()
        resp.headers.get = lambda key, default=None: {"X-RateLimit-Remaining": "50"}.get(key, default)
        mock_urlopen.return_value = resp

        with self.assertRaises(RuntimeError):
            fetch_issues("x", "y", max_pages=1)

    @patch("src.ingest.urlopen")
    @patch("src.ingest.time.sleep")
    def test_non_list_json_retries(self, mock_sleep, mock_urlopen):
        resp = MagicMock()
        resp.read.return_value = json.dumps({"message": "error"}).encode()
        resp.__enter__ = lambda s: s
        resp.__exit__ = lambda s, *a: None
        resp.headers = MagicMock()
        resp.headers.get = lambda key, default=None: {"X-RateLimit-Remaining": "50"}.get(key, default)
        mock_urlopen.return_value = resp

        with self.assertRaises(RuntimeError):
            fetch_issues("x", "y", max_pages=1)


# ---------------------------------------------------------------------------
# Tests: Persistence
# ---------------------------------------------------------------------------

class TestSaveDataset(unittest.TestCase):

    def test_save_creates_file(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            from pathlib import Path
            out_dir = Path(tmpdir)
            issues = [normalize_issue(_make_issue(1), "o", "r")]
            path = save_dataset(issues, "o", "r", output_dir=out_dir)

            self.assertTrue(path.exists())
            self.assertTrue(path.name.startswith("o_r_issues_"))
            self.assertTrue(path.name.endswith(".json"))

            loaded = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0]["number"], 1)


# ---------------------------------------------------------------------------
# Tests: Stats printing (smoke test)
# ---------------------------------------------------------------------------

class TestPrintStats(unittest.TestCase):

    def test_empty_issues(self):
        # Should not crash
        print_stats([])

    def test_stats_with_data(self):
        issues = [
            normalize_issue(_make_issue(1, labels=["bug"]), "o", "r"),
            normalize_issue(_make_issue(2, labels=["bug", "docs"], body=""), "o", "r"),
            normalize_issue(_make_issue(3, state="closed"), "o", "r"),
        ]
        # Should not crash — just prints to stdout
        print_stats(issues)


if __name__ == "__main__":
    unittest.main()
