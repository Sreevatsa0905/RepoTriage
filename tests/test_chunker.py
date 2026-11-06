"""
Unit tests for Chunker, DocsLoader parsing, and StorageEngine in Stage 1.
Offline tests - no real network or external services needed.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.ingest.chunker import (
    Chunker,
    count_tokens,
    parse_markdown_sections,
    parse_rst_sections,
    slugify,
    split_text_into_token_windows,
)
from src.ingest.storage import StorageEngine


class TestChunkerParsing(unittest.TestCase):
    """Test document parsing for Markdown and reStructuredText."""

    def test_slugify(self):
        self.assertEqual(slugify("Quick Start & Installation!"), "quick-start-installation")
        self.assertEqual(slugify("Configuration (v2.0)"), "configuration-v20")
        self.assertEqual(slugify("   Leading and Trailing   "), "leading-and-trailing")

    def test_parse_markdown_sections(self):
        md_text = """# FastAPI Overview
FastAPI is a modern, fast web framework for Python.

## Installation
Use pip to install FastAPI:
```bash
pip install fastapi
```

### Advanced Features
Dependency injection and automatic OpenAPI docs.
"""
        sections = parse_markdown_sections(md_text)
        self.assertEqual(len(sections), 3)

        headings = [h for h, _ in sections]
        self.assertEqual(headings, ["FastAPI Overview", "Installation", "Advanced Features"])
        self.assertIn("pip install fastapi", sections[1][1])
        self.assertIn("Dependency injection", sections[2][1])

    def test_parse_rst_sections(self):
        rst_text = """================
Flask Quickstart
================
Welcome to Flask.

A Minimal Application
---------------------
A minimal Flask application looks something like this:

.. code-block:: python

    from flask import Flask
    app = Flask(__name__)

Routing
~~~~~~~
Modern web applications use meaningful URLs.
"""
        sections = parse_rst_sections(rst_text)
        self.assertGreaterEqual(len(sections), 2)
        headings = [h for h, _ in sections]
        self.assertIn("A Minimal Application", headings)
        self.assertIn("Routing", headings)

    def test_split_text_into_token_windows(self):
        # 100 words repeated -> roughly ~500 tokens
        sample_para = "This is a detailed explanation of system architecture and design patterns. " * 10
        long_text = "\n\n".join([sample_para] * 6)

        windows = split_text_into_token_windows(long_text, target_tokens=300, overlap_tokens=40)
        self.assertGreater(len(windows), 1)
        for win in windows:
            tok_count = count_tokens(win)
            self.assertLessEqual(tok_count, 450)


class TestChunkerRecords(unittest.TestCase):
    """Test chunk generation for docs, issues, and comments with metadata retention."""

    def setUp(self):
        self.chunker = Chunker()

    def test_chunk_doc_file_markdown(self):
        doc = {
            "repo": "pydantic/pydantic",
            "file_path": "docs/usage.md",
            "title": "Usage Guide",
            "format": "md",
            "url": "https://github.com/pydantic/pydantic/blob/main/docs/usage.md",
            "content": "# Basic Usage\n\nDefine models like this:\n\n```python\nclass User(BaseModel): pass\n```\n\n# Validation\n\nValidation happens automatically.",
        }
        chunks = self.chunker.chunk_doc_file(doc)
        self.assertEqual(len(chunks), 2)

        c1 = chunks[0]
        self.assertEqual(c1["repo"], "pydantic/pydantic")
        self.assertEqual(c1["source_type"], "doc")
        self.assertEqual(c1["file_path"], "docs/usage.md")
        self.assertEqual(c1["heading"], "Basic Usage")
        self.assertIn("#basic-usage", c1["url"])
        self.assertIn("[pydantic/pydantic - docs/usage.md] Basic Usage", c1["text"])
        self.assertTrue(len(c1["content_hash"]) == 64)

        c2 = chunks[1]
        self.assertEqual(c2["heading"], "Validation")
        self.assertIn("#validation", c2["url"])

    def test_chunk_doc_file_rst(self):
        doc = {
            "repo": "pallets/flask",
            "file_path": "docs/quickstart.rst",
            "title": "Flask Quickstart",
            "format": "rst",
            "url": "https://github.com/pallets/flask/blob/main/docs/quickstart.rst",
            "content": "Quickstart\n==========\n\nGetting started with Flask.\n\nRouting\n-------\n\nUse route decorators.",
        }
        chunks = self.chunker.chunk_doc_file(doc)
        self.assertGreaterEqual(len(chunks), 1)
        for c in chunks:
            self.assertEqual(c["repo"], "pallets/flask")
            self.assertEqual(c["source_type"], "doc")
            self.assertIsNotNone(c["chunk_id"])
            self.assertIsNotNone(c["content_hash"])

    def test_chunk_issue_attaches_title(self):
        issue = {
            "repository": "encode/httpx",
            "number": 1042,
            "title": "Connection pool timeout when keepalive is False",
            "body": "When creating a Client with keepalive=False, connections time out.\n\nHere is a reproducer script:\nclient = httpx.Client()",
            "html_url": "https://github.com/encode/httpx/issues/1042",
        }
        chunks = self.chunker.chunk_issue(issue)
        self.assertEqual(len(chunks), 1)
        c = chunks[0]
        self.assertEqual(c["repo"], "encode/httpx")
        self.assertEqual(c["source_type"], "issue")
        self.assertEqual(c["issue_number"], 1042)
        self.assertIn("[encode/httpx #1042] Connection pool timeout", c["text"])
        self.assertEqual(c["url"], "https://github.com/encode/httpx/issues/1042")

    def test_chunk_comment_tagged_with_issue_and_user(self):
        comment = {
            "repository": "encode/httpx",
            "issue_number": 1042,
            "comment_id": 987654,
            "user": "florimondmanca",
            "body": "This was fixed in PR #1045 by closing unused sockets immediately.",
            "html_url": "https://github.com/encode/httpx/issues/1042#issuecomment-987654",
        }
        chunks = self.chunker.chunk_comment(comment)
        self.assertEqual(len(chunks), 1)
        c = chunks[0]
        self.assertEqual(c["repo"], "encode/httpx")
        self.assertEqual(c["source_type"], "comment")
        self.assertEqual(c["issue_number"], 1042)
        self.assertIn("Comment by @florimondmanca", c["text"])
        self.assertEqual(c["url"], "https://github.com/encode/httpx/issues/1042#issuecomment-987654")


class TestStorageEngine(unittest.TestCase):
    """Test SQLite persistence and JSONL export."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test.db"
        self.processed_dir = Path(self.temp_dir.name) / "processed"
        self.storage = StorageEngine(db_path=self.db_path, processed_dir=self.processed_dir)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_save_and_retrieve_counts(self):
        issues = [{
            "repository": "pallets/flask",
            "number": 1,
            "title": "First issue",
            "body": "Sample body",
            "state": "closed",
            "labels": ["bug"],
            "created_at": "2023-01-01",
            "updated_at": "2023-01-02",
            "html_url": "https://github.com/pallets/flask/issues/1",
            "comment_count": 1,
        }]
        comments = [{
            "repository": "pallets/flask",
            "issue_number": 1,
            "comment_id": 101,
            "user": "mitsuhiko",
            "body": "Fixed in master.",
            "created_at": "2023-01-02",
            "html_url": "https://github.com/pallets/flask/issues/1#c101",
        }]
        doc_files = [{
            "repo": "pallets/flask",
            "file_path": "README.rst",
            "title": "Flask",
            "content": "Flask README",
            "format": "rst",
            "url": "https://github.com/pallets/flask",
        }]
        chunks = [{
            "chunk_id": "testchunk001",
            "repo": "pallets/flask",
            "source_type": "doc",
            "issue_number": None,
            "file_path": "README.rst",
            "heading": "Flask",
            "url": "https://github.com/pallets/flask",
            "content_hash": "testhash123",
            "text": "Flask README chunk",
            "token_count": 15,
        }]

        self.storage.save_issues(issues)
        self.storage.save_comments(comments)
        self.storage.save_doc_files(doc_files)
        self.storage.save_chunks(chunks, repo="pallets/flask")

        counts = self.storage.get_repo_counts("pallets/flask")
        self.assertEqual(counts["issues"], 1)
        self.assertEqual(counts["comments"], 1)
        self.assertEqual(counts["doc_files"], 1)
        self.assertEqual(counts["chunks"], 1)

        # Verify JSONL file was also generated
        jsonl_file = self.processed_dir / "pallets_flask_chunks.jsonl"
        self.assertTrue(jsonl_file.exists())


if __name__ == "__main__":
    unittest.main()
