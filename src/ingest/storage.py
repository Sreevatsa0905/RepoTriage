"""
SQLite and JSONL Storage Engine for RepoTriage
==============================================
Manages persistence of normalized issues, comments, doc files, and search chunks
in data/processed/repotriage.db and data/processed/*.jsonl.
"""

from __future__ import annotations

import contextlib
import json
import logging
from pathlib import Path
import sqlite3
from typing import Any, Iterator

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DB_PATH = PROCESSED_DIR / "repotriage.db"


class StorageEngine:
    """Manages SQLite schema and storage for the retrieval corpus."""

    def __init__(self, db_path: Path = DB_PATH, processed_dir: Path = PROCESSED_DIR):
        self.db_path = db_path
        self.processed_dir = processed_dir
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextlib.contextmanager
    def _get_connection(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        """Create tables and indexes if they do not exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Issues table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS issues (
                    id TEXT PRIMARY KEY,
                    repo TEXT NOT NULL,
                    number INTEGER NOT NULL,
                    title TEXT NOT NULL,
                    body TEXT,
                    state TEXT NOT NULL,
                    labels TEXT,
                    created_at TEXT,
                    updated_at TEXT,
                    html_url TEXT,
                    comment_count INTEGER DEFAULT 0
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_issues_repo ON issues(repo);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_issues_number ON issues(repo, number);")

            # Comments table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS comments (
                    id TEXT PRIMARY KEY,
                    repo TEXT NOT NULL,
                    issue_number INTEGER NOT NULL,
                    comment_id INTEGER NOT NULL,
                    user TEXT,
                    body TEXT,
                    created_at TEXT,
                    html_url TEXT
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_comments_issue ON comments(repo, issue_number);")

            # Doc files table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS doc_files (
                    id TEXT PRIMARY KEY,
                    repo TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    title TEXT,
                    content TEXT,
                    format TEXT,
                    url TEXT
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_doc_files_repo ON doc_files(repo);")

            # Chunks table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    repo TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    issue_number INTEGER,
                    file_path TEXT,
                    heading TEXT,
                    url TEXT,
                    content_hash TEXT NOT NULL,
                    text TEXT NOT NULL,
                    token_count INTEGER NOT NULL
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_repo ON chunks(repo);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_type ON chunks(source_type);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_hash ON chunks(content_hash);")

            conn.commit()

    def save_issues(self, issues: list[dict[str, Any]]) -> int:
        """Insert or replace normalized issue records."""
        if not issues:
            return 0
        with self._get_connection() as conn:
            cursor = conn.cursor()
            rows = [
                (
                    f"{i['repository']}#{i['number']}",
                    i["repository"],
                    i["number"],
                    i.get("title", ""),
                    i.get("body", ""),
                    i.get("state", "unknown"),
                    json.dumps(i.get("labels", []), ensure_ascii=False),
                    i.get("created_at", ""),
                    i.get("updated_at", ""),
                    i.get("html_url", ""),
                    i.get("comment_count", 0),
                )
                for i in issues
            ]
            cursor.executemany("""
                INSERT OR REPLACE INTO issues
                (id, repo, number, title, body, state, labels, created_at, updated_at, html_url, comment_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, rows)
            conn.commit()
            return len(rows)

    def save_comments(self, comments: list[dict[str, Any]]) -> int:
        """Insert or replace normalized comment records."""
        if not comments:
            return 0
        with self._get_connection() as conn:
            cursor = conn.cursor()
            rows = [
                (
                    f"{c['repository']}#{c['issue_number']}_{c['comment_id']}",
                    c["repository"],
                    c["issue_number"],
                    c["comment_id"],
                    c.get("user", "unknown"),
                    c.get("body", ""),
                    c.get("created_at", ""),
                    c.get("html_url", ""),
                )
                for c in comments
            ]
            cursor.executemany("""
                INSERT OR REPLACE INTO comments
                (id, repo, issue_number, comment_id, user, body, created_at, html_url)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, rows)
            conn.commit()
            return len(rows)

    def save_doc_files(self, doc_files: list[dict[str, Any]]) -> int:
        """Insert or replace document files."""
        if not doc_files:
            return 0
        with self._get_connection() as conn:
            cursor = conn.cursor()
            rows = [
                (
                    f"{d['repo']}:{d['file_path']}",
                    d["repo"],
                    d["file_path"],
                    d.get("title", ""),
                    d.get("content", ""),
                    d.get("format", "md"),
                    d.get("url", ""),
                )
                for d in doc_files
            ]
            cursor.executemany("""
                INSERT OR REPLACE INTO doc_files
                (id, repo, file_path, title, content, format, url)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, rows)
            conn.commit()
            return len(rows)

    def save_chunks(self, chunks: list[dict[str, Any]], repo: str | None = None) -> int:
        """Insert or replace chunks into SQLite and export to processed JSONL."""
        if not chunks:
            return 0
        with self._get_connection() as conn:
            cursor = conn.cursor()
            rows = [
                (
                    c["chunk_id"],
                    c["repo"],
                    c["source_type"],
                    c.get("issue_number"),
                    c.get("file_path"),
                    c.get("heading"),
                    c.get("url"),
                    c["content_hash"],
                    c["text"],
                    c["token_count"],
                )
                for c in chunks
            ]
            cursor.executemany("""
                INSERT OR REPLACE INTO chunks
                (chunk_id, repo, source_type, issue_number, file_path, heading, url, content_hash, text, token_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, rows)
            conn.commit()

        # Save to processed JSONL
        if repo:
            safe_name = repo.replace("/", "_")
            out_jsonl = self.processed_dir / f"{safe_name}_chunks.jsonl"
            with open(out_jsonl, "w", encoding="utf-8") as f:
                for c in chunks:
                    f.write(json.dumps(c, ensure_ascii=False) + "\n")

        return len(rows)

    def get_repo_counts(self, repo: str) -> dict[str, int]:
        """Return counts of issues, comments, doc files, and chunks for a repo."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM issues WHERE repo = ?", (repo,))
            issue_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM comments WHERE repo = ?", (repo,))
            comment_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM doc_files WHERE repo = ?", (repo,))
            doc_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM chunks WHERE repo = ?", (repo,))
            chunk_count = cursor.fetchone()[0]

            return {
                "issues": issue_count,
                "comments": comment_count,
                "doc_files": doc_count,
                "chunks": chunk_count,
            }

    def get_all_counts(self) -> dict[str, dict[str, int]]:
        """Return counts across all repositories in SQLite."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT repo FROM (SELECT repo FROM issues UNION SELECT repo FROM doc_files)")
            repos = [row[0] for row in cursor.fetchall()]

        return {r: self.get_repo_counts(r) for r in sorted(repos)}
