"""
Retrieval Ground-Truth Dataset Builder for RepoTriage
====================================================
Mines GitHub duplicate closures, duplicate labels, and resolution cross-references
from the ingested corpus to construct a high-fidelity evaluation set without manual labeling.

Per RepoTriage_SPEC.md Section 8A:
- Query = duplicate issue (title + body)
- Relevant target = canonical referenced issue (#N)
- If canonical #N is outside the ingested window, fetches it on-demand via fetch_single_issue
  and chunks/indexes it so target chunks are never missing from the corpus.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from pathlib import Path
from typing import Any

from src.ingest.chunker import Chunker
from src.ingest.github_client import fetch_single_issue
from src.ingest.storage import StorageEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "processed" / "repotriage.db"
OUTPUT_DIR = PROJECT_ROOT / "eval"
OUTPUT_FILE = OUTPUT_DIR / "retrieval_ground_truth.json"

# Regex patterns for duplicate and resolution cross-references
PATTERNS = [
    ("duplicate", re.compile(r"(?:duplicate\s+of|dupe\s+of|duplicates)\s+#(\d+)", re.I)),
    ("duplicate", re.compile(r"(?:closing\s+in\s+favor\s+of|closed\s+in\s+favor\s+of)\s+#(\d+)", re.I)),
    ("duplicate", re.compile(r"(?:same\s+issue\s+as|same\s+as|tracked\s+in)\s+#(\d+)", re.I)),
    ("superseded", re.compile(r"(?:superseded\s+by|replaced\s+by)\s+#(\d+)", re.I)),
    ("cross_ref", re.compile(r"(?:see\s+also|see|related\s+to|fixed\s+by|fixed\s+in)\s+#(\d+)", re.I)),
]


def extract_references_from_text(text: str) -> list[tuple[str, int, str]]:
    """Return list of (relation_type, target_issue_number, matched_snippet)."""
    if not text:
        return []
    results = []
    for rel, pat in PATTERNS:
        for m in pat.finditer(text):
            target_num = int(m.group(1))
            start = max(0, m.start() - 30)
            end = min(len(text), m.end() + 30)
            snippet = text[start:end].replace("\n", " ").strip()
            results.append((rel, target_num, snippet))
    return results


def build_ground_truth(
    db_path: Path = DB_PATH,
    output_path: Path = OUTPUT_FILE,
    target_pair_count: int = 60,
) -> list[dict[str, Any]]:
    """Extract, verify, and persist retrieval evaluation pairs."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    storage = StorageEngine(db_path=db_path)
    chunker = Chunker()

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Load all issues
    cursor.execute("SELECT repo, number, title, body, state, labels, created_at, html_url FROM issues")
    issues = [dict(r) for r in cursor.fetchall()]
    issue_lookup = {(i["repo"], i["number"]): i for i in issues}

    # Load all comments
    cursor.execute("SELECT repo, issue_number, user, body, created_at FROM comments")
    comments = [dict(r) for r in cursor.fetchall()]
    conn.close()

    logger.info("Loaded %d issues and %d comments from %s", len(issues), len(comments), db_path.name)

    candidate_map: dict[tuple[str, int], dict[str, Any]] = {}

    # 1. Check issue bodies and titles
    for issue in issues:
        repo = issue["repo"]
        q_num = issue["number"]
        text = f"{issue.get('title', '')}\n\n{issue.get('body', '')}"

        # Also check labels for duplicate tag
        labels = json.loads(issue.get("labels") or "[]")
        has_dup_label = any("duplicate" in str(l).lower() for l in labels)

        refs = extract_references_from_text(text)
        for rel, t_num, snippet in refs:
            if t_num != q_num:
                key = (repo, q_num)
                if key not in candidate_map:
                    candidate_map[key] = {
                        "repo": repo,
                        "query_number": q_num,
                        "target_number": t_num,
                        "relation": "duplicate" if has_dup_label else rel,
                        "evidence_snippet": snippet,
                    }

    # 2. Check comments
    for comment in comments:
        repo = comment["repo"]
        q_num = comment["issue_number"]
        c_body = comment.get("body", "")

        refs = extract_references_from_text(c_body)
        for rel, t_num, snippet in refs:
            if t_num != q_num:
                key = (repo, q_num)
                # Comments from maintainers closing duplicates are top priority
                if key not in candidate_map or rel == "duplicate":
                    candidate_map[key] = {
                        "repo": repo,
                        "query_number": q_num,
                        "target_number": t_num,
                        "relation": rel,
                        "evidence_snippet": snippet,
                    }

    logger.info("Found %d candidate query-target pairs across corpus.", len(candidate_map))

    # 3. Verify each pair and fetch missing canonical issues on-demand
    verified_pairs: list[dict[str, Any]] = []
    missing_targets: set[tuple[str, int]] = set()

    for (repo, q_num), cand in candidate_map.items():
        t_num = cand["target_number"]
        if (repo, t_num) not in issue_lookup:
            missing_targets.add((repo, t_num))

    if missing_targets:
        logger.info(
            "Fetching %d missing canonical issues on-demand via fetch_single_issue...",
            len(missing_targets),
        )
        for repo, t_num in missing_targets:
            owner, rname = repo.split("/", 1)
            fetched = fetch_single_issue(owner, rname, t_num, fetch_comments=True)
            if fetched and isinstance(fetched, dict) and fetched.get("title"):
                # Save to DB and chunks
                storage.save_issues([fetched])
                issue_lookup[(repo, t_num)] = {
                    "repo": repo,
                    "number": t_num,
                    "title": fetched.get("title", ""),
                    "body": fetched.get("body", ""),
                    "created_at": fetched.get("created_at", ""),
                    "html_url": fetched.get("html_url", ""),
                }
                new_chunks = chunker.chunk_issue(fetched)
                storage.save_chunks(new_chunks, repo=repo)

    # 4. Filter verified pairs
    for (repo, q_num), cand in candidate_map.items():
        t_num = cand["target_number"]
        q_issue = issue_lookup.get((repo, q_num))
        t_issue = issue_lookup.get((repo, t_num))

        if not q_issue or not t_issue:
            continue

        # Skip PRs or deleted targets
        q_title = q_issue.get("title", "").strip()
        t_title = t_issue.get("title", "").strip()
        if not q_title or not t_title:
            continue

        # Query text formulation (free text: title + body)
        q_body = (q_issue.get("body") or "").strip()
        query_text = f"{q_title}\n\n{q_body}" if q_body else q_title

        # Exclude trivial one-word queries
        if len(query_text.split()) < 3:
            continue

        verified_pairs.append({
            "query_id": f"{repo}#{q_num}",
            "repo": repo,
            "query_number": q_num,
            "query_title": q_title,
            "query_body": q_body,
            "query_text": query_text,
            "query_created_at": q_issue.get("created_at", ""),
            "target_issue_number": t_num,
            "target_title": t_title,
            "target_url": t_issue.get("html_url", ""),
            "relation": cand["relation"],
            "evidence_snippet": cand["evidence_snippet"],
        })

    logger.info("Successfully assembled %d verified ground-truth pairs.", len(verified_pairs))

    # Persist to JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(verified_pairs, f, indent=2, ensure_ascii=False)

    logger.info("Saved ground-truth evaluation set to %s", output_path)

    # Summary by repo
    repo_counts: dict[str, int] = {}
    for p in verified_pairs:
        repo_counts[p["repo"]] = repo_counts.get(p["repo"], 0) + 1

    print("\n" + "=" * 65)
    print("  RETRIEVAL EVALUATION GROUND-TRUTH DATASET")
    print("=" * 65)
    print(f"  Total Verified Query Pairs : {len(verified_pairs)}")
    for r, count in sorted(repo_counts.items()):
        print(f"    • {r:<30}: {count} pairs")
    print("=" * 65 + "\n")

    return verified_pairs


if __name__ == "__main__":
    build_ground_truth()
