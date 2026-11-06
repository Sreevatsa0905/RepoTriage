"""
Documentation Loader for RepoTriage
====================================
Clones repository documentation using git clone --depth 1 into data/raw/repos/
and parses README, CONTRIBUTING, and docs/**/*.{md,mdx,rst}.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
REPOS_DIR = PROJECT_ROOT / "data" / "raw" / "repos"


class DocsLoader:
    """Manages cloning and loading of repository documentation files."""

    def __init__(self, repos_dir: Path = REPOS_DIR):
        self.repos_dir = repos_dir
        self.repos_dir.mkdir(parents=True, exist_ok=True)

    def clone_repo(self, owner: str, repo: str) -> Path:
        """
        Shallow clone (depth 1) a repository into data/raw/repos/<owner>_<repo>.
        Skips cloning if the repository directory already exists.
        """
        repo_dir_name = f"{owner}_{repo}"
        target_dir = self.repos_dir / repo_dir_name

        if target_dir.exists() and any(target_dir.iterdir()):
            logger.info("Documentation repository already exists at %s, skipping clone.", target_dir)
            return target_dir

        clone_url = f"https://github.com/{owner}/{repo}.git"
        logger.info("Cloning documentation for %s/%s into %s (depth 1)...", owner, repo, target_dir)

        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", clone_url, str(target_dir)],
                check=True,
                capture_output=True,
                text=True,
            )
            logger.info("Successfully cloned %s/%s.", owner, repo)
        except subprocess.CalledProcessError as e:
            logger.error("Failed to clone %s: %s\nStderr: %s", clone_url, e, e.stderr)
            raise RuntimeError(f"git clone failed for {clone_url}: {e.stderr}") from e

        return target_dir

    def extract_title(self, content: str, file_path: Path) -> str:
        """Extract a readable title from the document or fallback to filename."""
        # Check Markdown heading: # Title
        for line in content.splitlines()[:15]:
            line = line.strip()
            if line.startswith("# "):
                return line[2:].strip()

        # Check reStructuredText title (line followed by underline ===== or -----)
        lines = content.splitlines()[:20]
        for i in range(len(lines) - 1):
            cur = lines[i].strip()
            nxt = lines[i + 1].strip()
            if cur and len(nxt) >= len(cur) and len(nxt) >= 3:
                if re.match(r"^[=\-~^\"'`#*]{3,}$", nxt):
                    return cur

        return file_path.stem.replace("_", " ").replace("-", " ").title()

    def load_docs(self, owner: str, repo: str) -> list[dict[str, Any]]:
        """
        Scan and load README, CONTRIBUTING, and docs/**/*.{md,mdx,rst} files.
        """
        target_dir = self.clone_repo(owner, repo)
        doc_files: list[dict[str, Any]] = []
        seen_paths: set[str] = set()

        # Target patterns
        patterns = [
            "README*",
            "CONTRIBUTING*",
            "docs/**/*.md",
            "docs/**/*.mdx",
            "docs/**/*.rst",
            "doc/**/*.rst",
            "doc/**/*.md",
        ]

        found_files: list[Path] = []
        for pat in patterns:
            for p in target_dir.glob(pat):
                if p.is_file() and p.suffix.lower() in [".md", ".mdx", ".rst", ""]:
                    rel_str = str(p.relative_to(target_dir)).replace("\\", "/")
                    if rel_str not in seen_paths:
                        seen_paths.add(rel_str)
                        found_files.append(p)

        logger.info("Found %d documentation files for %s/%s", len(found_files), owner, repo)

        for p in found_files:
            rel_path = str(p.relative_to(target_dir)).replace("\\", "/")
            try:
                content = p.read_text(encoding="utf-8", errors="replace")
            except Exception as e:
                logger.warning("Could not read %s: %s", p, e)
                continue

            if not content.strip():
                continue

            suffix = p.suffix.lower().lstrip(".")
            fmt = "rst" if suffix == "rst" else "md"
            title = self.extract_title(content, p)
            url = f"https://github.com/{owner}/{repo}/blob/main/{rel_path}"

            doc_files.append({
                "id": f"{owner}/{repo}:{rel_path}",
                "repo": f"{owner}/{repo}",
                "file_path": rel_path,
                "title": title,
                "content": content,
                "format": fmt,
                "url": url,
            })

        logger.info("Loaded %d valid documentation files for %s/%s", len(doc_files), owner, repo)
        return doc_files
