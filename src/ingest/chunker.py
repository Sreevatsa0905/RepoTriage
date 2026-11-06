"""
Text Chunker for RepoTriage
============================
Splits repository documentation (.md and .rst), issues, and comments into
semantic ~250-350 token chunks with metadata retention and anchor generation.
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

TARGET_CHUNK_TOKENS = 300
MAX_CHUNK_TOKENS = 380
MIN_CHUNK_TOKENS = 40
OVERLAP_TOKENS = 40


def count_tokens(text: str) -> int:
    """Fast approximate token count based on whitespace and punctuation splitting."""
    if not text:
        return 0
    words = text.split()
    # Punctuation / code symbols add ~15-20% over raw word count
    return max(1, int(len(words) * 1.15))


def slugify(text: str) -> str:
    """Create a URL-safe anchor slug from heading text."""
    slug = text.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s_-]+", "-", slug).strip("-")
    return slug


def split_text_into_token_windows(
    text: str,
    target_tokens: int = TARGET_CHUNK_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS,
) -> list[str]:
    """
    Split a block of text into ~target_tokens windows with overlap,
    respecting paragraph breaks where possible.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        paragraphs = [text.strip()] if text.strip() else []

    chunks: list[str] = []
    current_tokens: list[str] = []
    current_count = 0

    for para in paragraphs:
        para_words = para.split()
        para_token_estimate = max(1, int(len(para_words) * 1.15))

        # If a single paragraph is larger than MAX_CHUNK_TOKENS, split it by words
        if para_token_estimate > MAX_CHUNK_TOKENS:
            # Flush existing buffer
            if current_tokens:
                chunks.append("\n\n".join(current_tokens))
                current_tokens = []
                current_count = 0

            # Split large paragraph into sliding word windows
            words = para.split()
            step = max(1, int((target_tokens - overlap_tokens) / 1.15))
            window_size = max(1, int(target_tokens / 1.15))

            for start in range(0, len(words), step):
                window = words[start : start + window_size]
                if window:
                    chunks.append(" ".join(window))
                if start + window_size >= len(words):
                    break
            continue

        if current_count + para_token_estimate > MAX_CHUNK_TOKENS and current_tokens:
            chunks.append("\n\n".join(current_tokens))
            # Keep the last paragraph for overlap if it fits
            if current_count > overlap_tokens and len(current_tokens) > 1:
                last_para = current_tokens[-1]
                current_tokens = [last_para, para]
                current_count = count_tokens(last_para) + para_token_estimate
            else:
                current_tokens = [para]
                current_count = para_token_estimate
        else:
            current_tokens.append(para)
            current_count += para_token_estimate

    if current_tokens:
        chunks.append("\n\n".join(current_tokens))

    return chunks


# ---------------------------------------------------------------------------
# Document Parsing (.md and .rst)
# ---------------------------------------------------------------------------

def parse_markdown_sections(content: str) -> list[tuple[str, str]]:
    """
    Split markdown content into (heading, body) sections.
    Recognizes `#`, `##`, `###`, `####`, etc.
    """
    lines = content.splitlines()
    sections: list[tuple[str, str]] = []

    current_heading = "Overview"
    current_lines: list[str] = []

    heading_pattern = re.compile(r"^(#{1,6})\s+(.+)$")

    for line in lines:
        match = heading_pattern.match(line.strip())
        if match:
            if current_lines:
                sections.append((current_heading, "\n".join(current_lines).strip()))
                current_lines = []
            current_heading = match.group(2).strip()
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_heading, "\n".join(current_lines).strip()))

    return [(h, b) for h, b in sections if b]


def parse_rst_sections(content: str) -> list[tuple[str, str]]:
    """
    Split reStructuredText content into (heading, body) sections.
    Recognizes underline headings with repeated punctuation: ===, ---, ~~~, ^^^, etc.
    """
    lines = content.splitlines()
    sections: list[tuple[str, str]] = []

    current_heading = "Overview"
    current_lines: list[str] = []

    rst_underline_re = re.compile(r"^[=\-~^\"'`#*]{3,}$")

    i = 0
    while i < len(lines):
        line = lines[i]
        next_line = lines[i + 1] if i + 1 < len(lines) else ""

        # Check for underline heading
        if (
            line.strip()
            and next_line.strip()
            and rst_underline_re.match(next_line.strip())
            and len(next_line.strip()) >= len(line.strip()) - 2
        ):
            if current_lines:
                sections.append((current_heading, "\n".join(current_lines).strip()))
                current_lines = []
            current_heading = line.strip()
            i += 2  # skip heading and underline
            continue

        current_lines.append(line)
        i += 1

    if current_lines:
        sections.append((current_heading, "\n".join(current_lines).strip()))

    return [(h, b) for h, b in sections if b]


# ---------------------------------------------------------------------------
# Chunker Class
# ---------------------------------------------------------------------------

class Chunker:
    """Converts issues, comments, and doc files into indexed retrieval chunks."""

    @staticmethod
    def make_chunk_id(repo: str, source_type: str, item_id: str | int, chunk_idx: int) -> str:
        raw_key = f"{repo}:{source_type}:{item_id}:{chunk_idx}"
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:16]

    def chunk_doc_file(self, doc_record: dict[str, Any]) -> list[dict[str, Any]]:
        """Chunk a documentation file (.md or .rst)."""
        repo = doc_record["repo"]
        file_path = doc_record["file_path"]
        content = doc_record["content"]
        fmt = doc_record.get("format", "md")
        base_url = doc_record["url"]

        if fmt == "rst":
            sections = parse_rst_sections(content)
        else:
            sections = parse_markdown_sections(content)

        if not sections:
            sections = [(doc_record.get("title", file_path), content)]

        chunks: list[dict[str, Any]] = []
        chunk_idx = 0

        for heading, section_text in sections:
            anchor = slugify(heading)
            section_url = f"{base_url}#{anchor}" if anchor else base_url
            sub_chunks = split_text_into_token_windows(section_text)

            for sub_text in sub_chunks:
                if not sub_text.strip():
                    continue

                full_chunk_text = f"[{repo} - {file_path}] {heading}\n\n{sub_text}"
                cid = self.make_chunk_id(repo, "doc", f"{file_path}_{chunk_idx}", chunk_idx)
                content_hash = hashlib.sha256(full_chunk_text.encode("utf-8")).hexdigest()

                chunks.append({
                    "chunk_id": cid,
                    "repo": repo,
                    "source_type": "doc",
                    "issue_number": None,
                    "file_path": file_path,
                    "heading": heading,
                    "url": section_url,
                    "content_hash": content_hash,
                    "text": full_chunk_text,
                    "token_count": count_tokens(full_chunk_text),
                })
                chunk_idx += 1

        return chunks

    def chunk_issue(self, issue: dict[str, Any]) -> list[dict[str, Any]]:
        """Chunk a normalized issue record, attaching title to every chunk."""
        repo = issue["repository"]
        num = issue["number"]
        title = issue.get("title", "")
        body = issue.get("body", "")
        url = issue.get("html_url", f"https://github.com/{repo}/issues/{num}")

        combined_body = body.strip() if body else "No issue description provided."
        windows = split_text_into_token_windows(combined_body)
        if not windows:
            windows = [combined_body]

        chunks: list[dict[str, Any]] = []
        for idx, win in enumerate(windows):
            part_str = f" (Part {idx + 1}/{len(windows)})" if len(windows) > 1 else ""
            full_text = f"[{repo} #{num}] {title}{part_str}\n\n{win}"
            cid = self.make_chunk_id(repo, "issue", num, idx)
            content_hash = hashlib.sha256(full_text.encode("utf-8")).hexdigest()

            chunks.append({
                "chunk_id": cid,
                "repo": repo,
                "source_type": "issue",
                "issue_number": num,
                "file_path": None,
                "heading": title,
                "url": url,
                "content_hash": content_hash,
                "text": full_text,
                "token_count": count_tokens(full_text),
            })

        return chunks

    def chunk_comment(self, comment: dict[str, Any]) -> list[dict[str, Any]]:
        """Chunk an issue comment record, tagging with the parent issue."""
        repo = comment["repository"]
        issue_number = comment["issue_number"]
        comment_id = comment["comment_id"]
        user = comment.get("user", "unknown")
        body = comment.get("body", "").strip()
        url = comment.get("html_url", f"https://github.com/{repo}/issues/{issue_number}#issuecomment-{comment_id}")

        if not body:
            return []

        windows = split_text_into_token_windows(body)
        if not windows:
            windows = [body]

        chunks: list[dict[str, Any]] = []
        for idx, win in enumerate(windows):
            full_text = f"[{repo} #{issue_number} Comment by @{user}]\n\n{win}"
            cid = self.make_chunk_id(repo, "comment", f"{issue_number}_{comment_id}", idx)
            content_hash = hashlib.sha256(full_text.encode("utf-8")).hexdigest()

            chunks.append({
                "chunk_id": cid,
                "repo": repo,
                "source_type": "comment",
                "issue_number": issue_number,
                "file_path": None,
                "heading": f"Comment on #{issue_number} by @{user}",
                "url": url,
                "content_hash": content_hash,
                "text": full_text,
                "token_count": count_tokens(full_text),
            })

        return chunks
