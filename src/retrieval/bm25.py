"""
Sparse Retrieval with BM25 for RepoTriage
=========================================
Implements custom tokenization optimized for software issues and documentation
(preserving code identifiers, error types, versions, and paths) and BM25 scoring.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

# Pattern to tokenize text while preserving code tokens, paths, error types, versions
# Matches:
# 1. URLs: https?://...
# 2. File paths: path/to/file.ext
# 3. Dotted/scoped identifiers: pydantic.v2, module.class.method
# 4. Version strings: v2.0, 3.12.1
# 5. CamelCase words: ValidationError, BaseModel
# 6. snake_case / kebab-case words: custom_validator, keep-alive
# 7. Standard alphanumeric tokens
_TOKEN_PATTERN = re.compile(
    r"""
    https?://\S+                                    # URLs
    | (?:[\w.-]+/[-\w./]+)                          # File paths
    | (?:[a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)+)          # Dotted names: foo.bar.baz
    | (?:v?\d+(?:\.\d+)+(?:[a-zA-Z0-9_-]*)?)       # Version numbers: v2.0, 1.2.3b1
    | (?:[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)+)        # CamelCase identifiers
    | (?:[a-zA-Z0-9]+(?:[-_][a-zA-Z0-9]+)+)        # snake_case / kebab-case
    | (?:\w+)                                       # Standard words
    """,
    re.VERBOSE,
)

# Common english stopwords
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
    "to", "was", "were", "will", "with",
}


def tokenize_code_text(text: str) -> list[str]:
    """
    Tokenize code and text while preserving technical identifiers,
    expanding subwords (e.g. CamelCase and snake_case) alongside the original tokens.
    """
    if not text:
        return []

    tokens: list[str] = []
    raw_matches = _TOKEN_PATTERN.findall(text)

    for match in raw_matches:
        cleaned = match.strip().lower()
        if not cleaned:
            continue

        # Keep original token
        if cleaned not in _STOPWORDS:
            tokens.append(cleaned)

        # Sub-tokenization for snake_case and kebab-case
        if "_" in match or "-" in match:
            subparts = re.split(r"[_\-]+", match)
            for sub in subparts:
                s_lower = sub.lower()
                if len(s_lower) > 1 and s_lower not in _STOPWORDS:
                    tokens.append(s_lower)

        # Sub-tokenization for CamelCase (e.g. ValidationError -> validation, error)
        camel_subparts = re.findall(r"[A-Z][a-z0-9]*|[a-z0-9]+", match)
        if len(camel_subparts) > 1:
            for sub in camel_subparts:
                s_lower = sub.lower()
                if len(s_lower) > 1 and s_lower not in _STOPWORDS:
                    tokens.append(s_lower)

    return tokens


class BM25Retriever:
    """Sparse retrieval engine using BM25Okapi over text chunks."""

    def __init__(self, chunks: list[dict[str, Any]]):
        self.chunks = chunks
        self.corpus_tokens = [tokenize_code_text(c["text"]) for c in chunks]
        self.bm25 = BM25Okapi(self.corpus_tokens) if self.corpus_tokens else None

    def search(
        self,
        query: str,
        top_k: int = 50,
        source_types: Sequence[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Search chunks and return top_k results with BM25 scores and ranks."""
        if not self.bm25 or not self.chunks:
            return []

        query_tokens = tokenize_code_text(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)

        # Filter by source_types before truncating to top_k
        if source_types:
            st_set = set(source_types)
            candidate_indices = [i for i, c in enumerate(self.chunks) if c.get("source_type") in st_set]
        else:
            candidate_indices = list(range(len(self.chunks)))

        top_indices = sorted(candidate_indices, key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for rank, idx in enumerate(top_indices, 1):
            score = float(scores[idx])
            if score <= 0.0:
                continue
            chunk = dict(self.chunks[idx])
            chunk["bm25_score"] = score
            chunk["bm25_rank"] = rank
            results.append(chunk)

        return results
