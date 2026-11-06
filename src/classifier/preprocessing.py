"""
Text preprocessing and validation for GitHub issue classification.
"""

from __future__ import annotations

import re
from typing import Any

from src.classifier.config import EXPECTED_LABELS

LABEL_TO_ID: dict[str, int] = {label: idx for idx, label in enumerate(EXPECTED_LABELS)}
ID_TO_LABEL: dict[int, str] = {idx: label for idx, label in enumerate(EXPECTED_LABELS)}


def validate_issue_input(title: Any, body: Any) -> tuple[str, str]:
    """
    Validate and coerce issue title and body inputs.

    Handles None, empty values, non-string types, and strips leading/trailing whitespace.
    """
    if title is None:
        title_str = ""
    elif isinstance(title, str):
        title_str = title.strip()
    else:
        title_str = str(title).strip()

    if body is None:
        body_str = ""
    elif isinstance(body, str):
        body_str = body.strip()
    else:
        body_str = str(body).strip()

    return title_str, body_str


def combine_issue_text(title: str, body: str) -> str:
    """
    Combine issue title and body into a single coherent text representation.

    If both are empty, returns an empty string or placeholder token to allow vectorization.
    """
    t, b = validate_issue_input(title, body)
    if not t and not b:
        return ""
    if not b:
        return t
    if not t:
        return b
    return f"{t} {b}"


def clean_markdown_and_noise(text: str) -> str:
    """
    Optional text cleaner that normalises markdown formatting, code blocks,
    and excessive whitespace while preserving code identifiers and error keywords.
    """
    if not text:
        return ""
    # Normalize excessive newlines and whitespace
    text = re.sub(r"[\r\n]+", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def encode_labels(labels: list[str]) -> list[int]:
    """Encode string labels into integer class IDs."""
    encoded = []
    for lbl in labels:
        lbl_clean = lbl.strip().lower()
        if lbl_clean not in LABEL_TO_ID:
            raise ValueError(f"Unknown label '{lbl}'. Expected one of {EXPECTED_LABELS}")
        encoded.append(LABEL_TO_ID[lbl_clean])
    return encoded


def decode_labels(ids: list[int]) -> list[str]:
    """Decode integer class IDs back into string labels."""
    decoded = []
    for i in ids:
        if i not in ID_TO_LABEL:
            raise ValueError(f"Unknown class ID '{i}'. Expected one of {list(ID_TO_LABEL.keys())}")
        decoded.append(ID_TO_LABEL[i])
    return decoded
