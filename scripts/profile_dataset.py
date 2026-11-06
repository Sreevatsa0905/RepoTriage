#!/usr/bin/env python3
"""
Dataset validation and profiling pipeline for NLBSE 2024.

Validates and profiles:
- Exactly 1,500 training and 1,500 testing records
- 5 repositories, 3 classes
- Class distribution overall and per repository
- Missing titles, bodies, empty records
- Duplicates and train/test overlap
- Text length distributions (chars, words)
- Generates DATASET.md and summary JSON
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any

# Increase CSV field size limit for large bodies (safe 32-bit limit)
csv.field_size_limit(2147483647)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.classifier.config import (
    TRAIN_CSV,
    TEST_CSV,
    EXPECTED_REPOS,
    EXPECTED_LABELS,
    EXPECTED_TRAIN_COUNT,
    EXPECTED_TEST_COUNT,
    PROJECT_ROOT,
    RESULTS_DIR,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_csv(path: Path) -> list[dict[str, str]]:
    """Load records from CSV using DictReader."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset file missing: {path}")
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        return list(reader)


def compute_text_stats(lengths: list[int]) -> dict[str, float]:
    """Compute min, max, mean, median for a sequence of numeric lengths."""
    if not lengths:
        return {"min": 0, "max": 0, "mean": 0.0, "median": 0.0}
    s = sorted(lengths)
    n = len(s)
    mean = sum(s) / n
    median = s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0
    return {
        "min": int(s[0]),
        "max": int(s[-1]),
        "mean": round(mean, 1),
        "median": round(median, 1),
    }


def profile_split(records: list[dict[str, str]], split_name: str) -> dict[str, Any]:
    """Profile a single dataset split."""
    count = len(records)
    repos = Counter(r["repo"] for r in records)
    labels = Counter(r["label"] for r in records)

    # Repo x Label contingency table
    repo_label_dist: dict[str, dict[str, int]] = {}
    for repo in EXPECTED_REPOS:
        repo_label_dist[repo] = {}
        for label in EXPECTED_LABELS:
            repo_label_dist[repo][label] = 0

    missing_titles = 0
    missing_bodies = 0
    empty_titles = 0
    empty_bodies = 0
    empty_both = 0

    title_char_lens: list[int] = []
    body_char_lens: list[int] = []
    comb_char_lens: list[int] = []
    title_word_lens: list[int] = []
    body_word_lens: list[int] = []
    comb_word_lens: list[int] = []

    seen_signatures: Counter[tuple[str, str]] = Counter()

    for r in records:
        repo = r.get("repo", "")
        label = r.get("label", "")
        title = r.get("title") or ""
        body = r.get("body") or ""

        if repo in repo_label_dist and label in repo_label_dist[repo]:
            repo_label_dist[repo][label] += 1

        t_clean = title.strip()
        b_clean = body.strip()

        if r.get("title") is None:
            missing_titles += 1
        elif len(t_clean) == 0:
            empty_titles += 1

        if r.get("body") is None:
            missing_bodies += 1
        elif len(b_clean) == 0:
            empty_bodies += 1

        if len(t_clean) == 0 and len(b_clean) == 0:
            empty_both += 1

        # Text lengths
        comb_text = f"{t_clean} {b_clean}".strip()
        title_char_lens.append(len(t_clean))
        body_char_lens.append(len(b_clean))
        comb_char_lens.append(len(comb_text))

        title_word_lens.append(len(t_clean.split()))
        body_word_lens.append(len(b_clean.split()))
        comb_word_lens.append(len(comb_text.split()))

        seen_signatures[(repo, t_clean)] += 1

    internal_duplicates = sum(c - 1 for c in seen_signatures.values() if c > 1)

    return {
        "split": split_name,
        "count": count,
        "repos": dict(repos),
        "labels": dict(labels),
        "repo_label_contingency": repo_label_dist,
        "data_quality": {
            "missing_titles": missing_titles,
            "empty_titles": empty_titles,
            "missing_bodies": missing_bodies,
            "empty_bodies": empty_bodies,
            "empty_both": empty_both,
            "internal_duplicates": internal_duplicates,
        },
        "text_statistics": {
            "title_characters": compute_text_stats(title_char_lens),
            "title_words": compute_text_stats(title_word_lens),
            "body_characters": compute_text_stats(body_char_lens),
            "body_words": compute_text_stats(body_word_lens),
            "combined_characters": compute_text_stats(comb_char_lens),
            "combined_words": compute_text_stats(comb_word_lens),
        },
    }


def check_train_test_overlap(
    train_records: list[dict[str, str]], test_records: list[dict[str, str]]
) -> dict[str, Any]:
    """Check for identical title or title+body pairs between train and test."""
    train_exact = {(r["repo"], r.get("title", "").strip(), r.get("body", "").strip()) for r in train_records}
    test_exact = {(r["repo"], r.get("title", "").strip(), r.get("body", "").strip()) for r in test_records}
    exact_overlap = len(train_exact.intersection(test_exact))

    train_titles = {(r["repo"], r.get("title", "").strip()) for r in train_records}
    test_titles = {(r["repo"], r.get("title", "").strip()) for r in test_records}
    title_overlap = len(train_titles.intersection(test_titles))

    return {
        "exact_record_overlap": exact_overlap,
        "title_repo_overlap": title_overlap,
    }


def run_profiling() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Run full validation and profiling pipeline."""
    logger.info("Loading training data: %s", TRAIN_CSV)
    train_records = load_csv(TRAIN_CSV)
    logger.info("Loading test data: %s", TEST_CSV)
    test_records = load_csv(TEST_CSV)

    train_profile = profile_split(train_records, "train")
    test_profile = profile_split(test_records, "test")
    overlap_info = check_train_test_overlap(train_records, test_records)

    # Validations
    assert train_profile["count"] == EXPECTED_TRAIN_COUNT, (
        f"Expected {EXPECTED_TRAIN_COUNT} train records, got {train_profile['count']}"
    )
    assert test_profile["count"] == EXPECTED_TEST_COUNT, (
        f"Expected {EXPECTED_TEST_COUNT} test records, got {test_profile['count']}"
    )

    for p in (train_profile, test_profile):
        found_repos = set(p["repos"].keys())
        assert found_repos == set(EXPECTED_REPOS), f"Mismatched repos: {found_repos}"
        found_labels = set(p["labels"].keys())
        assert found_labels == set(EXPECTED_LABELS), f"Mismatched labels: {found_labels}"
        for repo in EXPECTED_REPOS:
            for label in EXPECTED_LABELS:
                cell_count = p["repo_label_contingency"][repo][label]
                assert cell_count == 100, (
                    f"Expected 100 records for {repo}/{label} in {p['split']}, got {cell_count}"
                )

    logger.info("All schema and balance assertions PASSED.")
    return train_profile, test_profile, overlap_info


def generate_dataset_md(
    train_profile: dict[str, Any],
    test_profile: dict[str, Any],
    overlap: dict[str, Any],
    output_path: Path,
) -> None:
    """Generate DATASET.md."""
    md = f"""# NLBSE 2024 Issue Report Classification Dataset

## 1. Overview and Source

The dataset used in this microproject is the official **NLBSE'24 Tool Competition on Issue Report Classification** benchmark dataset.

- **Official Repository**: [https://github.com/nlbse2024/issue-report-classification](https://github.com/nlbse2024/issue-report-classification)
- **Competition Website**: [https://nlbse2024.github.io/tools/](https://nlbse2024.github.io/tools/)
- **Total Records**: 3,000 labelled GitHub issue reports
- **Splits**:
  - Training: exactly {train_profile['count']} records (`issues_train.csv`)
  - Testing: exactly {test_profile['count']} records (`issues_test.csv`)

## 2. Citation

```bibtex
@inproceedings{{nlbse2024,
  author={{Kallis, Rafael and Colavito, Giuseppe and Al-Kaswan, Ali and Pascarella, Luca and Chaparro, Oscar and Rani, Pooja}},
  title={{The NLBSE'24 Tool Competition}},
  booktitle={{Proceedings of The 3rd International Workshop on Natural Language-based Software Engineering (NLBSE'24)}},
  year={{2024}}
}}

@article{{kallis2020tickettagger,
  author={{Kallis, Rafael and Di Sorbo, Andrea and Canfora, Gerardo and Panichella, Sebastiano}},
  title={{Predicting issue types on GitHub}},
  journal={{Science of Computer Programming}},
  volume={{205}},
  pages={{102598}},
  year={{2021}},
  doi={{10.1016/j.scico.2020.102598}}
}}
```

## 3. Licensing and Attribution Notice

> [!WARNING]
> **Redistribution Uncertainty**: The upstream official repository `nlbse2024/issue-report-classification` currently provides an empty `LICENSE` file (0 bytes). While the underlying issue reports originate from public open-source projects on GitHub, the redistribution permissions of the compiled CSVs are not explicitly licensed. Therefore:
> - The raw dataset is strictly stored under `data/external/nlbse2024/` and excluded from Git commits via `.gitignore`.
> - Code, evaluation results, and preprocessing pipelines are tracked.

## 4. Dataset Fields

The raw CSV files contain the following five fields:

| Field | Type | Description |
|---|---|---|
| `repo` | `string` | The GitHub repository identifier (`owner/repo`) |
| `created_at` | `string` | Timestamp when the issue was created on GitHub |
| `label` | `string` | Canonical ground-truth classification label |
| `title` | `string` | Issue report title |
| `body` | `string` | Full Markdown text of the issue report body |

## 5. Classes and Repository Distribution

The dataset covers **5 diverse open-source repositories** and **3 mutually exclusive classes**:
- `bug`
- `feature`
- `question`

Both training and test sets are **perfectly balanced**, containing exactly **100 records per repository per class**:

| Repository | Split | Bug | Feature | Question | Total |
|---|---|---|---|---|---|
| `facebook/react` | Train / Test | 100 / 100 | 100 / 100 | 100 / 100 | 300 / 300 |
| `tensorflow/tensorflow` | Train / Test | 100 / 100 | 100 / 100 | 100 / 100 | 300 / 300 |
| `microsoft/vscode` | Train / Test | 100 / 100 | 100 / 100 | 100 / 100 | 300 / 300 |
| `bitcoin/bitcoin` | Train / Test | 100 / 100 | 100 / 100 | 100 / 100 | 300 / 300 |
| `opencv/opencv` | Train / Test | 100 / 100 | 100 / 100 | 100 / 100 | 300 / 300 |
| **Total** | **All** | **500 / 500** | **500 / 500** | **500 / 500** | **1500 / 1500** |

## 6. Text Length Distributions

### Training Split (1,500 records)
- **Title Words**: min={train_profile['text_statistics']['title_words']['min']}, max={train_profile['text_statistics']['title_words']['max']}, mean={train_profile['text_statistics']['title_words']['mean']}, median={train_profile['text_statistics']['title_words']['median']}
- **Body Words**: min={train_profile['text_statistics']['body_words']['min']}, max={train_profile['text_statistics']['body_words']['max']}, mean={train_profile['text_statistics']['body_words']['mean']}, median={train_profile['text_statistics']['body_words']['median']}
- **Combined Text Words**: min={train_profile['text_statistics']['combined_words']['min']}, max={train_profile['text_statistics']['combined_words']['max']}, mean={train_profile['text_statistics']['combined_words']['mean']}, median={train_profile['text_statistics']['combined_words']['median']}
- **Combined Characters**: min={train_profile['text_statistics']['combined_characters']['min']}, max={train_profile['text_statistics']['combined_characters']['max']}, mean={train_profile['text_statistics']['combined_characters']['mean']}, median={train_profile['text_statistics']['combined_characters']['median']}

### Test Split (1,500 records)
- **Title Words**: min={test_profile['text_statistics']['title_words']['min']}, max={test_profile['text_statistics']['title_words']['max']}, mean={test_profile['text_statistics']['title_words']['mean']}, median={test_profile['text_statistics']['title_words']['median']}
- **Body Words**: min={test_profile['text_statistics']['body_words']['min']}, max={test_profile['text_statistics']['body_words']['max']}, mean={test_profile['text_statistics']['body_words']['mean']}, median={test_profile['text_statistics']['body_words']['median']}
- **Combined Text Words**: min={test_profile['text_statistics']['combined_words']['min']}, max={test_profile['text_statistics']['combined_words']['max']}, mean={test_profile['text_statistics']['combined_words']['mean']}, median={test_profile['text_statistics']['combined_words']['median']}
- **Combined Characters**: min={test_profile['text_statistics']['combined_characters']['min']}, max={test_profile['text_statistics']['combined_characters']['max']}, mean={test_profile['text_statistics']['combined_characters']['mean']}, median={test_profile['text_statistics']['combined_characters']['median']}

## 7. Data Quality & Overlap Checks

| Check | Result |
|---|---|
| Train records with missing title | {train_profile['data_quality']['missing_titles']} |
| Train records with empty title | {train_profile['data_quality']['empty_titles']} |
| Train records with empty body | {train_profile['data_quality']['empty_bodies']} |
| Test records with missing title | {test_profile['data_quality']['missing_titles']} |
| Test records with empty title | {test_profile['data_quality']['empty_titles']} |
| Test records with empty body | {test_profile['data_quality']['empty_bodies']} |
| Exact train/test duplicates | {overlap['exact_record_overlap']} |
| Same repository & title across train and test | {overlap['title_repo_overlap']} |

## 8. Dataset Limitations

1. **Synthetic balance**: The 1:1:1 balance does not reflect real-world repository distributions where bugs usually outnumber questions or features.
2. **Extreme text length variance**: Issue bodies range from short one-liners to >100,000 character logs/stack traces. Truncation strategy must be carefully handled for transformer architectures.
3. **Repository-specific conventions**: Each project uses distinct issue templates (e.g., React's checklist vs. TensorFlow's environment dump), creating strong domain style signals.
"""
    output_path.write_text(md, encoding="utf-8")
    logger.info("Generated %s", output_path)


def main() -> None:
    print("=" * 70)
    print("  NLBSE 2024 Dataset Validation and Profiling")
    print("=" * 70)

    train_profile, test_profile, overlap = run_profiling()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    profile_summary = {
        "train": train_profile,
        "test": test_profile,
        "overlap": overlap,
    }
    summary_path = RESULTS_DIR / "dataset_profile.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(profile_summary, f, indent=2)
    logger.info("Saved profile summary to %s", summary_path)

    dataset_md_path = PROJECT_ROOT / "DATASET.md"
    generate_dataset_md(train_profile, test_profile, overlap, dataset_md_path)

    # Print summary to console
    print("\n--- SUMMARY ---")
    print(f"Train records: {train_profile['count']} (100 per repo x class)")
    print(f"Test records : {test_profile['count']} (100 per repo x class)")
    print(f"Exact train/test duplicate overlap: {overlap['exact_record_overlap']}")
    print(f"Title+repo train/test overlap     : {overlap['title_repo_overlap']}")
    print(f"DATASET.md generated at: {dataset_md_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
