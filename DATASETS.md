# RepoTriage Corpus & Ingestion Datasets Specification

This document details the corpus composition, ingestion methodology, chunking parameters, and source attribution for RepoTriage.

---

## 1. Multi-Repository Production Corpus

The live corpus is persisted in SQLite (`data/processed/repotriage.db`) and processed JSON Lines (`data/processed/`). All seed repositories were acquired via GitHub REST API pagination (prioritizing closed issues with comments) and shallow git clones (`git clone --depth 1`) for documentation.

- **Ingestion Date**: November 6–7, 2026
- **Corpus Limit**: Target ~400 genuine issues per repo + on-demand canonical duplicate resolution
- **Chunk Parameters**: 250–350 tokens, ~40 token overlap, structure-aware markdown (`#`, `##`, `###`) and reStructuredText (`===`, `---`, `~~~`, `^^^`) heading retention.

### Real Database Ingestion Counts

| Repository | Issues | Comments | Doc Files | Issue Chunks | Comment Chunks | Doc Chunks | Total Chunks |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`encode/httpx`** | 405 | 1,047 | 24 | 561 | 1,346 | 252 | **2,159** |
| **`pallets/flask`** | 404 | 735 | 77 | 476 | 742 | 437 | **1,655** |
| **`psf/requests`** | 408 | 892 | 16 | 538 | 912 | 136 | **1,586** |
| **`pydantic/pydantic`** | 403 | 868 | 93 | 535 | 910 | 859 | **2,304** |
| **Total** | **1,620** | **3,542** | **210** | **2,110** | **3,910** | **1,684** | **7,704** |

---

## 2. Ingestion & Preprocessing Details

### GitHub Issue & Comment Acquisition
- **API Endpoint**: `/repos/{owner}/{repo}/issues?state=all&per_page=100`
- **Filtering**: Strict exclusion of Pull Requests via `pull_request` key check.
- **Comment Cap**: Capped at ~10 comments per closed issue to focus on consensus resolution and root causes while mitigating spam.
- **On-Demand Canonical Fetching**: When duplicate issues reference canonical issues `#N` outside the initial 400-issue window, `fetch_single_issue(owner, repo, issue_number)` pulls the target issue and its discussion into the corpus and index.

### Documentation Loader
- **Seed Repositories**: Cloned into `data/raw/repos/` via `git clone --depth 1`.
- **File Matching**: `README*`, `CONTRIBUTING*`, and `docs/**/*.{md,mdx,rst}`.
- **Format Handling**: Custom parsers extract hierarchical document sections across Markdown and reStructuredText formats without losing anchor metadata.

---

## 3. NLBSE 2024 Issue Classification Benchmark

The frozen issue classifier was pre-trained and evaluated on the NLBSE 2024 Issue Classification Challenge dataset.

### Dataset Overview
- **Paper**: *The NLBSE'24 Tool Competition on Issue Report Classification*
- **Source**: 5 open-source repositories (`facebook/react`, `tensorflow/tensorflow`, `microsoft/vscode`, `bitcoin/bitcoin`, `opencv/opencv`).
- **Target Classes**: `bug`, `feature`, `question`.
- **Training Set**: 1,500 balanced issue reports (300 per repository, 100 per class).
- **Test Set**: 1,500 balanced issue reports (300 per repository, 100 per class).

### Label Mappings & Live Generalization Set
For live evaluation on the ingested production repositories:
- `bug` $\rightarrow$ `bug`
- `enhancement`, `feature`, `feature request` $\rightarrow$ `feature`
- `question` $\rightarrow$ `question`
- Ambiguous labels (`wontfix`, `duplicate`, `help wanted`) are discarded from evaluation splits.
- A threshold filter $\tau = 0.50$ is applied to route low-confidence predictions to `uncertain`.

---

## 4. Licenses & Attribution

All repository source materials, issue discussions, and documentation were obtained from publicly accessible GitHub repositories under their respective open-source licenses:

1. **`encode/httpx`**: BSD 3-Clause License (Copyright Encode OSS Ltd).
2. **`pallets/flask`**: BSD 3-Clause License (Copyright Pallets).
3. **`psf/requests`**: Apache License 2.0 (Copyright Python Software Foundation).
4. **`pydantic/pydantic`**: MIT License (Copyright Samuel Colvin & Pydantic contributors).
5. **NLBSE 2024 Repositories**: React (MIT), TensorFlow (Apache 2.0), VS Code (MIT), Bitcoin (MIT), OpenCV (Apache 2.0).
6. **ANLI Benchmark**: CC BY-NC 4.0 (Facebook AI Research).
