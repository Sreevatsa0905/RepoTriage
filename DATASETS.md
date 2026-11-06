# RepoTriage Datasets & Corpus Documentation

This document describes the corpora, external datasets, and processed retrieval indexes powering RepoTriage.

---

## 1. Stage 1 Seed Repositories Corpus (Retrieval & Grounding)

Acquired via `src.ingest` and `src.cli ingest` as specified in Stage 1 of `RepoTriage_SPEC.md`.

### Summary Counts Table

| Repository | Issues Ingested | Comments Ingested | Doc Files Ingested | Issue Chunks | Comment Chunks | Doc Chunks | Total Chunks |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`pallets/flask`** | 400 | 735 | 77 | 466 | 742 | 437 | **1,645** |
| **`encode/httpx`** | 400 | 1,047 | 24 | 556 | 1,346 | 252 | **2,154** |
| **`pydantic/pydantic`** | 400 | 868 | 93 | 531 | 910 | 859 | **2,300** |
| **TOTAL** | **1,200** | **2,650** | **194** | **1,553** | **2,998** | **1,548** | **6,099** |

*All counts verified directly from `data/processed/repotriage.db`.*

---

## 2. Ingestion & Preprocessing Details

### A. Issue Acquisition (`src/ingest/github_client.py`)
- **API Endpoint**: `https://api.github.com/repos/{owner}/{repo}/issues`
- **Filtering**: Strict exclusion of Pull Requests (`pull_request` key omitted from all records).
- **Strategy**: Target count of 400 issues per repository, prioritizing closed issues first to capture resolved discussions, debugging threads, and resolution comments.
- **Comments**: Fetched for closed issues with $\ge 1$ comment (capped at 10 comments per issue).
- **On-Demand Single Fetch**: `fetch_single_issue(owner, repo, issue_number)` checks raw JSONL cache first, and retrieves referenced canonical duplicate issues on demand if outside the 400-issue window.
- **Raw Cache**: Cached atomically to `data/raw/{owner}_{repo}_raw_issues.jsonl` and `data/raw/{owner}_{repo}_raw_comments.jsonl`.

### B. Documentation Loading (`src/ingest/docs_loader.py`)
- **Cloning**: Shallow git clone (`git clone --depth 1`) into `data/raw/repos/{owner}_{repo}`. Skipped if existing clone is present.
- **Patterns**: Scanned `README*`, `CONTRIBUTING*`, `docs/**/*.{md,mdx,rst}`, and `doc/**/*.{md,rst}`.
- **Format Support**:
  - Markdown (`.md`, `.mdx`): parsed via `#`, `##`, `###` headings with GitHub-compatible slug anchor URLs.
  - reStructuredText (`.rst`): parsed via underline headings (`===`, `---`, `~~~`, `^^^`), preserving section structure (essential for `pallets/flask`).

### C. Chunking (`src/ingest/chunker.py`)
- **Window Size**: 250–350 tokens per chunk with ~40 token sliding overlap.
- **Context Retention**:
  - Issue chunks attach `[{repo} #{number}] {title}` to every chunk window.
  - Comment chunks attach `[{repo} #{number} Comment by @{user}]` to every chunk window.
  - Doc chunks attach `[{repo} - {file_path}] {heading}` and preserve anchor URLs (`#{slug}`).
- **Schema**:
  - `chunk_id` (16-char SHA256 hex)
  - `repo` (`owner/repo`)
  - `source_type` (`issue | comment | doc`)
  - `issue_number` (integer or null)
  - `file_path` (string or null)
  - `heading` (string)
  - `url` (canonical web link with anchor where available)
  - `content_hash` (SHA256 hex)
  - `text` (chunk text)
  - `token_count` (estimated token count)

### D. Storage & Indexing (`src/ingest/storage.py`)
- **SQLite Database**: `data/processed/repotriage.db`
  - Tables: `issues`, `comments`, `doc_files`, `chunks`.
  - Indexes: `idx_issues_repo`, `idx_issues_number`, `idx_comments_issue`, `idx_doc_files_repo`, `idx_chunks_repo`, `idx_chunks_type`, `idx_chunks_hash`.
- **JSONL Exports**: `data/processed/{owner}_{repo}_chunks.jsonl`.

---

## 3. External Classification Benchmark (NLBSE 2024)

Stored in `data/external/nlbse2024/` (detailed in [`DATASET.md`](file:///DATASET.md)):
- **Training split**: 1,500 records (`issues_train.csv`)
- **Testing split**: 1,500 records (`issues_test.csv`)
- **Repositories**: `facebook/react`, `tensorflow/tensorflow`, `microsoft/vscode`, `bitcoin/bitcoin`, `opencv/opencv`
- **Classes**: `bug`, `feature`, `question`
