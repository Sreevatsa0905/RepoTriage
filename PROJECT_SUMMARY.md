# RepoTriage: Comprehensive Project Status & Milestone Summary

**Project**: RepoTriage — AI-Powered GitHub Issue Intelligence and Resolution Copilot  
**Status**: Microprojects 1 & 2 Complete | Ready for Microproject 3  
**Python Version**: 3.12  
**Platform**: Windows 11 (CUDA-accelerated PyTorch on NVIDIA GeForce RTX 3050 Laptop GPU)  
**Date**: November 2026  

---

## 1. Executive Summary & Trajectory

RepoTriage is an autonomous GitHub Copilot designed to ingest, classify, analyze, and assist in resolving software issues. To date, two complete engineering milestones have been completed and verified with automated test suites and empirical benchmarks:

```
┌─────────────────────────────────────────────────────────────┐
│                       RepoTriage Core                       │
├──────────────────────────────┬──────────────────────────────┤
│  Microproject 1 (Ingestion)  │  Microproject 2 (Classifier) │
│  - GitHub REST API Fetcher   │  - NLBSE 2024 Benchmark      │
│  - Standard library only     │  - Classical TF-IDF + LogReg │
│  - PR filtering & pagination │  - SetFit (all-mpnet-base-v2)│
│  - Exponential backoff retry │  - Pooled Universal Model    │
│  - Schema normalization      │  - LOGO Cross-Validation     │
│  - JSON dataset persistence  │  - Decoupled Inference API   │
└──────────────────────────────┴──────────────────────────────┘
                               │
                               ▼
               Microproject 3 (Upcoming / Next)
               - FastAPI Async Microservice
               - Batch Triage & Webhook Handlers
               - Confidence Thresholding & Routing
```

---

## 2. Microproject 1: GitHub Issue Ingestion

### What Was Built
A command-line tool implemented entirely in Python 3.12 **using standard library modules only** (`urllib.request`, `json`, `argparse`, `logging`, `unittest`):

- **Configurable Fetcher**: Supports custom `owner`, `repository`, `state` (`open`, `closed`, `all`), `per_page` (1–100), and `max_pages`.
- **RFC 5988 Pagination**: Parses GitHub `Link` response headers to follow pagination links until exhausted or reaching user limits.
- **Pull Request Filtering**: Detects and discards pull requests (which GitHub returns within `/issues` endpoints).
- **Resilient Network Handling**: Implements exponential backoff retries (2s, 4s, 8s) for transient HTTP errors (403, 429 rate limits, connection resets, malformed JSON), with instant abort on non-retryable errors (e.g., 404 Not Found).
- **Schema Normalization**: Maps raw issues to clean records containing: `repository`, `number`, `title`, `body`, `state`, `labels`, `created_at`, `updated_at`, `html_url`, and `comment_count`.
- **Ingestion Profiling**: Computes issue counts, label frequencies, missing body rates, and outputs ASCII distribution charts.
- **Automated Tests**: 21 unit tests covering pagination, error handling, normalization, and PR exclusion using mocked network responses.

### Key Commands
```bash
# Ingest fastapi/fastapi issues
python -m src.ingest --owner fastapi --repo fastapi --per-page 30 --max-pages 2

# Run ingestion unit tests
python -m unittest tests/test_ingest.py -v
```

---

## 3. Microproject 2: Issue Classification Engine

### 3.1 Official Dataset Acquisition & Profiling
- **Dataset**: Official **NLBSE 2024 Issue Report Classification** benchmark (`issues_train.csv` and `issues_test.csv`).
- **Repositories (5)**: `facebook/react`, `tensorflow/tensorflow`, `microsoft/vscode`, `bitcoin/bitcoin`, `opencv/opencv`.
- **Target Classes (3)**: `bug`, `feature`, `question`.
- **Volume & Balance**: Exactly 100 records per repository per class (1,500 training records, 1,500 test records).
- **Profiling Observations** ([`DATASET.md`](file:///DATASET.md)):
  - Body length ranges from 0 to 202,447 characters (handled via `csv.field_size_limit(2147483647)` and text truncation).
  - 2 empty bodies in test split (`bitcoin/bitcoin`).
  - 3 exact duplicate bodies between train and test splits; 14 instances of duplicate titles.
  - Zero-byte upstream `LICENSE` file; raw datasets and large model binaries are excluded from Git via [`.gitignore`](file:///.gitignore).

---

### 3.2 Official 5-Repository Benchmark Results

Per the official competition rules, five independent classifiers were trained strictly on each repository's training records and evaluated on its test records.

#### Overall Benchmark Summary

| Model Approach | Method / Backbone | Training Data | Mean Accuracy | Mean Macro F1 | External Ref. Macro F1 | Delta vs Published |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Baseline** | `DummyClassifier` (majority) | In-repo train (300) | 0.3333 | 0.1667 | — | — |
| **Model A** | TF-IDF + Logistic Regression | In-repo train (300) | 0.7453 | 0.7438 | — | — |
| **Model B** | **SetFit (`all-mpnet-base-v2`)** | In-repo train (300) | **0.7780** | **0.7751** | **0.8270** | **-0.0519** |
| **Model C** | **RepoTriage Pooled Classifier** | All train (1,500) | **0.7607** | **0.7591** | — | — |

#### Per-Repository Performance Breakdown

| Repository | Baseline F1 | Classical (Model A) F1 | Classical Best Config | SetFit (Model B) F1 | SetFit Accuracy | SetFit vs Classical ($\Delta$ F1) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `facebook/react` | 0.1667 | 0.7918 | word, $C=5.0$ | **0.8043** | 0.8100 | +0.0125 |
| `tensorflow/tensorflow` | 0.1667 | 0.7746 | word, $C=10.0$ | **0.7853** | 0.7867 | +0.0107 |
| `microsoft/vscode` | 0.1667 | 0.6895 | word+char, $C=0.1$ | **0.7742** | 0.7767 | **+0.0847** |
| `bitcoin/bitcoin` | 0.1667 | 0.7163 | word, $C=5.0$ | **0.7397** | 0.7433 | +0.0234 |
| `opencv/opencv` | 0.1667 | 0.7467 | word+char, $C=2.0$ | **0.7722** | 0.7733 | +0.0255 |
| **Arithmetic Mean** | **0.1667** | **0.7438** | — | **0.7751** | **0.7780** | **+0.0313** |

---

### 3.3 SetFit Implementation & Hardware Adaptation

The official competition benchmark reported a SetFit baseline Macro F1 of **0.8270**. Our independently reproduced implementation achieved **0.7751 Macro F1** ($\Delta = -0.0519$).

**Documented Hardware Deviations**:
1. **Pair Generation**: The official Colab notebook used `num_iterations = 20` (12,000 contrastive pairs per repo). On local mobile hardware (NVIDIA RTX 3050 6GB Laptop GPU), this required ~2.5 hours per repository (~12.5 hours total). We adapted `num_iterations = 1` (600 pairs per repo), executing in ~7.5 minutes per repo (~38 minutes total) with zero CUDA OOMs.
2. **Sequence Bounding**: Capped `max_seq_length = 128` and input body text to 1,500 characters, using Automatic Mixed Precision (FP16 `use_amp = True`).

---

### 3.4 Model C: Pooled Classifier & Generalization Regimes

For production triage of arbitrary GitHub repositories, we trained a universal pooled model across all 1,500 training issues.

- **Leave-One-Repository-Out (LOGO) Cross-Validation**:
  Holding out an entire repository during training yielded a mean validation score of **0.5941 Macro F1** (`facebook/react`: 0.5673, `tensorflow/tensorflow`: 0.6348, `microsoft/vscode`: 0.5997, `bitcoin/bitcoin`: 0.5769, `opencv/opencv`: 0.5916).
- **Official Test Split Evaluation**:
  Retrained on the full 1,500 training split and evaluated once on the 1,500 official test issues:
  - **Overall Accuracy**: **0.7607** (1,141 / 1,500 correct)
  - **Overall Macro F1**: **0.7605**
  - **Mean of Repository Macro F1s**: **0.7591**

#### Crucial Theoretical Distinction:
1. **Repository-Specific Benchmark (0.7751 / 0.7438)**: Evaluates specialized models on in-domain vocabulary.
2. **Pooled Test Performance (0.7591)**: Evaluates on the *same 5 repositories* seen in pooled training; benefits from shared multi-domain vocabulary.
3. **Leave-One-Repository-Out Generalization (0.5941)**: Measures **true zero-shot cold-start generalization to unseen repositories**. Testing on the official test set does **not** prove generalizability to new projects.

---

### 3.5 Production Inference Interface & CLI

A decoupled inference service ([`src/classifier/inference.py`](file:///src/classifier/inference.py)) and CLI tool ([`src/classifier/__main__.py`](file:///src/classifier/__main__.py)) allow classifying individual issues or batch-triaging ingested JSON files:

```bash
# Single issue triage
python -m src.classifier --title "Unexpected 422 error on query parameter" --body "FastAPI returns 422..."

# Batch triage from ingested issues
python -m src.classifier --file data/raw/fastapi_fastapi_issues_20261106T082931Z.json --limit 5
```

---

## 4. Test Suite Verification

All **39 unit tests** pass in ~0.14s:
```bash
python -m unittest discover -s tests -v
```
- `tests/test_ingest.py`: 21 tests (pagination, retries, PR exclusion, normalization)
- `tests/test_preprocessing.py`: 9 tests (input validation, empty handling, markdown cleaning)
- `tests/test_classifier.py`: 6 tests (word/char features, dummy baseline, train-only vocabulary fitting / leakage prevention, model serialization & inference service)
- `tests/test_evaluation.py`: 3 tests (precision, recall, F1, cross-repository scoring)

---

## 5. File Inventory & Repository Structure

```
RepoTriage/
├── data/
│   ├── raw/                                # Ingested GitHub JSON datasets (gitignored)
│   ├── external/nlbse2024/                 # Official benchmark CSVs (gitignored)
│   └── processed/                          # Intermediate data (gitignored)
├── models/                                 # Serialized model weights (gitignored)
│   ├── pooled_classifier.joblib            # Production pooled model
│   ├── classical_*.joblib                  # 5 repository-specific classical models
│   └── setfit_*/                           # 5 fine-tuned SetFit models
├── results/
│   ├── figures/                            # 11 Confusion matrix PNG plots
│   ├── benchmark_results.json              # Benchmark metrics (Dummy, Classical, SetFit)
│   ├── benchmark_results.csv               # Tabular benchmark metrics
│   ├── pooled_results.json                 # LOGO CV and test evaluation metrics
│   ├── dataset_profile.json                # Distribution and data quality profile
│   └── evaluation_report.md                # Detailed technical evaluation report
├── scripts/
│   ├── download_nlbse2024.py               # Official benchmark dataset downloader
│   ├── profile_dataset.py                  # Dataset profiler and DATASET.md generator
│   ├── train_benchmark.py                  # 5-repository benchmark runner
│   └── train_pooled.py                     # LOGO CV & pooled retraining runner
├── src/
│   ├── __init__.py
│   ├── __main__.py                         # Ingestion CLI entry point
│   ├── ingest.py                           # GitHub issue ingestion tool
│   └── classifier/
│       ├── __init__.py
│       ├── __main__.py                     # Classifier CLI entry point
│       ├── config.py                       # Configuration & hyperparameters
│       ├── preprocessing.py                # Validation and text concatenation
│       ├── classical.py                    # Classical TF-IDF pipelines & dummy baseline
│       ├── setfit_model.py                 # SetFit wrapper with FP16 and memory safeguards
│       ├── train.py                        # Repo-specific tuning and LOGO CV
│       ├── evaluate.py                     # Scoring, confusion matrices & export
│       └── inference.py                    # Production inference service
├── tests/
│   ├── test_ingest.py                      # 21 unit tests for ingestion
│   ├── test_preprocessing.py               # 9 unit tests for preprocessing
│   ├── test_classifier.py                  # 6 unit tests for classical/inference
│   └── test_evaluation.py                  # 3 unit tests for metrics
├── DATASET.md                              # Dataset documentation and licensing notes
├── readME2.md                              # Milestone 2 Detailed Technical Report
├── PROJECT_SUMMARY.md                      # Complete Project Status & Architecture Overview (this file)
├── README.md                               # Project Quickstart & CLI Guide
└── .gitignore                              # Git exclusion rules
```

---

## 6. Recommended Next Steps (Microproject 3)

1. **FastAPI Service**: Build an asynchronous REST API around `IssueClassifierService`.
2. **Pydantic Schemas**: Define validation contracts for issue payloads and confidence scores.
3. **Batch Triage Endpoints**: Expose endpoints for webhook ingestion and bulk classification.
4. **Confidence Thresholding**: Route low-confidence classifications ($P < 0.45$) to a manual review queue.
