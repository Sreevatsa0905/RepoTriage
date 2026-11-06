# RepoTriage — GitHub Issue Intelligence & Resolution Copilot

RepoTriage is an AI-powered GitHub Issue Intelligence and Resolution Copilot designed to automate issue ingestion, triage, categorization, and resolution assistance.

---

## Architecture Overview

- **Microproject 1 — GitHub Issue Ingestion**: Configurable CLI tool fetching and normalizing issues from public GitHub repositories using the standard library only.
- **Microproject 2 — Issue Classification Engine**: Multi-class categorization (`bug`, `feature`, `question`) evaluated on the official NLBSE 2024 benchmark (3,000 issues across 5 repositories) using Classical ML (TF-IDF + Logistic Regression), few-shot fine-tuning (SetFit with `all-mpnet-base-v2`), and a production pooled classifier.

---

## Quick Start

### Environment Setup

```bash
# Activate Python 3.12 virtual environment
# Windows
.\.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

---

## Microproject 1: GitHub Issue Ingestion

Fetch issues from any public GitHub repository into normalized JSON:

```bash
# Ingest issues from fastapi/fastapi
python -m src.ingest --owner fastapi --repo fastapi --per-page 30 --max-pages 2

# Ingest open issues with debug logging
python -m src.ingest --owner pallets --repo flask --state open --per-page 50 --max-pages 1 -v
```

Output files are stored under `data/raw/{owner}_{repo}_issues_{timestamp}.json`.

---

## Microproject 2: Issue Classification Engine

### 1. Dataset Acquisition & Profiling

Acquires the official NLBSE 2024 benchmark splits (`issues_train.csv` and `issues_test.csv`) and generates profiling statistics:

```bash
# Download official datasets
python scripts/download_nlbse2024.py

# Profile dataset distributions, quality, and leakage checks
python scripts/profile_dataset.py
```

Outputs:
- `data/external/nlbse2024/issues_train.csv` (1,500 records)
- `data/external/nlbse2024/issues_test.csv` (1,500 records)
- `DATASET.md` and `results/dataset_profile.json`

### 2. Official 5-Repository Benchmark

Evaluates three independent approaches following the official NLBSE 2024 protocol (5 independent models per approach, trained strictly on in-repository train sets, evaluated on in-repository test sets):

1. **DummyClassifier**: Majority-class sanity baseline
2. **Model A (Classical)**: TF-IDF + Logistic Regression (C-regularization and word vs word+char feature tuning on train validation fold)
3. **Model B (SetFit)**: Few-shot sentence transformer contrastive fine-tuning (`sentence-transformers/all-mpnet-base-v2`)

```bash
# Run complete official benchmark
python scripts/train_benchmark.py

# Or run SetFit with custom contrastive iterations
python scripts/train_benchmark.py --setfit-iterations 1
```

Outputs:
- `results/benchmark_results.json`
- `results/benchmark_results.csv`
- `results/figures/cm_classical_*.png`
- `results/figures/cm_setfit_*.png`

### 3. RepoTriage Pooled Classifier (Model C)

Trains a universal classifier across all 5 repositories with Leave-One-Repository-Out (LOGO) cross-validation for real-world unseen repository triage:

```bash
python scripts/train_pooled.py
```

Outputs:
- `models/pooled_classifier.joblib`
- `results/pooled_results.json`
- `results/figures/cm_pooled_overall_test.png`

---

## Benchmark Results Summary

| Model Approach | Method / Backbone | Training Data | Mean Accuracy | Mean Macro F1 | Reference F1 | Delta vs Published |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Baseline** | DummyClassifier | Repo Train (300) | 0.3333 | 0.1667 | — | — |
| **Model A** | TF-IDF + Logistic Regression | Repo Train (300) | 0.7453 | 0.7438 | — | — |
| **Model B** | SetFit (`all-mpnet-base-v2`) | Repo Train (300) | **0.7780** | **0.7751** | 0.8270 | -0.0519 |
| **Model C** | Pooled TF-IDF + LogReg | All Train (1500) | 0.7607 | 0.7591 | — | — |

*LOGO Cross-Validation Score (Cold-Start Unseen Repositories)*: **0.5941 Macro F1**

See [`results/evaluation_report.md`](file:///results/evaluation_report.md) for full per-class metrics and analysis.

---

## Production Inference Interface

RepoTriage provides a decoupled inference service designed for future FastAPI integration:

```python
from src.classifier.inference import IssueClassifierService

# Load production model
service = IssueClassifierService.from_saved_model(
    "models/pooled_classifier.joblib", 
    model_type="classical"
)

# Classify incoming issue
result = service.classify_issue(
    title="NullPointerException in authentication filter",
    body="Traceback: java.lang.NullPointerException at com.repo.Auth.verify..."
)

print(result.category)          # "bug"
print(result.scores)            # {"bug": 0.6279, "feature": 0.1660, "question": 0.2061}
print(result.model_identifier)  # "RepoTriage-Pooled-TFIDF-LogReg"
```

---

## Running Automated Tests

Run the full unit test suite (39 tests covering ingestion, preprocessing, leakage prevention, serialization, and inference):

```bash
python -m unittest discover -s tests -v
```

---

## Project Structure

```
RepoTriage/
├── data/
│   ├── raw/                       # Ingested GitHub JSON datasets (gitignored)
│   ├── external/nlbse2024/        # Official NLBSE 2024 benchmark CSVs (gitignored)
│   └── processed/                 # Intermediate processed data (gitignored)
├── models/                        # Serialized model weights & artifacts (gitignored)
├── results/
│   ├── figures/                   # Confusion matrix plots
│   ├── benchmark_results.json     # Official benchmark metrics
│   ├── benchmark_results.csv      # Tabular benchmark metrics
│   ├── pooled_results.json        # LOGO CV & pooled evaluation metrics
│   ├── dataset_profile.json       # Dataset distribution profile
│   └── evaluation_report.md       # Comprehensive evaluation report
├── scripts/
│   ├── download_nlbse2024.py      # Official dataset downloader
│   ├── profile_dataset.py         # Profiling & DATASET.md generator
│   ├── train_benchmark.py         # 5-repo official benchmark runner
│   └── train_pooled.py            # LOGO CV & pooled model trainer
├── src/
│   ├── __init__.py
│   ├── __main__.py                # Ingestion CLI entry point
│   ├── ingest.py                  # Issue fetcher and normalizer
│   └── classifier/
│       ├── __init__.py
│       ├── config.py              # Experiment configurations & hyperparameters
│       ├── preprocessing.py       # Validation and text combination
│       ├── classical.py           # TF-IDF + Logistic Regression pipeline
│       ├── setfit_model.py        # SetFit few-shot sentence transformer wrapper
│       ├── train.py               # Repository-specific and pooled training
│       ├── evaluate.py            # Metrics, confusion matrices, exporters
│       └── inference.py           # Decoupled production inference service
├── tests/
│   ├── test_ingest.py             # 21 tests for ingestion & error handling
│   ├── test_preprocessing.py      # 9 tests for preprocessing & input validation
│   ├── test_classifier.py         # 6 tests for pipelines, leakage & serialization
│   └── test_evaluation.py         # 3 tests for metrics & cross-repo calculation
├── DATASET.md                     # Dataset documentation, citation & licensing
├── .gitignore
└── README.md
```
