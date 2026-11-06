# RepoTriage — Microproject 2: Issue Classification Engine
## Milestone Report

**Project**: RepoTriage — AI-Powered GitHub Issue Intelligence and Resolution Copilot  
**Milestone**: Microproject 2 — Issue Classification Engine  
**Dataset**: Official NLBSE 2024 Issue Report Classification Benchmark  
**Date**: November 2026  
**Status**: Completed & Verified  

---

## 1. Executive Summary

This milestone establishes the **Issue Classification Engine** for RepoTriage. Using the official **NLBSE 2024 Issue Report Classification** dataset, we engineered, benchmarked, and evaluated three distinct classification paradigms alongside a sanity-check baseline to classify GitHub issue reports into `bug`, `feature`, or `question`:

1. **Dummy Baseline**: Majority-class rule establishing the empirical floor.
2. **Model A (Classical ML)**: Pipeline of word- and character-level TF-IDF features with tuned Logistic Regression.
3. **Model B (SetFit Few-Shot)**: Fine-tuned `sentence-transformers/all-mpnet-base-v2` backbone using contrastive learning (`CoSENTLoss`) and a logistic classification head.
4. **Model C (RepoTriage Pooled Classifier)**: Universal cross-repository model validated via Leave-One-Repository-Out (LOGO) cross-validation and deployed via a decoupled production inference service.

### Overall Benchmark Summary (5 Target Repositories)

Per the official NLBSE 2024 benchmark protocol, five independent repository-specific models were trained strictly on in-repository training splits (300 issues each) and evaluated on in-repository test splits (300 issues each). Cross-repository scores represent the unweighted arithmetic mean of the five repository-level Macro F1 scores.

| Model Approach | Method / Backbone | Training Set | Mean Accuracy | Mean Macro F1 | Mean Weighted F1 | External Ref. Macro F1 | Delta vs Published |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | `DummyClassifier` (majority) | In-repo train (300) | 0.3333 | 0.1667 | 0.1667 | — | — |
| **Model A** | TF-IDF + Logistic Regression | In-repo train (300) | 0.7453 | 0.7438 | 0.7438 | — | — |
| **Model B** | **SetFit (`all-mpnet-base-v2`)** | In-repo train (300) | **0.7780** | **0.7751** | **0.7751** | **0.8270** | **-0.0519** |
| **Model C** | **Pooled TF-IDF + LogReg** | All train (1,500) | **0.7607** | **0.7591** | **0.7605** | — | — |

*Model C Leave-One-Repository-Out (LOGO) Cross-Validation Score (Unseen Repositories)*: **0.5941 Macro F1**

---

## 2. Official Dataset Acquisition & Profiling

### 2.1 Provenance and Integrity

The dataset was obtained directly from the official NLBSE 2024 Tool Competition repository ([`nlbse2024/issue-report-classification`](https://github.com/nlbse2024/issue-report-classification)) using [`scripts/download_nlbse2024.py`](file:///scripts/download_nlbse2024.py) and stored under `data/external/nlbse2024/` without alteration:

- **Training Split (`issues_train.csv`)**:
  - Size: 3,680,400 bytes | Records: 1,500
  - SHA-256: `18dc42a30aa33dccadb723ad3baeb164d38bff521496f985ca2791c26b8939f5`
- **Testing Split (`issues_test.csv`)**:
  - Size: 3,449,390 bytes | Records: 1,500
  - SHA-256: `4f7d8619d4e5adbea126e548fd8c214449288f3a93bb3bc130c54cd307af7e85`

### 2.2 Dataset Schema and Distributions

Each split contains the following fields: `repo`, `created_at`, `label`, `title`, and `body`.

- **5 Target Repositories**:
  1. `facebook/react`
  2. `tensorflow/tensorflow`
  3. `microsoft/vscode`
  4. `bitcoin/bitcoin`
  5. `opencv/opencv`
- **Class Balance**: Exactly 100 records per repository per class across all 3 classes (`bug`, `feature`, `question`) in both train and test splits (300 records per repo in train; 300 records per repo in test). Total: 3,000 records.

### 2.3 Data Quality & Boundary Checks

A dedicated profiling pipeline ([`scripts/profile_dataset.py`](file:///scripts/profile_dataset.py)) generated [`DATASET.md`](file:///DATASET.md) and [`results/dataset_profile.json`](file:///results/dataset_profile.json):

1. **Missing Data**: No missing titles in either split. Two records in the test split have empty body text (both in `bitcoin/bitcoin`). Handled by safe coercion to empty string in [`src/classifier/preprocessing.py`](file:///src/classifier/preprocessing.py).
2. **Train/Test Duplicates**: 3 exact duplicate issue bodies exist across the train and test splits; 14 instances of duplicate titles exist.
3. **Extreme Text Variance**: Body text length ranges from 0 to 202,447 characters (mean: ~2,353 characters). In Python on Windows, this exceeded default CSV field limits, resolved by configuring `csv.field_size_limit(2147483647)`.
4. **Licensing**: The upstream repository contains an empty (0-byte) `LICENSE` file. To respect redistribution uncertainty, raw datasets and large model weights are strictly gitignored via [`.gitignore`](file:///.gitignore).

---

## 3. Official 5-Repository Benchmark Protocol

To ensure strict comparability with published literature:
- **Strict Repository Isolation**: For the benchmark track, each repository classifier was trained **only** on that repository's 300 training records and tested **only** on that repository's 300 test records. No cross-repository data or metadata was visible during training.
- **Leakage Prevention**: All vocabulary extraction (TF-IDF vectorizers) was fitted exclusively on training records. Hyperparameter tuning (regularization $C$, feature representations) used a stratified 20% validation split derived solely from each repository's training set.

---

## 4. Empirical Evaluation Results

### 4.1 Per-Repository Performance Breakdown

All models were evaluated on the held-out test splits (300 issues per repository):

| Repository | Baseline (Dummy) F1 | Classical (Model A) Macro F1 | Classical Best Configuration | SetFit (Model B) Macro F1 | SetFit Accuracy | SetFit vs Classical ($\Delta$ F1) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `facebook/react` | 0.1667 | 0.7918 | word, $C=5.0$ | **0.8043** | 0.8100 | +0.0125 |
| `tensorflow/tensorflow` | 0.1667 | 0.7746 | word, $C=10.0$ | **0.7853** | 0.7867 | +0.0107 |
| `microsoft/vscode` | 0.1667 | 0.6895 | word+char, $C=0.1$ | **0.7742** | 0.7767 | **+0.0847** |
| `bitcoin/bitcoin` | 0.1667 | 0.7163 | word, $C=5.0$ | **0.7397** | 0.7433 | +0.0234 |
| `opencv/opencv` | 0.1667 | 0.7467 | word+char, $C=2.0$ | **0.7722** | 0.7733 | +0.0255 |
| **Arithmetic Mean** | **0.1667** | **0.7438** | — | **0.7751** | **0.7780** | **+0.0313** |

### 4.2 Cross-Repository Per-Class Averages

| Class | Model A Precision | Model A Recall | Model A F1 | Model B Precision | Model B Recall | Model B F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `bug` | 0.7805 | 0.7420 | 0.7575 | 0.7996 | 0.8000 | **0.7991** |
| `feature` | 0.7643 | 0.7800 | 0.7693 | 0.7815 | 0.8700 | **0.8226** |
| `question` | 0.6989 | 0.7140 | 0.7044 | 0.7524 | 0.6640 | **0.7037** |

### 4.3 Key Findings

1. **Contextual Representations vs N-Grams**: SetFit outperformed TF-IDF + Logistic Regression across every single repository. The advantage was most pronounced on `microsoft/vscode` (+8.47% F1 gain), where issue reports frequently contain ambiguous syntax errors and configuration snippets that benefit heavily from pre-trained transformer embeddings.
2. **Class-Level Difficulty**:
   - `feature` was the easiest category for all models (Model B F1 = 0.8226), driven by explicit imperative language (*"add support for..."*, *"allow user to..."*).
   - `question` was consistently the most difficult class across both classical (0.7044) and neural models (0.7037), as developers frequently phrase questions within bug reports (*"Is this expected behavior?"*) or feature requests (*"Can we do X?"*).

---

## 5. Model B: SetFit Reproduction & Hardware Adaptation

The official NLBSE 2024 competition cites a published SetFit baseline Macro F1 of **~0.8270**.

- **Published SetFit Benchmark Reference**: **0.8270 Macro F1**
- **Independently Reproduced SetFit**: **0.7751 Macro F1**
- **Observed Difference ($\Delta$)**: **-0.0519**

### Documented Implementation Deviations & Rationale:

1. **Contrastive Pair Sampling (`num_iterations`)**:
   - *Upstream Notebook*: Used `num_iterations = 20` ($300 \times 20 \times 2 = 12,000$ pairs per repository). On high-end datacenter GPUs (A100/V100 with 40–80GB VRAM), 750 training steps execute in minutes.
   - *Local GPU Hardware*: NVIDIA GeForce RTX 3050 Laptop GPU (6GB VRAM, mobile power limit ~50W). In our initial trial, 12,000 pairs caused self-attention out-of-memory errors on 200 KB texts and required ~2.5 hours per repository (~12.5 hours total).
   - *Applied Adaptation*: We set `num_iterations = 1` (600 contrastive pairs per repository) with `batch_size = (16, 2)`, reducing training time to ~7.5 minutes per repository (~38 minutes total) with zero OOM errors.
2. **Sequence Length Bounding**:
   - Capped `max_seq_length = 128` tokens and truncated raw input strings to 1,500 characters, filtering extreme stack trace noise while preserving issue titles and problem descriptions.
   - Enabled Automatic Mixed Precision (FP16 `use_amp = True`).
3. **Conclusion**: Despite generating only 5% of the contrastive pairs ($1/20$th), our fine-tuned SetFit models attained 0.7751 Macro F1 (within 5.2% of the published baseline), validating the efficiency of contrastive sentence-transformer adaptation.

---

## 6. Model C: RepoTriage Pooled Classifier & Generalization Analysis

A production triage copilot must categorize issues from arbitrary, newly onboarded repositories rather than relying on separate pre-trained models.

### 6.1 Leave-One-Repository-Out (LOGO) Cross-Validation (Training Split Only)

To measure genuine out-of-domain cross-repository generalization, we conducted Leave-One-Group-Out CV strictly on the 1,500 training records. In each fold, an entire repository (300 issues) was held out while training on the remaining 4 repositories (1,200 issues):

| Held-out Validation Repository | F1 Score |
| :--- | :---: |
| `facebook/react` | 0.5673 |
| `tensorflow/tensorflow` | 0.6348 |
| `microsoft/vscode` | 0.5997 |
| `bitcoin/bitcoin` | 0.5769 |
| `opencv/opencv` | 0.5916 |
| **Mean LOGO Cross-Validation Score** | **0.5941 Macro F1** |

*Selected Best Configuration*: `feature_mode = 'word_char'` with $C = 2.0$.

### 6.2 Official Test Split Evaluation

After hyperparameter selection via LOGO CV, the pooled classifier was retrained on the complete 1,500 training records and evaluated once on the 1,500 test records:

- **Overall Accuracy**: **0.7607** (1,141 / 1,500 correct)
- **Overall Macro F1**: **0.7605**
- **Overall Weighted F1**: **0.7605**
- **Arithmetic Mean of Repo Macro F1s**: **0.7591**

#### Test Performance Breakdown by Repository:
- `facebook/react`: Macro F1 = 0.7938 (Accuracy: 0.7967)
- `tensorflow/tensorflow`: Macro F1 = 0.7648 (Accuracy: 0.7667)
- `microsoft/vscode`: Macro F1 = 0.7391 (Accuracy: 0.7400)
- `bitcoin/bitcoin`: Macro F1 = 0.7249 (Accuracy: 0.7267)
- `opencv/opencv`: Macro F1 = 0.7731 (Accuracy: 0.7733)

### 6.3 Understanding Generalization: Three Key Performance Regimes

The empirical data highlights three fundamental performance regimes:

```
┌────────────────────────────────────────────────────────────────────────┐
│  1. Repository-Specific Benchmark (Train on Repo A, Test on Repo A)    │
│     - Mean Macro F1: 0.7751 (SetFit) / 0.7438 (Classical)              │
│     - In-domain distribution; captures project-specific vocabulary     │
├────────────────────────────────────────────────────────────────────────┤
│  2. Pooled Test Performance (Train on 5 Repos, Test on Same 5 Repos)   │
│     - Mean Macro F1: 0.7591 (Classical Pooled)                         │
│     - Multi-domain representation; benefits from 5x data volume        │
├────────────────────────────────────────────────────────────────────────┤
│  3. Leave-One-Repository-Out Generalization (Train on 4, Test Unseen) │
│     - Mean Macro F1: 0.5941 (LOGO CV on Train)                         │
│     - Zero-shot domain transfer; realistic for newly connected repos    │
└────────────────────────────────────────────────────────────────────────┘
```

1. **Why Pooled Test Performance (0.7591) Outperforms LOGO CV (0.5941)**:  
   The official test split contains issues from the same five repositories seen during pooled training. Project-specific terminology (`hooks`, `tensor`, `bitcoind`, `Mat`) is known to the model.
2. **Why LOGO CV (0.5941) Reflects True Cold-Start Performance**:  
   When RepoTriage encounters an unseen repository, project-specific tokens provide no signal. The model must rely strictly on domain-agnostic vocabulary (*"error: expected X but got Y"*, *"traceback"*, *"feature request"*). The 0.5941 score provides an honest estimate of cold-start performance. Testing on the official test split does **not** prove generalizability to unseen codebases.
3. **Engineering Takeaway**:  
   RepoTriage should deploy the Pooled Classifier for initial onboarding, then offer an active-learning or few-shot adaptation loop as issues are triaged in that repository.

---

## 7. Production Integration Interface

The classification engine is exposed through a decoupled inference service ([`src/classifier/inference.py`](file:///src/classifier/inference.py)) returning standardized dataclass results:

### Python API Usage

```python
from src.classifier.inference import IssueClassifierService

# Instantiate service from saved model artifact
service = IssueClassifierService.from_saved_model(
    "models/pooled_classifier.joblib", 
    model_type="classical"
)

# Classify incoming issue
result = service.classify_issue(
    title="Unexpected 422 error on query parameter validation",
    body="FastAPI returns 422 Unprocessable Entity when passing an integer string in query param."
)

print(result.category)          # "bug"
print(result.scores)            # {"bug": 0.4850, "feature": 0.1206, "question": 0.3943}
print(result.model_identifier)  # "RepoTriage-Pooled-TFIDF-LogReg"
```

### Command-Line Interface (CLI)

RepoTriage includes a command-line interface ([`src/classifier/__main__.py`](file:///src/classifier/__main__.py)):

```bash
# 1. Single issue classification:
python -m src.classifier --title "NullPointer in auth" --body "Traceback at line 42..."

# 2. Batch classification from an ingested JSON file:
python -m src.classifier --file data/raw/fastapi_fastapi_issues_20261106T082931Z.json --limit 10
```

---

## 8. Verification & Test Suite

All **39 automated unit tests** execute cleanly with zero failures:

```bash
python -m unittest discover -s tests -v
```

```
test_dummy_baseline (test_classifier.TestClassifier.test_dummy_baseline) ... ok
test_leakage_prevention_vocab_fitted_on_train_only ... ok
test_model_serialization_and_inference_service ... ok
test_pipeline_word_and_char_features ... ok
test_pipeline_word_features ... ok
test_repository_separation ... ok
test_compute_cross_repo_score (test_evaluation.TestEvaluation) ... ok
test_compute_metrics_known_imperfect ... ok
test_compute_metrics_perfect_predictions ... ok
test_http_403_retries_then_fails (test_ingest.TestErrorHandling) ... ok
test_http_404_raises_immediately ... ok
test_malformed_json_retries ... ok
test_network_error_retries ... ok
test_non_list_json_retries ... ok
test_empty_page_stops_pagination (test_ingest.TestFetchIssues) ... ok
test_max_pages_limit_respected ... ok
test_pagination_follows_next_link ... ok
test_prs_are_excluded ... ok
test_single_page_no_prs ... ok
test_basic_normalization (test_ingest.TestNormalizeIssue) ... ok
test_label_with_non_dict_skipped ... ok
test_missing_labels ... ok
test_null_body_becomes_empty_string ... ok
test_empty_header (test_ingest.TestParseLinkNext) ... ok
test_no_next ... ok
test_none_header ... ok
test_valid_next_link ... ok
test_empty_issues (test_ingest.TestPrintStats) ... ok
test_stats_with_data ... ok
test_save_creates_file (test_ingest.TestSaveDataset) ... ok
test_clean_markdown_and_noise (test_preprocessing.TestPreprocessing) ... ok
test_combine_issue_text ... ok
test_combine_issue_text_empty_cases ... ok
test_label_encoding_and_decoding ... ok
test_unknown_id_raises ... ok
test_unknown_label_raises ... ok
test_validate_issue_input_non_string_types ... ok
test_validate_issue_input_none_and_empty ... ok
test_validate_issue_input_normal ... ok
----------------------------------------------------------------------
Ran 39 tests in 0.139s
OK
```

---

## 9. Deliverables and File Inventory

```
RepoTriage/
├── data/
│   ├── raw/                                # Ingested GitHub JSON datasets (gitignored)
│   ├── external/nlbse2024/                 # Official benchmark CSVs (gitignored)
│   └── processed/                          # Intermediate data (gitignored)
├── models/                                 # Serialized model weights (gitignored)
│   ├── pooled_classifier.joblib            # Production pooled model
│   ├── classical_*.joblib                  # 5 repository-specific classical models
│   └── setfit_*/                           # 5 repository-specific fine-tuned SetFit models
├── results/
│   ├── figures/                            # 11 Confusion matrix PNG plots
│   ├── benchmark_results.json              # Complete official benchmark metrics
│   ├── benchmark_results.csv               # Tabular benchmark metrics
│   ├── pooled_results.json                 # LOGO CV and test evaluation metrics
│   ├── dataset_profile.json                # Distribution and data quality profile
│   └── evaluation_report.md                # Comprehensive technical evaluation report
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
│       ├── config.py                       # Configuration, constants & hyperparameters
│       ├── preprocessing.py                # Input validation and text concatenation
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
├── readME2.md                              # Complete Milestone 2 Report (this document)
├── README.md                               # Project guide
└── .gitignore                              # Git exclusion rules
```

---

## 10. Reproduction Commands

To replicate all results from scratch:

```bash
# 1. Download official benchmark dataset
python scripts/download_nlbse2024.py

# 2. Profile dataset distributions and data quality
python scripts/profile_dataset.py

# 3. Train and evaluate the 5-repository benchmark (Dummy, Classical, SetFit)
python scripts/train_benchmark.py --setfit-iterations 1

# 4. Run LOGO CV and train the production pooled model
python scripts/train_pooled.py

# 5. Run the full unit test suite
python -m unittest discover -s tests -v

# 6. Test inference CLI
python -m src.classifier --title "Crash on startup" --body "NullPointerException at Auth:42"
```

---

## 11. Recommended Next Engineering Step

With the **GitHub Issue Ingestion** (Microproject 1) and **Issue Classification Engine** (Microproject 2) fully operational and tested, the next phase is **Microproject 3: REST API & Copilot Service Layer**:

1. **FastAPI Service**: Wrap `IssueClassifierService` into a production-grade asynchronous REST API with OpenAPI/Swagger documentation.
2. **Pydantic Schemas**: Define strict request models (`title: str`, `body: Optional[str]`, `repository: Optional[str]`) and response contracts (`category: IssueCategory`, `confidence: float`, `scores: dict[str, float]`).
3. **Batch Triage Endpoint**: Provide endpoints for bulk issue triage from GitHub webhook events or ingestion jobs.
4. **Confidence Thresholding**: Introduce an `uncertain` fallback flag when maximum class probability falls below a tunable threshold (e.g., $P < 0.45$).
