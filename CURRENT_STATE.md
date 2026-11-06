# RepoTriage — Current Repository State (Stage 0)

**Date**: November 2026  
**Phase**: Stage 0: Repository Inspection & Frozen Classifier Wrap  
**Specification**: `RepoTriage_SPEC.md`  

---

## 1. Inspection of Unreviewed Components

Per Section 1b of `RepoTriage_SPEC.md`, the contents of `src/analyze.py`, `src/data/`, and `output/` have been inspected and documented below. None of these files or directories have been deleted or modified.

### 1.1 `src/analyze.py`
- **Purpose**: A pure Python standard-library text analysis, keyword extraction, and heuristic categorization module (525 lines).
- **Core Capabilities**:
  - **Text Preprocessing**: Strips Markdown code blocks, inline backticks, HTML tags, and URLs using regular expressions; filters against a curated 60+ token English and developer noise stopword list.
  - **Standard-Library TF-IDF**: Implements custom term frequency (`compute_tf`) and smoothed inverse document frequency (`compute_idf` with formula $\log\frac{1+N}{1+\text{df}(t)} + 1$) without external dependencies like scikit-learn.
  - **Keyword Extraction**: Computes top document keywords and corpus-wide aggregated TF-IDF keywords.
  - **Heuristic Pattern Categorization**: Implements weighted regex pattern matching across 40+ domain patterns for `bug` (e.g. *crash, traceback, segfault, 500, regression*), `feature` (e.g. *proposal, add support, enhancement*), `question` (e.g. *how to, is it possible*), and `docs` (e.g. *typo, readme, tutorial*). Titles are weighted $2\times$ over body text.
  - **Corpus Profiler & Exporter**: Computes title/body character and word-length distributions (min, max, mean, median), prints ASCII distribution histograms to stdout, and optionally serializes enriched records to `data/analyzed/*_analyzed.json`.
- **Status & Role**: Independent of the ML classification pipelines; safe to keep as a standalone analytical tool or reference for text normalization.

### 1.2 `src/data/`
- **Contents**: Contains an empty nested directory path: `src/data/external/nlbse2024/` (0 bytes, 0 files).
- **Origin**: Created during early project initialization before dataset storage was standardized at the repository root `data/external/nlbse2024/`.
- **Status**: Kept intact without modification. Active data assets remain exclusively under root `data/`.

### 1.3 `output/`
- **Contents**: Contains five subdirectories created during SetFit diagnostic and memory profiling experiments:
  - `output/test_mem/checkpoint-3/`: Intermediate HuggingFace `Trainer` checkpoint (1.3 GB) containing `model.safetensors` (438 MB), `optimizer.pt` (871 MB), tokenizers, and configuration files.
  - `output/test_quick_eval/`: Output directory from initial SetFit evaluation tests.
  - `output/test_quick_eval_1it/`: Output directory from 1-iteration contrastive fine-tuning timing tests.
  - `output/test_setfit/`: Output directory from initial SetFit instantiation tests.
  - `output/test_timing/`: Output directory from GPU step latency tests.
- **Status**: Excluded from version control via `.gitignore`; preserved intact on disk.

---

## 2. Active Python Training Processes

- **Check Method**: Process enumeration via Windows PowerShell (`Get-Process python*`, `tasklist /FI "IMAGENAME eq python.exe"`).
- **Result**: **0 active Python training processes**. No background training runs or orphaned worker processes are currently executing.

---

## 3. Ollama Installation & Qwen 4B Tag Availability

- **Installation Status**: **Installed**.
  - Binary Location: `C:\Users\sreek\AppData\Local\Programs\Ollama\ollama.exe`
  - Version: `0.40.15`
  - Background Service: The Ollama daemon is currently stopped.
- **Available Qwen 4B Tags in Ollama Registry**:
  - Primary 4B options: `qwen3:4b`, `qwen3.5:4b`
  - Specialized 4B option: `qwen3-embedding:4b`
  - Fallback 2-3B options (per Section 2 of spec): `qwen2.5:3b`, `qwen2.5-coder:3b`

---

## 4. Frozen Classifier Wrapper & Uncertainty Threshold

Per Section 1b and Section 5 of `RepoTriage_SPEC.md`:
- **Frozen Artifact**: `models/pooled_classifier.joblib` (Pooled TF-IDF + Logistic Regression).
- **Threshold Determination**: Analyzed confidence scores across all 1,500 validation records from Leave-One-Repository-Out (LOGO) cross-validation on the official training split (never from the test split):
  - At threshold $\tau = 0.50$, retained coverage is 60.1% (902 / 1,500) and accuracy on retained predictions rises from 59.4% to 67.4%, successfully filtering low-confidence borderline samples into `uncertain`.
  - Default threshold configured to **`0.50`**.
- **Implementation**:
  - `src/classify/predict.py`: Exposes `predict_issue_type(title, body="", threshold=0.50)`.
  - Returns `{label, predicted_category, scores, confidence, threshold, needs_review, model_version, benchmark_metrics}`.
  - Exposes the two mandatory labelled benchmark figures:
    - **Pooled Test Macro-F1**: `0.76` (repositories seen in training)
    - **Cold-Start Macro-F1**: `0.594` (leave-one-repository-out cross-validation)

---

## 5. CLI Acceptance Verification

The command-line interface was implemented in [`src/cli.py`](file:///src/cli.py):

```bash
# General usage
python -m src.cli classify "<issue text>" [--body "<body text>"] [--threshold 0.50]
```

### Verification Examples:

1. **Clear Bug (High Confidence)**:
   ```bash
   python -m src.cli classify "Crash on startup" --body "Traceback: NullPointerException at com.repo.Auth"
   ```
   *Output*:
   ```
   =================================================================
     REPOTRIAGE ISSUE CLASSIFICATION
   =================================================================
     Input Text   : Crash on startup
     Predicted Type: BUG
     Confidence   : 0.6860 (Threshold: 0.50)
     Needs Review : NO
     Model ID     : RepoTriage-Pooled-TFIDF-LogReg

     Class Confidence Scores:
       bug        : 0.6860  ████████████████████
       feature    : 0.1838  █████
       question   : 0.1302  ███

     Benchmark Reference Metrics:
       • Pooled Test Macro-F1   : 0.76 (repositories seen in training)
       • Cold-Start Macro-F1    : 0.594 (leave-one-repository-out cross-validation)
   =================================================================
   ```

2. **Feature Request**:
   ```bash
   python -m src.cli classify "Feature request: add support for async streaming responses"
   ```
   *Output*: `Predicted Type: FEATURE` (Confidence: 0.9793, `Needs Review: NO`).

3. **Ambiguous / Out-of-Domain Input (Uncertain Threshold Triggered)**:
   ```bash
   python -m src.cli classify "ValidationError when using a custom validator with Optional field in pydantic v2"
   ```
   *Output*: `Predicted Type: UNCERTAIN` (Confidence: 0.3788 < 0.50, `Needs Review: YES`).

---

## 6. Test Suite Status

- **Automated Unit Tests**: 43 tests passing in 0.58s:
  - `tests/test_cli.py` (4 tests): Verifies `predict_issue_type` schema, uncertainty thresholding, CLI output formatting, and empty input handling.
  - `tests/test_classifier.py` (6 tests): Pipeline features, dummy baseline, leakage prevention, serialization.
  - `tests/test_ingest.py` (21 tests): Retries, rate limits, pagination, PR filtering, normalization.
  - `tests/test_preprocessing.py` (9 tests): Text combination, sanitization, label mapping.
  - `tests/test_evaluation.py` (3 tests): Evaluation metrics and cross-repository scoring.
