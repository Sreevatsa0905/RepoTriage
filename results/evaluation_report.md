# RepoTriage — Issue Classification Engine: Evaluation Report
**Project Milestone**: Microproject 2: Issue Classification Engine  
**Dataset**: Official NLBSE 2024 Issue Report Classification Benchmark (`issues_train.csv` / `issues_test.csv`)  
**Evaluation Date**: 2026-11-06  

---

## 1. Executive Summary

This report documents the empirical evaluation of three model architectures and one dummy baseline for GitHub issue report classification (`bug`, `feature`, `question`) on the official NLBSE 2024 dataset (3,000 issues across 5 target repositories: `facebook/react`, `tensorflow/tensorflow`, `microsoft/vscode`, `bitcoin/bitcoin`, and `opencv/opencv`).

### Overall Cross-Repository Benchmark Summary

| Model Approach | Method / Backbone | Training Data | Mean Accuracy | Mean Macro F1 | Mean Weighted F1 | External Reference F1 | Delta vs Published |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | DummyClassifier (majority class) | Repo Train (300) | 0.3333 | 0.1667 | 0.1667 | — | — |
| **Model A** | TF-IDF + Logistic Regression | Repo Train (300) | 0.7453 | 0.7438 | 0.7438 | — | — |
| **Model B** | SetFit (`all-mpnet-base-v2`) | Repo Train (300) | **0.7780** | **0.7751** | **0.7751** | 0.8270 | -0.0519 |
| **Model C** | Pooled TF-IDF + LogReg | All Train (1500) | 0.7607 | 0.7591 | 0.7605 | — | — |

*Note: Per the NLBSE 2024 protocol, benchmark cross-repository scores represent the arithmetic mean of the five repository-specific macro F1 scores on the official test set (1,500 records, 300 per repository).*

---

## 2. Official 5-Repository Benchmark Results

Per the official competition track, separate independent models were trained strictly on each repository's 300 training records and evaluated strictly on each repository's 300 test records. No cross-repository data or metadata was exposed during training or hyperparameter selection.

### 2.1 Per-Repository Performance Breakdown

| Repository | Dummy Macro F1 | Classical (Model A) Macro F1 | Classical Best Config | SetFit (Model B) Macro F1 | SetFit Accuracy | SetFit Delta vs Classical |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `facebook/react` | 0.1667 | 0.7918 | word, C=5.0 | **0.8043** | 0.8100 | +0.0125 |
| `tensorflow/tensorflow` | 0.1667 | 0.7746 | word, C=10.0 | **0.7853** | 0.7867 | +0.0107 |
| `microsoft/vscode` | 0.1667 | 0.6895 | word+char, C=0.1 | **0.7742** | 0.7767 | **+0.0847** |
| `bitcoin/bitcoin` | 0.1667 | 0.7163 | word, C=5.0 | **0.7397** | 0.7433 | +0.0234 |
| `opencv/opencv` | 0.1667 | 0.7467 | word+char, C=2.0 | **0.7722** | 0.7733 | +0.0255 |
| **Arithmetic Mean** | **0.1667** | **0.7438** | — | **0.7751** | **0.7780** | **+0.0313** |

### 2.2 Per-Class Breakdown (Model B: SetFit)

| Class | Precision (Mean) | Recall (Mean) | F1 (Mean) |
| :--- | :---: | :---: | :---: |
| `bug` | 0.7996 | 0.8000 | 0.7991 |
| `feature` | 0.7815 | 0.8700 | 0.8226 |
| `question` | 0.7524 | 0.6640 | 0.7037 |

### 2.3 Model A (Classical) vs Model B (SetFit) Analysis

1. **Transformer Semantics vs N-Grams**: SetFit outperformed TF-IDF + Logistic Regression across every single repository. The gain was largest on `microsoft/vscode` (+8.47% F1), where issues frequently feature ambiguous error reports and code snippets that benefit heavily from pre-trained contextual embeddings.
2. **Class Imbalance Sensitivity**: While class support is perfectly balanced (100 per class per repo), `question` issues proved hardest for all models (Model A F1 = 0.7044, Model B F1 = 0.7037). Users often phrase questions as bugs ("Why does X fail?") or features ("Can we do Y?"), leading to semantic overlap. `feature` achieved the highest F1 (0.8226 for SetFit) driven by strong intent verbs (*add, support, allow, provide*).

---

## 3. Comparison with Published SetFit Benchmark Reference

The official NLBSE 2024 competition README cites an overall SetFit baseline macro F1 of **~0.8270**.

- **Published Reference Score**: 0.8270 Macro F1
- **Our Independently Reproduced Score**: **0.7751 Macro F1**
- **Observed Difference**: -0.0519

### Documented Implementation Deviations & Rationale:

1. **Pair Generation Iterations (`num_iterations`)**:
   - *Official Colab Notebook*: Used `num_iterations = 20` (generating $300 \times 20 \times 2 = 12,000$ contrastive pairs per repository). On high-end datacenter GPUs (A100/V100 with 40–80GB VRAM), this executes in minutes.
   - *Our Environment*: On local hardware (NVIDIA RTX 3050 6GB Laptop GPU with a 15–50W power envelope), 12,000 pairs required ~2.5 hours per repository (~12.5 hours total). To ensure deterministic completion, avoid GPU thermal throttling, and prevent CUDA out-of-memory errors, `num_iterations` was configured to `1` (600 contrastive pairs per repository, ~7.5 minutes per repo, ~38 minutes total).
2. **Maximum Sequence Length (`max_seq_length`)**:
   - The official notebook left tokenizer truncation unbounded or at 384 tokens. Due to GitHub issue bodies containing large stack traces (up to 202 KB), full self-attention on long texts caused CUDA Out-of-Memory faults.
   - We capped `max_seq_length = 128` and truncated raw text strings to 1,500 characters, while enabling Automatic Mixed Precision (FP16 `use_amp=True`).
3. **Conclusion**: Despite utilizing only 5% of the contrastive pairs ($1/20$th), the fine-tuned SetFit models attained 0.7751 Macro F1 (within 5.2% of the published baseline), validating the efficiency of contrastive few-shot learning for issue triage.

---

## 4. Model C: RepoTriage Pooled Classifier

For real-world deployment in RepoTriage, a production system must triage issues from arbitrary repositories rather than maintaining separate models per repository.

### 4.1 Leave-One-Repository-Out (LOGO) Cross-Validation (Training Split Only)

To evaluate true out-of-domain cross-repository generalization, we conducted Leave-One-Group-Out CV on the 1,500 training records. In each fold, an entire repository's 300 issues were held out as the validation set, while the model was trained on the remaining 4 repositories (1,200 issues).

| Held-out Validation Repository | F1 Score |
| :--- | :---: |
| `facebook/react` | 0.5673 |
| `tensorflow/tensorflow` | 0.6348 |
| `microsoft/vscode` | 0.5997 |
| `bitcoin/bitcoin` | 0.5769 |
| `opencv/opencv` | 0.5916 |
| **Mean LOGO Cross-Validation F1** | **0.5941** |

*Selected Best Configuration*: `feature_mode = 'word_char'` (word unigrams/bigrams + char 3-5 n-grams) with $C = 2.0$.

### 4.2 Official Test Split Evaluation

After hyperparameter selection via LOGO CV, the pooled classifier was retrained on the full 1,500 training issues and evaluated once on the 1,500 official test issues:

- **Overall Accuracy**: **0.7607** (1,141 / 1,500 correct)
- **Overall Macro F1**: **0.7605**
- **Overall Weighted F1**: **0.7605**
- **Arithmetic Mean of Repo Macro F1s**: **0.7591**

#### Test Set Performance by Repository (Pooled Model):
- `facebook/react`: Macro F1 = 0.7938 (Acc: 0.7967)
- `tensorflow/tensorflow`: Macro F1 = 0.7648 (Acc: 0.7667)
- `microsoft/vscode`: Macro F1 = 0.7391 (Acc: 0.7400)
- `bitcoin/bitcoin`: Macro F1 = 0.7249 (Acc: 0.7267)
- `opencv/opencv`: Macro F1 = 0.7731 (Acc: 0.7733)

---

## 5. Understanding Generalization: Three Key Performance Regimes

The empirical results reveal three distinct performance tiers that must not be conflated:

```
┌────────────────────────────────────────────────────────────────────────┐
│  1. Repository-Specific Benchmark (Train on Repo A, Test on Repo A)    │
│     - Mean Macro F1: 0.7751 (SetFit) / 0.7438 (Classical)              │
│     - In-domain distribution; captures project-specific jargon         │
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
   The official test split contains issues from the same five repositories seen during pooled training. Thus, project-specific jargon (e.g., `useState`, `tensor`, `bitcoind`, `Mat`) is known to the model.
2. **Why LOGO CV (0.5941) Reflects True Cold-Start Performance**:
   When RepoTriage encounters a brand-new repository that has never contributed training data, domain-specific terminology provides zero signal. The model must rely purely on generic issue phrasing (*"error: expected X but got Y"*, *"would it be possible to add..."*). The 0.5941 F1 demonstrates the expected baseline when onboarding cold repositories.
3. **Engineering Implication**:
   For new repositories, RepoTriage should start with the Pooled Classifier, but offer a lightweight fine-tuning or few-shot adaptation loop as the repository accumulates triaged issues.

---

## 6. Production Inference Interface

The classifier is exposed through a decoupled inference service (`src/classifier/inference.py`):

```python
from src.classifier.inference import IssueClassifierService

# Instantiate service from saved artifact
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

The interface:
- Standardizes output schema via `IssueClassificationResult`.
- Handles `None`, empty strings, and non-string types safely.
- Decouples feature preprocessing and inference from pipeline training.
- Ready for zero-rewrite embedding inside FastAPI endpoints in Microproject 3.

---

## 7. Artifacts and Generated Deliverables

| File / Directory | Purpose |
| :--- | :--- |
| `DATASET.md` | Dataset provenance, schema, profiling, and license disclaimer |
| `models/pooled_classifier.joblib` | Serialized production pooled classifier |
| `models/classical_*.joblib` | 5 repository-specific classical models |
| `models/setfit_*/` | 5 repository-specific fine-tuned SetFit backbones |
| `results/dataset_profile.json` | Dataset validation and statistics profile |
| `results/benchmark_results.json` | Complete benchmark metrics (Dummy, Classical, SetFit) |
| `results/benchmark_results.csv` | Tabular benchmark metrics for all repos |
| `results/pooled_results.json` | LOGO cross-validation and official test evaluation |
| `results/figures/cm_*.png` | 11 confusion matrix visualization plots |
