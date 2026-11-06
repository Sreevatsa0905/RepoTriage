"""
Configuration for the RepoTriage classification pipeline.

All experiment settings, hyperparameters, and paths in one place.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_DIR = PROJECT_ROOT / "data"
EXTERNAL_DIR = DATA_DIR / "external" / "nlbse2024"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

TRAIN_CSV = EXTERNAL_DIR / "issues_train.csv"
TEST_CSV = EXTERNAL_DIR / "issues_test.csv"

# ---------------------------------------------------------------------------
# Dataset URLs (official NLBSE 2024 GitHub repository)
# ---------------------------------------------------------------------------
DATASET_URL_TRAIN = (
    "https://raw.githubusercontent.com/nlbse2024/"
    "issue-report-classification/main/data/issues_train.csv"
)
DATASET_URL_TEST = (
    "https://raw.githubusercontent.com/nlbse2024/"
    "issue-report-classification/main/data/issues_test.csv"
)

EXPECTED_REPOS = [
    "facebook/react",
    "tensorflow/tensorflow",
    "microsoft/vscode",
    "bitcoin/bitcoin",
    "opencv/opencv",
]

EXPECTED_LABELS = ["bug", "feature", "question"]

EXPECTED_TRAIN_COUNT = 1500
EXPECTED_TEST_COUNT = 1500

# CSV columns as used in the official dataset
COL_REPO = "repo"
COL_CREATED_AT = "created_at"
COL_LABEL = "label"
COL_TITLE = "title"
COL_BODY = "body"

# ---------------------------------------------------------------------------
# Classical ML (Model A) — TF-IDF + Logistic Regression
# ---------------------------------------------------------------------------
CLASSICAL_CONFIG = {
    "random_seed": 42,
    "val_fraction": 0.2,  # held from training for HP selection

    # TF-IDF word-level
    "tfidf_word": {
        "analyzer": "word",
        "ngram_range": (1, 2),
        "max_features": 50_000,
        "sublinear_tf": True,
        "min_df": 2,
        "max_df": 0.95,
    },

    # TF-IDF char-level
    "tfidf_char": {
        "analyzer": "char_wb",
        "ngram_range": (3, 5),
        "max_features": 50_000,
        "sublinear_tf": True,
        "min_df": 2,
        "max_df": 0.95,
    },

    # Logistic Regression
    "logreg": {
        "C": 1.0,
        "max_iter": 1000,
        "solver": "lbfgs",
        "class_weight": "balanced",
        "random_state": 42,
    },

    # Hyperparameter search space (C values)
    "hp_search": {
        "C_values": [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
    },
}

# ---------------------------------------------------------------------------
# SetFit (Model B)
# ---------------------------------------------------------------------------
SETFIT_CONFIG = {
    "model_name": "sentence-transformers/all-mpnet-base-v2",
    "num_iterations": 1,  # 1 iteration per sample = 600 pairs per repo; optimized for local GPU constraint
    "num_epochs": 1,
    "batch_size": (16, 2),
    "body_learning_rate": 2e-5,
    "random_seed": 42,
    "max_length": 128,
    "use_amp": True,
}

# ---------------------------------------------------------------------------
# Pooled model (Model C)
# ---------------------------------------------------------------------------
POOLED_CONFIG = {
    "random_seed": 42,
    "method": "tfidf_logreg",
}

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
