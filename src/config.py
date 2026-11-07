"""
Central Configuration and Environment Loader for RepoTriage
============================================================
Safely parses `.env` and exports environment variables (HF_TOKEN, GITHUB_TOKEN)
before third-party libraries (Hugging Face, GitHub client) are initialized.
Never logs or prints secret tokens.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"


def load_env() -> None:
    """Parse .env file in project root and populate os.environ safely."""
    # Suppress Windows symlinks warning from Hugging Face Hub
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    if not ENV_PATH.exists():
        return

    try:
        with open(ENV_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip()
                if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                    val = val[1:-1]
                # Only set if not already set or empty
                if key and val and key not in os.environ:
                    os.environ[key] = val
    except Exception:
        pass


# Automatically load environment upon module import
load_env()

# Stage 6.6-B: Production calibrated guardrail thresholds
PRODUCTION_TAU_S: float = 0.80
PRODUCTION_TAU_C: float = 0.80

# Out-of-domain ANLI benchmark thresholds (tuned on ANLI dev split)
ANLI_TAU_S: float = 0.30
ANLI_TAU_C: float = 0.30
