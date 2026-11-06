"""
SetFit Few-Shot Classification Implementation (Model B)
=========================================================
Implements the official NLBSE 2024 SetFit baseline using sentence-transformers/all-mpnet-base-v2.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import torch

# Ensure HuggingFace uses local cache and avoids unnecessary network polls
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from src.classifier.config import EXPECTED_LABELS, MODELS_DIR, RANDOM_SEED, SETFIT_CONFIG
from src.classifier.preprocessing import combine_issue_text

logger = logging.getLogger(__name__)


class SetFitClassifierWrapper:
    """
    Standardized wrapper around a fine-tuned SetFitModel.
    """

    def __init__(self, model: Any, model_id: str):
        self.model = model
        self.model_id = model_id

    def predict(self, texts: list[str]) -> list[str]:
        preds = self.model.predict(texts)
        if hasattr(preds, "tolist"):
            preds = preds.tolist()
        return [str(p) for p in preds]

    def predict_proba(self, texts: list[str]) -> list[dict[str, float]]:
        if hasattr(self.model, "predict_proba"):
            try:
                probs = self.model.predict_proba(texts)
                if hasattr(probs, "tolist"):
                    probs = probs.tolist()
                classes = list(getattr(self.model, "labels", EXPECTED_LABELS))
                results = []
                for row in probs:
                    results.append({str(cls_name): float(p) for cls_name, p in zip(classes, row)})
                return results
            except Exception as e:
                logger.warning("predict_proba failed: %s, falling back to 1.0 for predicted class", e)

        # Fallback if probability head not available
        preds = self.predict(texts)
        return [{lbl: (1.0 if lbl == p else 0.0) for lbl in EXPECTED_LABELS} for p in preds]

    def save(self, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(output_dir))
        logger.info("Saved SetFit model to %s", output_dir)
        return output_dir

    @classmethod
    def load(cls, model_dir: Path) -> SetFitClassifierWrapper:
        from setfit import SetFitModel
        model = SetFitModel.from_pretrained(str(model_dir), local_files_only=True)
        if torch.cuda.is_available():
            model.to("cuda")
        model_id = f"SetFit-{model_dir.name}"
        return cls(model=model, model_id=model_id)


def train_setfit_repo_model(
    repo: str,
    train_records: list[dict[str, str]],
    model_name: str | None = None,
    num_iterations: int | None = None,
    num_epochs: int | None = None,
    batch_size: tuple[int, int] | int | None = None,
    learning_rate: float | None = None,
    random_seed: int = RANDOM_SEED,
    max_length: int | None = None,
    use_amp: bool | None = None,
) -> tuple[SetFitClassifierWrapper, dict[str, Any]]:
    """
    Train a repository-specific SetFit model following the official NLBSE 2024 benchmark notebook.
    """
    from datasets import Dataset
    from setfit import SetFitModel, Trainer, TrainingArguments

    if model_name is None:
        model_name = SETFIT_CONFIG["model_name"]
    if num_iterations is None:
        num_iterations = SETFIT_CONFIG["num_iterations"]
    if num_epochs is None:
        num_epochs = SETFIT_CONFIG["num_epochs"]
    if batch_size is None:
        batch_size = SETFIT_CONFIG["batch_size"]
    if learning_rate is None:
        learning_rate = SETFIT_CONFIG["body_learning_rate"]
    if max_length is None:
        max_length = SETFIT_CONFIG.get("max_length", 128)
    if use_amp is None:
        use_amp = SETFIT_CONFIG.get("use_amp", True)

    safe_name = repo.replace("/", "-")
    output_dir = MODELS_DIR / f"setfit_{safe_name}"

    # Filter records for this repository
    repo_records = [r for r in train_records if r["repo"] == repo]
    texts = [
        combine_issue_text(
            r.get("title", ""),
            (r.get("body") or "")[:1500] if r.get("body") else "",
        )
        for r in repo_records
    ]
    labels = [r["label"].strip().lower() for r in repo_records]

    dataset_dict = {
        "text": texts,
        "label": labels,
    }
    train_dataset = Dataset.from_dict(dataset_dict).shuffle(seed=random_seed)

    logger.info("Instantiating SetFit backbone '%s' for '%s'...", model_name, repo)
    model = SetFitModel.from_pretrained(model_name, local_files_only=True)
    if torch.cuda.is_available():
        model.to("cuda")
    if hasattr(model, "model_body") and hasattr(model.model_body, "max_seq_length"):
        model.model_body.max_seq_length = max_length

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        save_strategy="no",
        report_to=[],  # disable wandb
        run_name=safe_name,
        seed=random_seed,
        batch_size=batch_size,
        num_epochs=num_epochs,
        num_iterations=num_iterations,
        body_learning_rate=learning_rate,
        max_length=max_length,
        use_amp=use_amp,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    logger.info("Starting SetFit contrastive training on %d samples for %s...", len(texts), repo)
    trainer.train()

    # Save fine-tuned model
    wrapper = SetFitClassifierWrapper(model=model, model_id=f"setfit-{safe_name}")
    wrapper.save(output_dir)

    metadata = {
        "repo": repo,
        "model_name": model_name,
        "num_iterations": num_iterations,
        "num_epochs": num_epochs,
        "batch_size": list(batch_size) if isinstance(batch_size, (tuple, list)) else batch_size,
        "learning_rate": learning_rate,
        "max_length": max_length,
        "use_amp": use_amp,
        "train_samples": len(texts),
        "model_dir": str(output_dir),
    }
    return wrapper, metadata
