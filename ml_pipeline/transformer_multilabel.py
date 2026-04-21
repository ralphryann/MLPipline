"""
Hugging Face transformer fine-tuning for multi-label ticket classification.

- ``problem_type=\"multi_label_classification\"`` + BCEWithLogitsLoss (handled by Trainer).
- Validation during training **never** uses the holdout test set.
- Tokenization uses ``datasets`` map + ``DataCollatorWithPadding`` for memory-efficient batching.
"""

from __future__ import annotations

import inspect
import os
from typing import Any

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import f1_score, hamming_loss
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from ml_pipeline.config import TransformerTrainConfig
from ml_pipeline.model_io import save_label_names
from ml_pipeline.text_clean import clean_ticket_text


def _build_hf_dataset(texts: list[str], Y: np.ndarray) -> Dataset:
    labels = [row.astype(float).tolist() for row in np.asarray(Y, dtype=float)]
    return Dataset.from_dict({"text": texts, "labels": labels})


def train_multilabel_transformer(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    Y_train: np.ndarray,
    Y_val: np.ndarray,
    *,
    text_column: str = "clean_text",
    cfg: TransformerTrainConfig | None = None,
    label_names: list[str] | None = None,
    resume_from_checkpoint: str | bool | None = None,
) -> tuple[Trainer, Any]:
    """
    Fine-tune a multi-label sequence classifier.

    Parameters
    ----------
    label_names
        Vocabulary aligned with columns of ``Y_train`` / ``Y_val`` (e.g. ``mlb.classes_.tolist()``).
        Saved next to the checkpoint for deployment decoding.
    resume_from_checkpoint
        Hugging Face Trainer resume: checkpoint directory, or ``True`` to resume from ``output_dir``.
    """
    cfg = cfg or TransformerTrainConfig()
    os.makedirs(cfg.output_dir, exist_ok=True)

    num_labels = Y_train.shape[1]
    if label_names is not None and len(label_names) != num_labels:
        raise ValueError("label_names length must match Y_train.shape[1]")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    def tokenize_batch(batch: dict[str, list]) -> dict[str, Any]:
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=cfg.max_length,
            padding=False,
        )

    train_ds = _build_hf_dataset(
        [clean_ticket_text(t) for t in train_df[text_column].tolist()],
        Y_train,
    )
    val_ds = _build_hf_dataset(
        [clean_ticket_text(t) for t in val_df[text_column].tolist()],
        Y_val,
    )

    train_ds = train_ds.map(tokenize_batch, batched=True, remove_columns=["text"])
    val_ds = val_ds.map(tokenize_batch, batched=True, remove_columns=["text"])

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=num_labels,
        problem_type="multi_label_classification",
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = torch.sigmoid(torch.tensor(logits)).numpy()
        preds = (probs >= cfg.prediction_threshold).astype(int)
        labels = labels.astype(int)
        return {
            "f1_micro": float(f1_score(labels, preds, average="micro", zero_division=0)),
            "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
            "hamming_loss": float(hamming_loss(labels, preds)),
        }

    supported = set(inspect.signature(TrainingArguments.__init__).parameters)
    kwargs: dict[str, Any] = {
        "output_dir": cfg.output_dir,
        "learning_rate": cfg.learning_rate,
        "weight_decay": cfg.weight_decay,
        "warmup_ratio": cfg.warmup_ratio,
        "num_train_epochs": cfg.num_train_epochs,
        "per_device_train_batch_size": cfg.per_device_train_batch_size,
        "per_device_eval_batch_size": cfg.per_device_eval_batch_size,
        "save_strategy": "epoch",
        "logging_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1_micro",
        "greater_is_better": True,
        "save_total_limit": 2,
        "report_to": "none",
    }
    if cfg.fp16 and torch.cuda.is_available():
        kwargs["fp16"] = True
    if "evaluation_strategy" in supported:
        kwargs["evaluation_strategy"] = "epoch"
    elif "eval_strategy" in supported:
        kwargs["eval_strategy"] = "epoch"
    else:
        kwargs.pop("load_best_model_at_end", None)
        kwargs.pop("metric_for_best_model", None)
        kwargs.pop("greater_is_better", None)

    kwargs = {k: v for k, v in kwargs.items() if k in supported}
    training_args = TrainingArguments(**kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    train_kw: dict[str, Any] = {}
    if resume_from_checkpoint:
        train_kw["resume_from_checkpoint"] = resume_from_checkpoint

    trainer.train(**train_kw)
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    if label_names is not None:
        save_label_names(cfg.output_dir, label_names)
    return trainer, tokenizer


def load_multilabel_transformer(
    model_dir: str | os.PathLike[str],
    *,
    device_map: str | dict[str, int] | None = None,
) -> tuple[Any, Any]:
    """
    Load model + tokenizer from disk (no ``Trainer`` required for inference).

    ``device_map`` is passed to ``from_pretrained`` when supported (e.g. ``\"auto\"`` on GPU).
    """
    path = str(model_dir)
    load_kw: dict[str, Any] = {}
    if device_map is not None:
        load_kw["device_map"] = device_map

    tokenizer = AutoTokenizer.from_pretrained(path)
    try:
        model = AutoModelForSequenceClassification.from_pretrained(path, **load_kw)
    except TypeError:
        model = AutoModelForSequenceClassification.from_pretrained(path)
    model.eval()
    return model, tokenizer


def predict_multilabel_with_model(
    texts: list[str],
    model: Any,
    tokenizer: Any,
    *,
    threshold: float = 0.5,
    max_length: int = 256,
    device: torch.device | None = None,
    batch_size: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """Run sigmoid multi-label inference given a loaded HF model + tokenizer."""
    model.eval()
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    cleaned = [clean_ticket_text(t) for t in texts]
    all_probs: list[np.ndarray] = []

    for start in range(0, len(cleaned), batch_size):
        batch = cleaned[start : start + batch_size]
        enc = tokenizer(
            batch,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
            all_probs.append(torch.sigmoid(logits).cpu().numpy())

    n_labels = int(getattr(model.config, "num_labels", 0)) or (
        all_probs[0].shape[1] if all_probs else 0
    )
    probs = np.vstack(all_probs) if all_probs else np.zeros((0, n_labels))
    preds = (probs >= threshold).astype(int)
    return preds, probs


def predict_multilabel_transformer(
    texts: list[str],
    trainer: Trainer,
    tokenizer: Any,
    *,
    threshold: float = 0.5,
    max_length: int = 256,
    device: torch.device | None = None,
    batch_size: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (binary_predictions, probabilities) for raw ticket strings
    (cleaned with ``clean_ticket_text`` before tokenization).
    """
    return predict_multilabel_with_model(
        texts,
        trainer.model,
        tokenizer,
        threshold=threshold,
        max_length=max_length,
        device=device,
        batch_size=batch_size,
    )


def evaluate_multilabel_transformer_on_test(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    Y_true: np.ndarray,
    *,
    threshold: float = 0.5,
    max_length: int = 256,
    batch_size: int = 32,
) -> dict[str, Any]:
    """Convenience: predict on a list of texts and return sklearn-style multi-label metrics dict."""
    from ml_pipeline.metrics import multilabel_metric_bundle

    preds, probs = predict_multilabel_with_model(
        texts,
        model,
        tokenizer,
        threshold=threshold,
        max_length=max_length,
        batch_size=batch_size,
    )
    metrics = multilabel_metric_bundle(Y_true, preds, y_scores=probs, threshold=threshold)
    metrics["n_samples"] = len(texts)
    return metrics
