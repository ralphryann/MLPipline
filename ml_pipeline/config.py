"""Paths, seeds, and default hyperparameters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "customer_support_tickets_200k.csv"


@dataclass(frozen=True)
class SplitConfig:
    """Train / validation / test ratios (validation is carved from train pool)."""

    test_size: float = 0.15
    val_size: float = 0.15  # fraction of (train+val) pool
    random_state: int = 42


@dataclass(frozen=True)
class TransformerTrainConfig:
    model_name: str = "distilbert-base-uncased"
    max_length: int = 256
    num_train_epochs: int = 2
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 32
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    output_dir: str = "outputs/distilbert_multilabel"
    prediction_threshold: float = 0.5
    fp16: bool = False  # set True on CUDA for speed (Trainer enables when safe)


# Columns that must exist in the raw CSV
REQUIRED_RAW_COLUMNS = [
    "issue_description",
    "category",
    "priority",
]

# Outcome / post-hoc fields — never use as features for triage from issue text
LEAKAGE_COLUMNS = frozenset(
    {
        "resolution_notes",
        "resolution_time_hours",
        "ticket_resolved_date",
        "customer_satisfaction_score",
        "status",  # often updated during ticket lifecycle
        "sla_breached",
        "escalated",
        "first_response_time_hours",
    }
)

# Safe tabular features (snapshot / customer context, not resolution outcomes)
NUMERIC_FEATURE_COLS = [
    "customer_age",
    "customer_tenure_months",
    "previous_tickets",
    "issue_complexity_score",
]

CATEGORICAL_FEATURE_COLS = [
    "product",
    "channel",
    "region",
    "subscription_type",
    "customer_gender",
    "operating_system",
    "browser",
    "language",
    "preferred_contact_time",
    "customer_segment",
]

TEXT_COLUMN = "issue_description"
