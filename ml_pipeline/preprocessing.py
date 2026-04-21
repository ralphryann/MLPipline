"""
Load tickets, handle missing values, build multi-label targets, and split without leakage.

Critical fixes vs. the old notebook:
- Targets are true multi-label via ``MultiLabelBinarizer`` (category + priority tags).
- Missing values handled with nullable dtypes + explicit imputation for tabular features.
- ``MultiLabelBinarizer`` is fit **only on training rows** to avoid label leakage.
- Train / validation / test: validation never touches test; test used once for final metrics.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

from ml_pipeline.config import (
    CATEGORICAL_FEATURE_COLS,
    LEAKAGE_COLUMNS,
    NUMERIC_FEATURE_COLS,
    REQUIRED_RAW_COLUMNS,
    TEXT_COLUMN,
    SplitConfig,
)
from ml_pipeline.text_clean import clean_ticket_text


def _tag_category(val: Any) -> str | None:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    s = str(val).strip()
    if not s or s.lower() in {"", "nan", "none", "null", "na", "n/a"}:
        return None
    return f"category:{s}"


def _tag_priority(val: Any) -> str | None:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    s = str(val).strip()
    if not s.lower() in {"", "nan", "none", "null", "na", "n/a"}:
        return f"priority:{s}"
    return None


def row_to_label_set(row: pd.Series) -> list[str]:
    """Two active labels per typical row: category + priority (multi-label over union)."""
    tags: list[str] = []
    c = _tag_category(row.get("category"))
    p = _tag_priority(row.get("priority"))
    if c:
        tags.append(c)
    if p:
        tags.append(p)
    return tags


def load_raw_tickets(path: str | pd.PathLike[str], *, nrows: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(path, nrows=nrows, low_memory=False)
    missing = [c for c in REQUIRED_RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    return df


def build_preprocessed_frame(
    df: pd.DataFrame,
    *,
    drop_duplicate_tickets: bool = True,
) -> pd.DataFrame:
    """
    Structural cleaning on raw frame; adds ``clean_text`` and ``label_set`` columns.

    Does not fit ML estimators — safe to run on all data before splitting.
    """
    work = df.copy()

    # Coerce text / categoricals without turning NaN into the literal string "nan" prematurely
    work[TEXT_COLUMN] = work[TEXT_COLUMN].replace("", np.nan)

    # Label sets from raw category / priority (NaN -> omitted from set)
    work["label_set"] = work.apply(row_to_label_set, axis=1)

    # Drop rows with no usable text or no labels
    work["clean_text"] = work[TEXT_COLUMN].map(clean_ticket_text)
    valid = work["clean_text"].str.len() > 0
    has_label = work["label_set"].map(len) > 0
    work = work.loc[valid & has_label].reset_index(drop=True)

    if drop_duplicate_tickets:
        # Same description + same label set = duplicate for supervised learning
        work["_label_key"] = work["label_set"].map(lambda s: tuple(sorted(s)))
        work = work.drop_duplicates(subset=["clean_text", "_label_key"]).reset_index(drop=True)
        work = work.drop(columns=["_label_key"])

    # Optional tabular columns: keep only if present
    num_cols = [c for c in NUMERIC_FEATURE_COLS if c in work.columns]
    cat_cols = [c for c in CATEGORICAL_FEATURE_COLS if c in work.columns]

    for c in num_cols:
        work[c] = pd.to_numeric(work[c], errors="coerce")

    for c in cat_cols:
        work[c] = work[c].astype("string").replace({pd.NA: np.nan})

    leak_present = LEAKAGE_COLUMNS.intersection(work.columns)
    if leak_present:
        warnings.warn(
            "Dropping leakage columns from feature matrix: " + ", ".join(sorted(leak_present)),
            UserWarning,
            stacklevel=2,
        )
        work = work.drop(columns=list(leak_present), errors="ignore")

    # Modeling uses the canonical text column name with cleaned content (sklearn ColumnTransformer).
    work[TEXT_COLUMN] = work["clean_text"]

    return work


def make_train_val_test(
    df: pd.DataFrame,
    mlb: MultiLabelBinarizer | None,
    split: SplitConfig | None = None,
    *,
    stratify_key: str = "category",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, MultiLabelBinarizer, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split into train / val / test.

    Fits ``MultiLabelBinarizer`` on **train** only; transforms val/test.

    Stratification uses ``stratify_key`` column (single-label proxy) when cardinality allows.
    """
    split = split or SplitConfig()
    strat_col = stratify_key if stratify_key in df.columns else None
    if strat_col is None and "category" in df.columns:
        strat_col = "category"

    idx = np.arange(len(df))
    strat = df[strat_col] if strat_col else None

    if strat is not None:
        counts = strat.value_counts()
        if counts.min() < 2:
            strat = None

    idx_train, idx_temp = train_test_split(
        idx,
        test_size=split.test_size + split.val_size,
        random_state=split.random_state,
        stratify=strat,
    )

    strat_temp = df.loc[idx_temp, strat_col] if strat_col else None
    if strat_temp is not None and strat_temp.value_counts().min() < 2:
        strat_temp = None

    rel_test = split.test_size / (split.test_size + split.val_size)
    idx_val, idx_test = train_test_split(
        idx_temp,
        test_size=rel_test,
        random_state=split.random_state,
        stratify=strat_temp,
    )

    train_df = df.iloc[idx_train].reset_index(drop=True)
    val_df = df.iloc[idx_val].reset_index(drop=True)
    test_df = df.iloc[idx_test].reset_index(drop=True)

    if mlb is None:
        mlb = MultiLabelBinarizer(sparse_output=False)
        Y_train = mlb.fit_transform(train_df["label_set"])
    else:
        Y_train = mlb.transform(train_df["label_set"])

    classes_set = set(mlb.classes_)

    def _only_known_labels(ls: list[str]) -> bool:
        return bool(ls) and all(t in classes_set for t in ls)

    n_val_before = len(val_df)
    val_df = val_df.loc[val_df["label_set"].map(_only_known_labels)].reset_index(drop=True)
    n_test_before = len(test_df)
    test_df = test_df.loc[test_df["label_set"].map(_only_known_labels)].reset_index(drop=True)

    if len(val_df) < n_val_before:
        warnings.warn(
            f"Dropped {n_val_before - len(val_df)} validation rows with labels unseen in train.",
            UserWarning,
            stacklevel=2,
        )
    if len(test_df) < n_test_before:
        warnings.warn(
            f"Dropped {n_test_before - len(test_df)} test rows with labels unseen in train.",
            UserWarning,
            stacklevel=2,
        )

    Y_val = mlb.transform(val_df["label_set"])
    Y_test = mlb.transform(test_df["label_set"])

    return train_df, val_df, test_df, mlb, Y_train, Y_val, Y_test
