"""Select modeling feature columns present in a DataFrame (text + safe tabular)."""

from __future__ import annotations

import pandas as pd

from ml_pipeline.config import CATEGORICAL_FEATURE_COLS, NUMERIC_FEATURE_COLS, TEXT_COLUMN


def modeling_feature_columns(df: pd.DataFrame) -> list[str]:
    cols = [TEXT_COLUMN]
    for c in NUMERIC_FEATURE_COLS + CATEGORICAL_FEATURE_COLS:
        if c in df.columns:
            cols.append(c)
    return cols
