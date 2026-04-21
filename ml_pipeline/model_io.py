"""
Save and load trained artifacts for production and “continue” workflows.

- Sklearn ``Pipeline`` + ``MultiLabelBinarizer`` (+ feature column list) via joblib.
- Label names JSON next to HF checkpoints for deployment decoding.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer


def save_sklearn_bundle(
    output_dir: str | Path,
    pipeline: Pipeline,
    mlb: MultiLabelBinarizer,
    feature_columns: list[str],
) -> None:
    """Persist fitted sklearn pipeline, label binarizer, and feature column order."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": pipeline,
            "mlb": mlb,
            "feature_columns": feature_columns,
        },
        out / "sklearn_bundle.joblib",
    )
    (out / "label_names.json").write_text(
        json.dumps(mlb.classes_.tolist(), indent=2),
        encoding="utf-8",
    )


def load_sklearn_bundle(
    input_dir: str | Path,
) -> tuple[Pipeline, MultiLabelBinarizer, list[str]]:
    data: dict[str, Any] = joblib.load(Path(input_dir) / "sklearn_bundle.joblib")
    return data["pipeline"], data["mlb"], data["feature_columns"]


def binary_matrix_to_label_lists(mlb: MultiLabelBinarizer, Y: np.ndarray) -> list[list[str]]:
    """Turn (n_samples, n_labels) binary predictions into human-readable tag lists."""
    Y = np.asarray(Y, dtype=int)
    names = mlb.classes_
    return [[str(names[j]) for j in np.flatnonzero(row)] for row in Y]


def predict_sklearn_multilabel(
    pipeline: Pipeline,
    mlb: MultiLabelBinarizer,
    feature_columns: list[str],
    X: pd.DataFrame,
) -> tuple[np.ndarray, list[list[str]]]:
    """
    Run inference on a feature frame with the same columns as training.

    Returns binary matrix and decoded label lists per row.
    """
    Xf = X[feature_columns]
    Y_hat = pipeline.predict(Xf)
    return Y_hat, binary_matrix_to_label_lists(mlb, Y_hat)


def save_label_names(output_dir: str | Path, label_names: list[str]) -> None:
    """Write label vocabulary (aligns with HF classifier head order)."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "label_names.json").write_text(json.dumps(label_names, indent=2), encoding="utf-8")


def load_label_names(model_dir: str | Path) -> list[str]:
    path = Path(model_dir) / "label_names.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}")
    return json.loads(path.read_text(encoding="utf-8"))
