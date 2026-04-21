"""Multi-label classification metrics (micro/macro F1, Hamming, precision/recall)."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    hamming_loss,
    precision_recall_fscore_support,
)


def multilabel_metric_bundle(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    threshold: float = 0.5,
    y_scores: np.ndarray | None = None,
) -> dict[str, Any]:
    """
    Compute standard multi-label metrics.

    Parameters
    ----------
    y_true : (n_samples, n_labels) int {0,1}
    y_pred : same, or probabilities if y_scores is None and values in [0,1] float
    y_scores : optional raw scores; binarized with ``threshold``
    """
    yt = np.asarray(y_true, dtype=int)
    if y_scores is not None:
        yhat = (np.asarray(y_scores) >= threshold).astype(int)
    else:
        yhat = np.asarray(y_pred)
        if np.issubdtype(yhat.dtype, np.floating):
            yhat = (yhat >= threshold).astype(int)
        else:
            yhat = yhat.astype(int)

    p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(
        yt, yhat, average="micro", zero_division=0
    )
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        yt, yhat, average="macro", zero_division=0
    )

    # Subset accuracy (exact label-set match) — strict but interpretable
    subset_acc = accuracy_score(yt, yhat)

    return {
        "hamming_loss": float(hamming_loss(yt, yhat)),
        "f1_micro": float(f1_micro),
        "f1_macro": float(f1_macro),
        "precision_micro": float(p_micro),
        "recall_micro": float(r_micro),
        "precision_macro": float(p_macro),
        "recall_macro": float(r_macro),
        "subset_accuracy": float(subset_acc),
    }
