"""
Scikit-learn baseline: heterogeneous features + OneVsRest linear model.

Uses ``ColumnTransformer`` + ``OneVsRestClassifier`` for true multi-label prediction.
Cross-validation fits on training folds only (test held out separately).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold, cross_validate
from sklearn.multioutput import ClassifierChain
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from ml_pipeline.config import CATEGORICAL_FEATURE_COLS, NUMERIC_FEATURE_COLS, TEXT_COLUMN
from ml_pipeline.metrics import multilabel_metric_bundle
from ml_pipeline.text_clean import clean_ticket_text


def _ensure_text_column(X: pd.DataFrame) -> pd.DataFrame:
    if TEXT_COLUMN not in X.columns:
        raise ValueError(f"Expected column {TEXT_COLUMN!r} in X.")
    return X


def _text_clean_column(X: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Receive a single-column slice from ColumnTransformer (DataFrame or 2D array)."""
    if hasattr(X, "iloc"):
        col = X.iloc[:, 0]
        return col.map(clean_ticket_text).to_numpy()
    arr = np.asarray(X).ravel()
    return np.array([clean_ticket_text(str(t)) for t in arr], dtype=object)


def build_sklearn_multilabel_pipeline(
    X_layout: pd.DataFrame | None = None,
    *,
    max_features: int = 30_000,
    ngram_range: tuple[int, int] = (1, 2),
    C: float = 4.0,
) -> Pipeline:
    """
    Full preprocessing + OneVsRest(LogisticRegression) for sparse multi-label Y.

    Text: TF-IDF on cleaned issue description (cleaning inside vectorizer pipeline).
    Numeric: median impute + StandardScaler.
    Categorical: impute + one-hot (unknown categories ignored at test time).

    Pass ``X_layout`` (any row slice of your feature matrix) so missing optional
    columns are skipped instead of erroring at fit time.
    """
    numeric_cols = [c for c in NUMERIC_FEATURE_COLS]
    categorical_cols = [c for c in CATEGORICAL_FEATURE_COLS]
    if X_layout is not None:
        numeric_cols = [c for c in numeric_cols if c in X_layout.columns]
        categorical_cols = [c for c in categorical_cols if c in X_layout.columns]

    text_pipe = Pipeline(
        steps=[
            ("clean", FunctionTransformer(_text_clean_column, validate=False)),
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=max_features,
                    ngram_range=ngram_range,
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
        ]
    )

    transformers: list[tuple[str, Any, list[str]]] = [
        ("text", text_pipe, [TEXT_COLUMN]),
    ]

    if numeric_cols:
        numeric_pipe = Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]
        )
        transformers.append(("num", numeric_pipe, numeric_cols))

    if categorical_cols:
        categorical_pipe = Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="most_frequent")),
                (
                    "oh",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=True, max_categories=50),
                ),
            ]
        )
        transformers.append(("cat", categorical_pipe, categorical_cols))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.3,
    )

    clf = OneVsRestClassifier(
        LogisticRegression(
            max_iter=300,
            C=C,
            solver="saga",
            n_jobs=1,
            random_state=42,
        ),
        n_jobs=-1,
    )

    return Pipeline(steps=[("prep", preprocessor), ("clf", clf)])


def build_classifier_chain_pipeline(
    X_layout: pd.DataFrame | None = None,
    *,
    max_features: int = 20_000,
    order: str | list[int] | None = None,
    C: float = 4.0,
) -> Pipeline:
    """
    Classifier chain (captures label correlations) with the same feature preprocessor.

    ``order`` follows sklearn ``ClassifierChain`` semantics (``None`` = random).
    """
    base = build_sklearn_multilabel_pipeline(
        X_layout=X_layout,
        max_features=max_features,
        C=C,
    )
    prep = base.named_steps["prep"]
    chain = ClassifierChain(
        LogisticRegression(
            max_iter=300,
            C=C,
            solver="saga",
            n_jobs=1,
            random_state=42,
        ),
        order=order,
        random_state=42,
    )
    return Pipeline(steps=[("prep", prep), ("clf", chain)])


def run_cross_validation(
    model: Pipeline,
    X_train: pd.DataFrame,
    Y_train: np.ndarray,
    *,
    cv_splits: int = 3,
    random_state: int = 42,
    n_jobs: int = -1,
) -> dict[str, Any]:
    """
    K-fold CV on the **training** portion only. Uses custom multi-label F1 micro scorer.
    """
    if cv_splits < 2:
        raise ValueError("cv_splits must be >= 2")

    def f1_micro_multilabel(estimator, X, y):
        y_pred = estimator.predict(X)
        return multilabel_metric_bundle(y, y_pred)["f1_micro"]

    scorer = make_scorer(f1_micro_multilabel, needs_proba=False)

    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    _ensure_text_column(X_train)

    results = cross_validate(
        model,
        X_train,
        Y_train,
        cv=cv,
        scoring={"f1_micro": scorer},
        n_jobs=n_jobs,
        return_train_score=False,
    )
    return {
        "f1_micro_mean": float(np.mean(results["test_f1_micro"])),
        "f1_micro_std": float(np.std(results["test_f1_micro"])),
        "fold_scores": results["test_f1_micro"].tolist(),
    }
