"""
Multi-label support-ticket ML pipeline.

Public entry points:
- build_preprocessed_frame, make_train_val_test
- build_sklearn_multilabel_pipeline, run_cross_validation
- train_multilabel_transformer, predict_multilabel_transformer
- multilabel_metric_bundle
"""

from ml_pipeline.metrics import multilabel_metric_bundle
from ml_pipeline.preprocessing import build_preprocessed_frame, make_train_val_test
from ml_pipeline.features import modeling_feature_columns
from ml_pipeline.sklearn_pipeline import (
    build_classifier_chain_pipeline,
    build_sklearn_multilabel_pipeline,
    run_cross_validation,
)
from ml_pipeline.model_io import (
    binary_matrix_to_label_lists,
    load_sklearn_bundle,
    predict_sklearn_multilabel,
    save_sklearn_bundle,
)
from ml_pipeline.transformer_multilabel import (
    evaluate_multilabel_transformer_on_test,
    load_multilabel_transformer,
    predict_multilabel_transformer,
    predict_multilabel_with_model,
    train_multilabel_transformer,
)

__all__ = [
    "build_preprocessed_frame",
    "make_train_val_test",
    "modeling_feature_columns",
    "build_sklearn_multilabel_pipeline",
    "build_classifier_chain_pipeline",
    "run_cross_validation",
    "save_sklearn_bundle",
    "load_sklearn_bundle",
    "predict_sklearn_multilabel",
    "binary_matrix_to_label_lists",
    "train_multilabel_transformer",
    "load_multilabel_transformer",
    "predict_multilabel_transformer",
    "predict_multilabel_with_model",
    "evaluate_multilabel_transformer_on_test",
    "multilabel_metric_bundle",
]
