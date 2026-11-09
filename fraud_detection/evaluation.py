"""Ranking metrics and operating-point selection with explicit undefined cases."""

from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _validate_arrays(y_true: ArrayLike, scores: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    labels, values = np.asarray(y_true), np.asarray(scores, dtype=float)
    if labels.ndim != 1 or values.ndim != 1 or not len(labels) or len(labels) != len(values):
        raise ValueError("Labels and scores must be non-empty, equal-length 1D arrays.")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Labels must be binary 0/1.")
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Scores must be finite values in [0, 1].")
    return labels.astype(int), values


def threshold_metrics(y_true: ArrayLike, scores: ArrayLike, threshold: float) -> dict[str, Any]:
    """Undefined precision/recall/F1 use zero; ranking undefined cases use null."""
    labels, values = _validate_arrays(y_true, scores)
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Threshold must be a finite value in [0, 1].")
    predicted = (values >= threshold).astype(int)
    matrix = confusion_matrix(labels, predicted, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "accuracy": float(accuracy_score(labels, predicted)),
        "alert_rate": float(predicted.mean()),
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
    }


def evaluate_scores(y_true: ArrayLike, scores: ArrayLike, threshold: float) -> dict[str, Any]:
    """Report AP separately from trapezoidal PR area; do not conflate the two."""
    labels, values = _validate_arrays(y_true, scores)
    report = threshold_metrics(labels, values, threshold)
    report.update(
        {
            "rows": len(labels),
            "positive_count": int(labels.sum()),
            "negative_count": int(len(labels) - labels.sum()),
            "fraud_rate": float(labels.mean()),
            "average_precision": None,
            "pr_auc_trapezoidal": None,
            "roc_auc": None,
            "ranking_metrics_defined": len(np.unique(labels)) == 2,
        }
    )
    if report["ranking_metrics_defined"]:
        precision, recall, _ = precision_recall_curve(labels, values)
        report.update(
            average_precision=float(average_precision_score(labels, values)),
            pr_auc_trapezoidal=float(auc(recall, precision)),
            roc_auc=float(roc_auc_score(labels, values)),
        )
    return report


def select_threshold(y_validation: ArrayLike, validation_scores: ArrayLike) -> float:
    """Maximize validation F1. Higher thresholds win exact ties.

    The terminal PR point has no threshold and must be excluded. This function
    deliberately accepts no test labels; the caller must pass validation only.
    """
    labels, values = _validate_arrays(y_validation, validation_scores)
    if len(np.unique(labels)) != 2:
        raise ValueError("Threshold selection requires both classes in validation data.")
    precision, recall, thresholds = precision_recall_curve(labels, values)
    denominator = precision[:-1] + recall[:-1]
    f1 = np.divide(
        2 * precision[:-1] * recall[:-1],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    return float(thresholds[np.flatnonzero(f1 == f1.max())[-1]])
