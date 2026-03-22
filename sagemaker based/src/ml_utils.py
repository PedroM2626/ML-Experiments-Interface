"""
ML Utilities – metrics computation and model evaluation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn import metrics as sk_metrics


def compute_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    problem_type: str,
    y_proba: pd.Series | np.ndarray | None = None,
) -> dict[str, float]:
    """
    Compute evaluation metrics for classification or regression.
    Returns a dict with relevant metric names and float values.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    result: dict[str, float] = {}

    if problem_type in ("binary", "multiclass"):
        result["accuracy"] = float(sk_metrics.accuracy_score(y_true, y_pred))
        result["f1_macro"] = float(sk_metrics.f1_score(y_true, y_pred, average="macro", zero_division=0))
        result["precision_macro"] = float(sk_metrics.precision_score(y_true, y_pred, average="macro", zero_division=0))
        result["recall_macro"] = float(sk_metrics.recall_score(y_true, y_pred, average="macro", zero_division=0))

        if problem_type == "binary" and y_proba is not None:
            y_proba = np.array(y_proba)
            if y_proba.ndim == 2:
                y_proba = y_proba[:, 1]
            try:
                result["roc_auc"] = float(sk_metrics.roc_auc_score(y_true, y_proba))
                result["avg_precision"] = float(sk_metrics.average_precision_score(y_true, y_proba))
            except Exception:
                pass

    elif problem_type == "regression":
        result["rmse"] = float(np.sqrt(sk_metrics.mean_squared_error(y_true, y_pred)))
        result["mae"] = float(sk_metrics.mean_absolute_error(y_true, y_pred))
        result["r2"] = float(sk_metrics.r2_score(y_true, y_pred))
        result["mape"] = float(
            np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-9))) * 100
        )

    return result


def get_confusion_matrix(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    """Return confusion matrix array and label list."""
    labels = sorted(list(set(list(y_true) + list(y_pred))))
    cm = sk_metrics.confusion_matrix(y_true, y_pred, labels=labels)
    return cm, [str(l) for l in labels]


def get_roc_curve(
    y_true: pd.Series | np.ndarray,
    y_proba: pd.Series | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (fpr, tpr, auc) for binary classification."""
    y_proba = np.array(y_proba)
    if y_proba.ndim == 2:
        y_proba = y_proba[:, 1]
    fpr, tpr, _ = sk_metrics.roc_curve(y_true, y_proba)
    auc = sk_metrics.roc_auc_score(y_true, y_proba)
    return fpr.tolist(), tpr.tolist(), float(auc)


def get_precision_recall(
    y_true: pd.Series | np.ndarray,
    y_proba: pd.Series | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (precision, recall, avg_precision) arrays."""
    y_proba = np.array(y_proba)
    if y_proba.ndim == 2:
        y_proba = y_proba[:, 1]
    prec, rec, _ = sk_metrics.precision_recall_curve(y_true, y_proba)
    ap = float(sk_metrics.average_precision_score(y_true, y_proba))
    return prec.tolist(), rec.tolist(), ap


def format_metric_display(metric_name: str, value: float, problem_type: str) -> str:
    """Format a metric value as a pretty string."""
    pct_metrics = {"accuracy", "roc_auc", "avg_precision", "f1_macro", "precision_macro", "recall_macro"}
    if metric_name in pct_metrics:
        return f"{value * 100:.1f}%"
    elif metric_name in ("rmse", "mae"):
        return f"{value:.4f}"
    elif metric_name == "r2":
        return f"{value:.4f}"
    elif metric_name == "mape":
        return f"{value:.2f}%"
    return f"{value:.4f}"


def primary_metric(problem_type: str) -> str:
    """Return the primary display metric for a given problem type."""
    return {
        "binary": "roc_auc",
        "multiclass": "accuracy",
        "regression": "rmse",
    }.get(problem_type, "accuracy")


def primary_metric_label(problem_type: str) -> str:
    """Human-friendly label for the primary metric."""
    return {
        "binary": "ROC AUC",
        "multiclass": "Accuracy",
        "regression": "RMSE",
    }.get(problem_type, "Score")
