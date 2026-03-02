"""
Evaluator — computes all metrics for classification and regression pipelines.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate, StratifiedKFold, KFold
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, precision_score, recall_score,
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings("ignore")


CV_METRICS_CLASSIFICATION = {
    "accuracy": "accuracy",
    "roc_auc": "roc_auc_ovr_weighted",
    "f1": "f1_weighted",
    "precision": "precision_weighted",
    "recall": "recall_weighted",
}

CV_METRICS_REGRESSION = {
    "r2": "r2",
    "neg_mse": "neg_mean_squared_error",
    "neg_mae": "neg_mean_absolute_error",
}

OPTIMIZATION_METRICS = {
    "classification": {
        "ROC AUC": "roc_auc",
        "Accuracy": "accuracy",
        "F1 Score": "f1",
    },
    "regression": {
        "R²": "r2",
        "RMSE": "neg_mse",
        "MAE": "neg_mae",
    },
    "time_series": {
        "RMSE": "rmse",
        "MAE": "mae",
        "MAPE (%)": "mape",
        "R²": "r2",
    },
}


def get_cv_splitter(task_type: str, n_folds: int):
    if task_type == "classification":
        return StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    return KFold(n_splits=n_folds, shuffle=True, random_state=42)


def run_cross_validation(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    task_type: str,
    n_folds: int = 5,
) -> Dict[str, float]:
    """Run cross-validation and return mean metric scores."""
    cv = get_cv_splitter(task_type, n_folds)
    scoring = CV_METRICS_CLASSIFICATION if task_type == "classification" else CV_METRICS_REGRESSION

    try:
        scores = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, error_score="raise")
    except Exception:
        # Fallback: only accuracy/r2
        fallback = "accuracy" if task_type == "classification" else "r2"
        scores = cross_validate(pipeline, X, y, cv=cv, scoring=fallback, error_score=np.nan)
        return {fallback: float(np.nanmean(scores["test_score"]))}

    result = {}
    for metric_name in scoring:
        key = f"test_{metric_name}"
        if key in scores:
            val = float(np.nanmean(scores[key]))
            # MSE / MAE are negated
            if "neg_" in metric_name:
                val = -val
            result[metric_name] = round(val, 6)
    return result


def evaluate_holdout(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    task_type: str,
) -> Dict[str, float]:
    """Evaluate a fitted pipeline on the holdout set."""
    y_pred = pipeline.predict(X_test)
    result = {}

    if task_type == "classification":
        result["accuracy"] = round(accuracy_score(y_test, y_pred), 6)
        result["f1"] = round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 6)
        result["precision"] = round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 6)
        result["recall"] = round(recall_score(y_test, y_pred, average="weighted", zero_division=0), 6)
        try:
            if hasattr(pipeline, "predict_proba"):
                y_proba = pipeline.predict_proba(X_test)
                classes = len(np.unique(y_test))
                if classes == 2:
                    result["roc_auc"] = round(roc_auc_score(y_test, y_proba[:, 1]), 6)
                else:
                    result["roc_auc"] = round(
                        roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted"), 6
                    )
        except Exception:
            pass
    else:
        result["r2"] = round(r2_score(y_test, y_pred), 6)
        result["neg_mse"] = round(mean_squared_error(y_test, y_pred), 6)
        result["rmse"] = round(np.sqrt(mean_squared_error(y_test, y_pred)), 6)
        result["neg_mae"] = round(mean_absolute_error(y_test, y_pred), 6)

    return result


def get_confusion_matrix(pipeline: Pipeline, X_test, y_test) -> Optional[np.ndarray]:
    try:
        y_pred = pipeline.predict(X_test)
        return confusion_matrix(y_test, y_pred)
    except Exception:
        return None


def get_primary_metric(scores: Dict[str, float], task_type: str, metric_key: str) -> float:
    """Extract the primary optimization metric from a score dict."""
    if task_type == "classification":
        return scores.get(metric_key, scores.get("accuracy", 0.0))
    else:
        val = scores.get(metric_key, scores.get("r2", 0.0))
        # For neg metrics, negate back for display
        if metric_key in ("neg_mse", "neg_mae"):
            return -val if val > 0 else val
        return val
