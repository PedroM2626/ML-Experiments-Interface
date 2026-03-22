"""
MLflow integration utilities for AutoML Studio.
Handles experiment tracking, model logging, and registry.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import pandas as pd

from config import MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME


def setup_mlflow() -> None:
    """Configure the MLflow tracking URI and create experiment if needed."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


def start_run(model_name: str, problem_type: str, build_type: str = "quick") -> mlflow.ActiveRun:
    """Start and return an MLflow run."""
    setup_mlflow()
    run = mlflow.start_run(run_name=model_name)
    mlflow.set_tags({
        "model_name": model_name,
        "problem_type": problem_type,
        "build_type": build_type,
        "framework": "AutoGluon",
    })
    return run


def log_params(params: dict[str, Any]) -> None:
    """Log a dictionary of hyperparameters."""
    for k, v in params.items():
        try:
            mlflow.log_param(k, v)
        except Exception:
            pass


def log_metrics(metrics: dict[str, float]) -> None:
    """Log a dictionary of evaluation metrics."""
    for k, v in metrics.items():
        try:
            mlflow.log_metric(k, float(v))
        except Exception:
            pass


def log_artifact_path(local_path: str | Path) -> None:
    """Log a local file or directory as an MLflow artifact."""
    try:
        mlflow.log_artifact(str(local_path))
    except Exception:
        pass


def log_model_artifact(predictor_path: str | Path, artifact_dir: str = "model") -> None:
    """Log the saved AutoGluon model directory as an artifact."""
    try:
        mlflow.log_artifact(str(predictor_path), artifact_path=artifact_dir)
    except Exception:
        pass


def log_figure(fig, name: str) -> None:
    """Log a Plotly figure as an HTML artifact."""
    try:
        import tempfile
        tmp_dir = Path(tempfile.gettempdir())
        tmp_path = tmp_dir / f"{name}.html"
        fig.write_html(str(tmp_path))
        mlflow.log_artifact(str(tmp_path), artifact_path="charts")
        tmp_path.unlink(missing_ok=True)
    except Exception:
        pass


def end_run() -> None:
    """End the current MLflow run."""
    try:
        mlflow.end_run()
    except Exception:
        pass


def register_model(run_id: str, model_name: str) -> str | None:
    """Register the best model to the MLflow Model Registry."""
    try:
        setup_mlflow()
        result = mlflow.register_model(
            model_uri=f"runs:/{run_id}/model",
            name=model_name,
        )
        return result.version
    except Exception as e:
        print(f"Model registration failed: {e}")
        return None


def get_all_runs() -> pd.DataFrame:
    """Return a DataFrame of all runs in the active experiment."""
    try:
        setup_mlflow()
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
        if experiment is None:
            return pd.DataFrame()
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
        )
        records = []
        for r in runs:
            record = {
                "run_id": r.info.run_id,
                "run_name": r.data.tags.get("mlflow.runName", r.info.run_id[:8]),
                "model_name": r.data.tags.get("model_name", ""),
                "problem_type": r.data.tags.get("problem_type", ""),
                "build_type": r.data.tags.get("build_type", ""),
                "status": r.info.status,
                "start_time": pd.Timestamp(r.info.start_time, unit="ms"),
            }
            record.update(r.data.metrics)
            records.append(record)
        return pd.DataFrame(records)
    except Exception:
        return pd.DataFrame()
