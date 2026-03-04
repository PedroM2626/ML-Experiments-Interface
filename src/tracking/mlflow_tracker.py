"""
MLflow tracking integration — logs all experiment data, metrics, artifacts, and models.
"""

import os
import io
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv

load_dotenv()


def get_tracking_uri() -> str:
    return os.getenv("MLFLOW_TRACKING_URI", "mlruns")


def get_experiment_name() -> str:
    return os.getenv("MLFLOW_EXPERIMENT_NAME", "MLine_Experiment")


def setup_mlflow(tracking_uri: Optional[str] = None, experiment_name: Optional[str] = None):
    """Configure MLflow tracking URI and experiment."""
    uri = tracking_uri or get_tracking_uri()
    name = experiment_name or get_experiment_name()

    mlflow.set_tracking_uri(uri)
    try:
        mlflow.set_experiment(name)
    except Exception:
        pass
    return uri, name


def log_pipeline_run(
    pipeline_result: Dict,
    dataset_name: str,
    task_type: str,
    n_rows: int,
    n_features: int,
    n_folds: int,
    optimization_metric: str,
    pipeline_obj=None,
    run_name: Optional[str] = None,
) -> Optional[str]:
    """Log a single pipeline run to MLflow. Returns run_id."""
    try:
        with mlflow.start_run(run_name=run_name or pipeline_result.get("pipeline_id", "Pipeline")) as run:
            run_id = run.info.run_id

            # Tags
            mlflow.set_tags({
                "pipeline_id": pipeline_result.get("pipeline_id", ""),
                "algorithm": pipeline_result.get("algorithm", ""),
                "transformer": pipeline_result.get("transformer", ""),
                "task_type": task_type,
                "dataset": dataset_name,
            })

            # Dataset params
            mlflow.log_params({
                "dataset_rows": n_rows,
                "dataset_features": n_features,
                "n_folds": n_folds,
                "optimization_metric": optimization_metric,
                "algorithm": pipeline_result.get("algorithm", ""),
                "transformer": pipeline_result.get("transformer", ""),
                "use_pca": pipeline_result.get("use_pca", False),
                "use_select_k": pipeline_result.get("use_select_k", False),
            })

            # Hyperparams
            hparams = pipeline_result.get("hyperparams", {})
            if hparams:
                safe_hp = {k: (str(v) if not isinstance(v, (int, float, bool, str)) else v)
                           for k, v in hparams.items()}
                mlflow.log_params(safe_hp)

            # CV metrics
            cv_scores = pipeline_result.get("cv_scores", {})
            for metric, value in cv_scores.items():
                if value is not None and not np.isnan(float(value)):
                    mlflow.log_metric(f"cv_{metric}", float(value))

            # Holdout metrics
            holdout_scores = pipeline_result.get("holdout_scores", {})
            for metric, value in holdout_scores.items():
                if value is not None and not np.isnan(float(value)):
                    mlflow.log_metric(f"holdout_{metric}", float(value))

            mlflow.log_metric("primary_metric_cv", pipeline_result.get("primary_metric_cv", 0.0))
            mlflow.log_metric("primary_metric_holdout", pipeline_result.get("primary_metric_holdout", 0.0))
            mlflow.log_metric("build_time_seconds", pipeline_result.get("build_time", 0.0))

            # Feature importance chart
            fi_data = pipeline_result.get("feature_importance", [])
            if fi_data:
                fi_df = pd.DataFrame(fi_data).head(20)
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.set_facecolor("#1a1a2e")
                fig.patch.set_facecolor("#1a1a2e")
                ax.barh(fi_df["feature"][::-1], fi_df["importance"][::-1], color="#8B5CF6")
                ax.set_xlabel("Importance", color="white")
                ax.set_title(f"Feature Importance — {pipeline_result.get('pipeline_id', '')}", color="white")
                ax.tick_params(colors="white")
                ax.spines["bottom"].set_color("#444")
                ax.spines["left"].set_color("#444")
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                buf = io.BytesIO()
                plt.tight_layout()
                plt.savefig(buf, format="png", bbox_inches="tight", dpi=100)
                buf.seek(0)
                mlflow.log_figure(fig, "feature_importance.png")
                plt.close(fig)

            # Log model artifact
            if pipeline_obj is not None:
                try:
                    mlflow.sklearn.log_model(pipeline_obj, "model")
                except Exception:
                    pass

        return run_id

    except Exception as e:
        print(f"[MLflow] Warning: Failed to log run — {e}")
        return None


def log_experiment_summary(
    results: List[Dict],
    dataset_name: str,
    task_type: str,
    elapsed_time: float,
) -> Optional[str]:
    """Log a summary run for the entire experiment."""
    try:
        with mlflow.start_run(run_name=f"[Summary] {dataset_name}") as run:
            mlflow.set_tags({
                "run_type": "experiment_summary",
                "dataset": dataset_name,
                "task_type": task_type,
            })
            mlflow.log_params({
                "n_pipelines": len(results),
                "dataset": dataset_name,
                "task_type": task_type,
            })
            if results:
                best = max(results, key=lambda r: r.get("primary_metric_cv", 0))
                mlflow.log_metric("best_cv_score", best.get("primary_metric_cv", 0))
                mlflow.log_metric("best_holdout_score", best.get("primary_metric_holdout", 0))
                mlflow.log_param("best_algorithm", best.get("algorithm", ""))
                mlflow.log_param("best_pipeline_id", best.get("pipeline_id", ""))
            mlflow.log_metric("elapsed_seconds", elapsed_time)
            return run.info.run_id
    except Exception as e:
        print(f"[MLflow] Warning: Failed to log summary — {e}")
        return None


def get_experiment_runs(experiment_name: str) -> pd.DataFrame:
    """Retrieve all runs from the current experiment as a DataFrame."""
    try:
        client = MlflowClient()
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            return pd.DataFrame()
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.primary_metric_cv DESC"],
        )
        records = []
        for r in runs:
            records.append({
                "run_id": r.info.run_id,
                "pipeline_id": r.data.tags.get("pipeline_id", ""),
                "algorithm": r.data.tags.get("algorithm", ""),
                "status": r.info.status,
                **{k: v for k, v in r.data.metrics.items()},
                **{k: v for k, v in r.data.params.items()},
            })
        return pd.DataFrame(records)
    except Exception:
        return pd.DataFrame()
