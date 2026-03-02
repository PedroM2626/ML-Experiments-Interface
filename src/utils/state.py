"""
Streamlit session state helpers — centralized state management.
"""

import streamlit as st
import queue
import threading
import time
from typing import Any, Optional


def init_state():
    """Initialize all session state keys with defaults."""
    defaults = {
        # Dataset
        "df": None,
        "dataset_name": "",
        "target_column": None,
        "task_type": "classification",
        # Time Series
        "date_column": "",
        "horizon": 12,
        "freq": "M",
        # Config
        "test_size": 0.1,
        "n_folds": 3,
        "max_algorithms": 4,
        "n_estimators_per_algo": 2,
        "hpo_trials": 10,
        "optimization_metric": "roc_auc",
        # Multi-experiment manager
        "active_experiment_id": None,
        # Legacy single-experiment training state (kept for backward compat)
        "is_training": False,
        "training_done": False,
        "event_queue": None,
        "training_thread": None,
        "engine": None,
        # Progress
        "completed_stages": [],
        "current_stage": "",
        "pipelines_state": [],
        "results": [],
        "logs": [],
        "best_result": None,
        "elapsed_start": None,
        # UI
        "current_page": "upload",
        "selected_pipeline_id": None,
        "mlflow_run_id": None,
        "mlflow_tracking_uri": "mlruns",
        "mlflow_experiment_name": "AutoML_Experiment",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def reset_training_state():
    """Reset all training-related state."""
    keys = [
        "is_training", "training_done", "event_queue", "training_thread", "engine",
        "completed_stages", "current_stage", "pipelines_state", "results",
        "logs", "best_result", "elapsed_start", "selected_pipeline_id",
    ]
    for k in keys:
        st.session_state[k] = None if "queue" in k or "thread" in k or "engine" in k else (
            [] if k in ("completed_stages", "pipelines_state", "results", "logs") else
            False if k in ("is_training", "training_done") else
            "" if k in ("current_stage",) else
            None
        )


def get(key: str, default=None) -> Any:
    return st.session_state.get(key, default)


def set(key: str, value: Any):
    st.session_state[key] = value


def append_log(msg: str):
    if "logs" not in st.session_state:
        st.session_state["logs"] = []
    st.session_state["logs"].append(f"[{time.strftime('%H:%M:%S')}] {msg}")
    # Keep last 200 logs
    if len(st.session_state["logs"]) > 200:
        st.session_state["logs"] = st.session_state["logs"][-200:]


def update_pipeline_node(pipeline_id: str, nodes_done: int):
    """Update the nodes_done for a specific pipeline in pipelines_state."""
    for p in st.session_state.get("pipelines_state", []):
        if p["pipeline_id"] == pipeline_id:
            p["nodes_done"] = nodes_done
            return
    # Not found — skip
