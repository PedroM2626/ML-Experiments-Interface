"""
Experiment Manager — multi-experiment state store.
Each experiment runs in its own thread with an isolated event queue.
Inspired by Azure ML Jobs, SageMaker Canvas, and Vertex AI AutoML.
"""

import queue
import time
import uuid
import threading
import streamlit as st
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import json
from ..db.experiment_store import upsert_experiment, load_all_experiments, delete_experiment_db


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


# ── Key used in st.session_state ─────────────────────────────────────────────
_STATE_KEY = "_automl_experiments"


def _get_store() -> Dict[str, Dict]:
    """Return the global experiment store from session state, initialized from DB if empty."""
    if _STATE_KEY not in st.session_state:
        # Load from SQLite
        db_exps = load_all_experiments()
        st.session_state[_STATE_KEY] = {e["id"]: e for e in db_exps}
    return st.session_state[_STATE_KEY]


# ── Experiment status constants ────────────────────────────────────────────────
class ExpStatus:
    PENDING  = "pending"
    RUNNING  = "running"
    DONE     = "done"
    FAILED   = "failed"
    STOPPED  = "stopped"


# ── Create / manage experiments ───────────────────────────────────────────────

def create_experiment(
    name: str,
    df,
    config,                         # EngineConfig or TSEngineConfig
    dataset_name: str = "",
    task_type: str = "classification",
    target_column: str = "",
    optimization_metric: str = "roc_auc",
) -> str:
    """Create a new experiment entry and return its ID."""
    store = _get_store()
    exp_id = f"exp_{uuid.uuid4().hex[:8]}"
    exp = {
        "id": exp_id,
        "name": name,
        "status": ExpStatus.PENDING,
        "task_type": task_type,
        "dataset_name": dataset_name,
        "target_column": target_column,
        "optimization_metric": optimization_metric,
        "config": config,
        "df": df,
        # Runtime
        "engine": None,
        "event_queue": None,
        "thread": None,
        # Progress
        "completed_stages": [],
        "current_stage": "",
        "pipelines_state": [],
        "results": [],
        "logs": [],
        "best_result": None,
        "elapsed_start": None,
        # Metadata
        "created_at": _now_iso(),
        "finished_at": None,
        "mlflow_run_id": None,
        "n_pipelines_done": 0,
        "error": None,
    }
    store[exp_id] = exp
    try:
        upsert_experiment(exp)
    except Exception:
        pass
    return exp_id


def start_experiment(exp_id: str):
    """Launch the AutoML engine for this experiment in a background thread."""
    from src.automl.engine import AutoMLEngine, EventType

    store = _get_store()
    exp = store.get(exp_id)
    if exp is None:
        raise ValueError(f"Experiment {exp_id} not found")

    config = exp["config"]
    df = exp["df"]

    engine = AutoMLEngine(config)
    event_q = queue.Queue()
    thread = engine.run_async(df, event_q)

    exp["engine"] = engine
    exp["event_queue"] = event_q
    exp["thread"] = thread
    exp["status"] = ExpStatus.RUNNING
    exp["elapsed_start"] = time.time()
    exp["completed_stages"] = []
    exp["pipelines_state"] = []
    exp["results"] = []
    exp["logs"] = [f"[{time.strftime('%H:%M:%S')}] 🚀 Experiment started: {exp['name']}"]


def start_ts_experiment(exp_id: str):
    """Launch the Time Series engine for this experiment."""
    from src.automl.ts_engine import TimeSeriesEngine

    store = _get_store()
    exp = store.get(exp_id)
    if exp is None:
        raise ValueError(f"Experiment {exp_id} not found")

    config = exp["config"]
    df = exp["df"]

    engine = TimeSeriesEngine(config)
    event_q = queue.Queue()
    thread = engine.run_async(df, event_q)

    exp["engine"] = engine
    exp["event_queue"] = event_q
    exp["thread"] = thread
    exp["status"] = ExpStatus.RUNNING
    exp["elapsed_start"] = time.time()
    exp["completed_stages"] = []
    exp["pipelines_state"] = []
    exp["results"] = []
    exp["logs"] = [f"[{time.strftime('%H:%M:%S')}] 🚀 TS Experiment started: {exp['name']}"]


def stop_experiment(exp_id: str):
    """Signal the engine to stop."""
    store = _get_store()
    exp = store.get(exp_id)
    if exp and exp.get("engine"):
        exp["engine"].stop()
    if exp:
        exp["status"] = ExpStatus.STOPPED
        exp["finished_at"] = _now_iso()
        try:
            upsert_experiment(exp)
        except Exception:
            pass


def delete_experiment(exp_id: str):
    """Remove experiment from store and DB."""
    store = _get_store()
    store.pop(exp_id, None)
    try:
        delete_experiment_db(exp_id)
    except Exception:
        pass


def get_experiment(exp_id: str) -> Optional[Dict]:
    return _get_store().get(exp_id)


def list_experiments(sort_by: str = "created_at", reverse: bool = True) -> List[Dict]:
    store = _get_store()
    return sorted(store.values(), key=lambda e: e.get(sort_by, ""), reverse=reverse)


def process_events(exp_id: str):
    """Drain the event queue for this experiment and update its state."""
    from src.automl.engine import EventType

    store = _get_store()
    exp = store.get(exp_id)
    if exp is None:
        return

    q = exp.get("event_queue")
    if q is None:
        return

    try:
        while True:
            evt = q.get_nowait()
            _handle_event(exp, evt)
    except queue.Empty:
        pass

    # Check if thread finished
    thread = exp.get("thread")
    if thread is not None and not thread.is_alive():
        if exp["status"] == ExpStatus.RUNNING:
            exp["status"] = ExpStatus.DONE
            exp["finished_at"] = _now_iso()


def _handle_event(exp: Dict, evt: Dict):
    """Update experiment state based on an engine event."""
    from src.automl.engine import EventType

    etype = evt.get("type")

    if etype == EventType.STAGE:
        stage = evt.get("stage", "")
        if stage not in exp["completed_stages"]:
            exp["completed_stages"].append(stage)
        exp["current_stage"] = stage

    elif etype == EventType.PIPELINE_START:
        pid = evt.get("pipeline_id")
        if not any(p["pipeline_id"] == pid for p in exp["pipelines_state"]):
            exp["pipelines_state"].append({
                "pipeline_id": pid,
                "algorithm": evt.get("algorithm", ""),
                "color": evt.get("color", "#8B5CF6"),
                "nodes_done": 0,
            })

    elif etype == EventType.HPO_PROGRESS:
        pid = evt.get("pipeline_id")
        _update_nodes(exp, pid, 2)

    elif etype == EventType.PIPELINE_DONE:
        pid = evt.get("pipeline_id")
        _update_nodes(exp, pid, 4)
        result_dict = evt.get("result")
        if result_dict:
            results = exp["results"]
            idx = next((i for i, r in enumerate(results) if r["pipeline_id"] == pid), None)
            if idx is not None:
                results[idx] = result_dict
            else:
                results.append(result_dict)
            exp["n_pipelines_done"] = len(results)
            # Update best
            if results:
                best = max(results, key=lambda r: r.get("primary_metric_cv", 0))
                exp["best_result"] = best
            # MLflow logging
            try:
                from src.tracking.mlflow_tracker import log_pipeline_run
                df = exp.get("df")
                log_pipeline_run(
                    result_dict,
                    dataset_name=exp["dataset_name"],
                    task_type=exp["task_type"],
                    n_rows=df.shape[0] if df is not None else 0,
                    n_features=df.shape[1] - 1 if df is not None else 0,
                    n_folds=getattr(exp["config"], "n_folds", 3),
                    optimization_metric=exp["optimization_metric"],
                    run_name=f"{pid} — {result_dict.get('algorithm', '')} [{exp['name']}]",
                )
            except Exception:
                pass

    elif etype == EventType.PIPELINE_FAILED:
        pid = evt.get("pipeline_id")
        _update_nodes(exp, pid, 0)

    elif etype == EventType.LOG:
        msg = evt.get("message", "")
        ts = time.strftime("%H:%M:%S")
        exp["logs"].append(f"[{ts}] {msg}")
        if len(exp["logs"]) > 300:
            exp["logs"] = exp["logs"][-300:]

    elif etype == EventType.DONE:
        exp["status"] = ExpStatus.DONE
        exp["finished_at"] = _now_iso()
        results = exp.get("results", [])
        if results:
            exp["best_result"] = max(results, key=lambda r: r.get("primary_metric_cv", 0))
        try:
            from src.tracking.mlflow_tracker import log_experiment_summary
            elapsed = time.time() - (exp["elapsed_start"] or time.time())
            log_experiment_summary(
                results=results,
                dataset_name=exp["dataset_name"],
                task_type=exp["task_type"],
                elapsed_time=elapsed,
            )
        except Exception:
            pass

    elif etype == EventType.ERROR:
        exp["status"] = ExpStatus.FAILED
        exp["error"] = evt.get("error", "Unknown error")
        exp["finished_at"] = _now_iso()
        ts = time.strftime("%H:%M:%S")
        exp["logs"].append(f"[{ts}] 💥 ERROR: {exp['error']}")
    
    # Persist to DB periodically or on completion
    if etype in (EventType.DONE, EventType.ERROR, EventType.PIPELINE_DONE, EventType.STAGE):
        try:
            # We don't save the full 'df' to SQLite (too big), only metadata
            # experiment_store.py handles skipping large fields usually
            upsert_experiment(exp)
        except Exception:
            pass


def _update_nodes(exp: Dict, pipeline_id: str, nodes_done: int):
    for p in exp["pipelines_state"]:
        if p["pipeline_id"] == pipeline_id:
            p["nodes_done"] = nodes_done


# ── Utility helpers ────────────────────────────────────────────────────────────

def count_running() -> int:
    return sum(1 for e in _get_store().values() if e["status"] == ExpStatus.RUNNING)


def get_active_experiment_id() -> Optional[str]:
    """Return the most recently created running experiment, or last done."""
    store = _get_store()
    running = [e for e in store.values() if e["status"] == ExpStatus.RUNNING]
    if running:
        return sorted(running, key=lambda e: e["created_at"], reverse=True)[0]["id"]
    done = [e for e in store.values() if e["status"] == ExpStatus.DONE]
    if done:
        return sorted(done, key=lambda e: e.get("finished_at", ""), reverse=True)[0]["id"]
    return None
