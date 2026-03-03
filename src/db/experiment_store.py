"""
SQLite persistence layer for AutoML Studio experiments.
Survives Streamlit page refreshes — experiments and pipeline results are
stored in `experiments.db` at the project root.
"""

import json
import sqlite3
import os
import threading
from datetime import datetime
from typing import List, Dict, Optional

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "experiments.db")

_lock = threading.Lock()


# ── Schema ─────────────────────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    id              TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'pending',
    task_type       TEXT NOT NULL DEFAULT 'classification',
    dataset_name    TEXT,
    target_column   TEXT,
    optimization_metric TEXT,
    created_at      TEXT,
    finished_at     TEXT,
    error           TEXT,
    config_json     TEXT,
    n_pipelines_done INTEGER DEFAULT 0,
    best_result_json TEXT
);

CREATE TABLE IF NOT EXISTS pipelines (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id   TEXT NOT NULL,
    pipeline_id     TEXT NOT NULL,
    algorithm       TEXT,
    transformer     TEXT,
    cv_scores_json  TEXT,
    holdout_scores_json TEXT,
    primary_metric_cv   REAL,
    primary_metric_holdout REAL,
    build_time      REAL,
    hyperparams_json TEXT,
    created_at      TEXT,
    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
);
"""


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables if they don't exist."""
    with _lock:
        conn = _get_conn()
        try:
            conn.executescript(_SCHEMA)
            conn.commit()
        finally:
            conn.close()


# ── Experiments ────────────────────────────────────────────────────────────────

def upsert_experiment(exp: Dict):
    """Insert or replace experiment record."""
    with _lock:
        conn = _get_conn()
        try:
            config = exp.get("config")
            config_json = None
            if config is not None:
                try:
                    config_json = json.dumps(config.__dict__)
                except Exception:
                    config_json = str(config)

            best = exp.get("best_result")
            conn.execute("""
                INSERT OR REPLACE INTO experiments
                (id, name, status, task_type, dataset_name, target_column,
                 optimization_metric, created_at, finished_at, error,
                 config_json, n_pipelines_done, best_result_json)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                exp["id"], exp.get("name", ""), exp.get("status", "pending"),
                exp.get("task_type", "classification"),
                exp.get("dataset_name", ""), exp.get("target_column", ""),
                exp.get("optimization_metric", "roc_auc"),
                exp.get("created_at", datetime.now().isoformat()),
                exp.get("finished_at"),
                exp.get("error"),
                config_json,
                exp.get("n_pipelines_done", 0),
                json.dumps(best) if best else None,
            ))
            conn.commit()
        finally:
            conn.close()


def update_experiment_status(exp_id: str, status: str,
                              finished_at: Optional[str] = None,
                              error: Optional[str] = None,
                              n_pipelines_done: Optional[int] = None,
                              best_result: Optional[Dict] = None):
    """Update mutable fields of an experiment."""
    with _lock:
        conn = _get_conn()
        try:
            updates = ["status = ?"]
            vals = [status]
            if finished_at is not None:
                updates.append("finished_at = ?")
                vals.append(finished_at)
            if error is not None:
                updates.append("error = ?")
                vals.append(error)
            if n_pipelines_done is not None:
                updates.append("n_pipelines_done = ?")
                vals.append(n_pipelines_done)
            if best_result is not None:
                updates.append("best_result_json = ?")
                vals.append(json.dumps(best_result))
            vals.append(exp_id)
            conn.execute(f"UPDATE experiments SET {', '.join(updates)} WHERE id = ?", vals)
            conn.commit()
        finally:
            conn.close()


def save_pipeline_result(exp_id: str, result: Dict):
    """Save a single pipeline result row."""
    with _lock:
        conn = _get_conn()
        try:
            conn.execute("""
                INSERT INTO pipelines
                (experiment_id, pipeline_id, algorithm, transformer,
                 cv_scores_json, holdout_scores_json,
                 primary_metric_cv, primary_metric_holdout,
                 build_time, hyperparams_json, created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """, (
                exp_id,
                result.get("pipeline_id", ""),
                result.get("algorithm", ""),
                result.get("transformer", ""),
                json.dumps(result.get("cv_scores", {})),
                json.dumps(result.get("holdout_scores", {})),
                float(result.get("primary_metric_cv", 0.0)),
                float(result.get("primary_metric_holdout", 0.0)),
                float(result.get("build_time", 0.0)),
                json.dumps(result.get("hyperparams", {})),
                datetime.now().isoformat(),
            ))
            conn.commit()
        finally:
            conn.close()


def delete_experiment_db(exp_id: str):
    """Delete experiment and its pipelines."""
    with _lock:
        conn = _get_conn()
        try:
            conn.execute("DELETE FROM pipelines WHERE experiment_id = ?", (exp_id,))
            conn.execute("DELETE FROM experiments WHERE id = ?", (exp_id,))
            conn.commit()
        finally:
            conn.close()


# ── Load / Restore ─────────────────────────────────────────────────────────────

def load_all_experiments() -> List[Dict]:
    """Load all experiment records from DB (no pipelines, no heavy objects)."""
    init_db()
    with _lock:
        conn = _get_conn()
        try:
            rows = conn.execute(
                "SELECT * FROM experiments ORDER BY created_at DESC"
            ).fetchall()
            result = []
            for row in rows:
                d = dict(row)
                d["best_result"] = json.loads(d.pop("best_result_json") or "null")
                # config_json left as-is (not re-instantiated — not needed for display)
                result.append(d)
            return result
        finally:
            conn.close()


def load_pipelines_for_experiment(exp_id: str) -> List[Dict]:
    """Load pipeline results for a specific experiment."""
    with _lock:
        conn = _get_conn()
        try:
            rows = conn.execute(
                "SELECT * FROM pipelines WHERE experiment_id = ? ORDER BY id",
                (exp_id,)
            ).fetchall()
            results = []
            for row in rows:
                d = dict(row)
                d["cv_scores"] = json.loads(d.pop("cv_scores_json") or "{}")
                d["holdout_scores"] = json.loads(d.pop("holdout_scores_json") or "{}")
                d["hyperparams"] = json.loads(d.pop("hyperparams_json") or "{}")
                results.append(d)
            return results
        finally:
            conn.close()


# Initialise on import
init_db()
