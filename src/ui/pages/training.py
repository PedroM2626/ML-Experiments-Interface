"""
Training page — live training dashboard with Progress Map, Relationship Map,
real-time log console, and Pipeline Leaderboard.
"""

import time
import queue
import streamlit as st
import pandas as pd

from src.utils import state
from src.automl.engine import AutoMLEngine, EngineConfig, EventType
from src.ui.components.progress_map import build_progress_map
from src.ui.components.relationship_map import build_relationship_map
from src.ui.components.leaderboard import render_leaderboard
from src.ui.components.pipeline_detail import render_pipeline_detail
from src.tracking.mlflow_tracker import setup_mlflow, log_pipeline_run, log_experiment_summary


def render():
    state.init_state()

    df = state.get("df")
    if df is None:
        st.warning("⚠️ No dataset loaded. Please go back to the upload page.")
        if st.button("← Back to Upload"):
            state.set("current_page", "upload")
            st.rerun()
        return

    # ── Launch training thread on first render ─────────────────────────────────
    if not state.get("is_training") and not state.get("training_done"):
        _start_training(df)

    # ── Process all available events from queue ────────────────────────────────
    if state.get("is_training"):
        _process_events()

    # ── Header ─────────────────────────────────────────────────────────────────
    _render_header()

    # ── Main visualization area ────────────────────────────────────────────────
    map_col, rel_col = st.columns([1.5, 1], gap="medium")

    with map_col:
        st.markdown("#### 🗺️ Progress Map")
        if state.get("target_column"):
            st.markdown(
                f"<p style='color:#64748b;font-size:12px;margin-top:-8px;margin-bottom:8px;'>"
                f"Prediction column: <strong style='color:#a78bfa;'>{state.get('target_column')}</strong></p>",
                unsafe_allow_html=True,
            )
        fig_map = build_progress_map(
            completed_stages=state.get("completed_stages", []),
            pipelines=state.get("pipelines_state", []),
            running_pipeline_id=_get_running_pid(),
        )
        st.plotly_chart(fig_map, use_container_width=True, key="progress_map_chart")

    with rel_col:
        st.markdown("#### 🌀 Relationship Map")
        fig_rel = build_relationship_map(pipelines=state.get("pipelines_state", []))
        st.plotly_chart(fig_rel, use_container_width=True, key="relationship_map_chart")

        # Status badge
        _render_status_badge()

    st.divider()

    # ── Pipeline Leaderboard ───────────────────────────────────────────────────
    st.markdown("#### 📊 Pipeline Leaderboard")
    new_selected = render_leaderboard(
        results=state.get("results", []),
        optimization_metric=state.get("optimization_metric", "roc_auc"),
        task_type=state.get("task_type", "classification"),
        selected_pipeline_id=state.get("selected_pipeline_id"),
    )
    if new_selected != state.get("selected_pipeline_id"):
        state.set("selected_pipeline_id", new_selected)

    # ── Pipeline Detail ────────────────────────────────────────────────────────
    selected_id = state.get("selected_pipeline_id")
    if selected_id:
        st.divider()
        st.markdown(f"#### 🔍 Pipeline Detail — {selected_id}")
        result_map = {r["pipeline_id"]: r for r in state.get("results", [])}
        if selected_id in result_map:
            render_pipeline_detail(result_map[selected_id], task_type=state.get("task_type", "classification"))

    # ── Log console ─────────────────────────────────────────────────────────────
    st.divider()
    with st.expander("📋 Training Logs", expanded=False):
        logs = state.get("logs", [])
        log_text = "\n".join(reversed(logs[-100:]))
        st.code(log_text, language=None)

    # ── Footer buttons ──────────────────────────────────────────────────────────
    f_col1, f_col2, f_col3 = st.columns([1, 1, 1])
    with f_col1:
        if st.button("← New Experiment", use_container_width=True):
            _stop_training()
            state.set("current_page", "upload")
            state.reset_training_state()
            st.rerun()
    with f_col2:
        if state.get("training_done"):
            mlflow_uri = state.get("mlflow_tracking_uri", "mlruns")
            st.markdown(
                f"<a href='{mlflow_uri}' target='_blank'>"
                f"<button style='width:100%;background:#7c3aed;color:white;border:none;border-radius:6px;padding:8px;cursor:pointer;'>📦 Open MLflow</button>"
                f"</a>",
                unsafe_allow_html=True,
            )
    with f_col3:
        if state.get("training_done"):
            if st.button("📈 View Full Results", use_container_width=True, type="primary"):
                state.set("current_page", "results")
                st.rerun()

    # ── Auto-refresh while training ─────────────────────────────────────────────
    if state.get("is_training"):
        time.sleep(1.0)
        st.rerun()


# ── Helper functions ───────────────────────────────────────────────────────────

def _start_training(df):
    """Initialize and launch the AutoML engine."""
    mlflow_uri, exp_name = setup_mlflow(
        tracking_uri=state.get("mlflow_tracking_uri"),
        experiment_name=state.get("mlflow_experiment_name"),
    )

    config = EngineConfig(
        task_type=state.get("task_type", "classification"),
        target_column=state.get("target_column"),
        test_size=state.get("test_size", 0.1),
        n_folds=state.get("n_folds", 3),
        n_estimators_per_algo=state.get("n_estimators_per_algo", 2),
        max_algorithms=state.get("max_algorithms", 4),
        hpo_trials=state.get("hpo_trials", 10),
        optimization_metric=state.get("optimization_metric", "roc_auc"),
        export_dir="exports",
    )

    engine = AutoMLEngine(config)
    event_q = queue.Queue()
    thread = engine.run_async(df, event_q)

    state.set("engine", engine)
    state.set("event_queue", event_q)
    state.set("training_thread", thread)
    state.set("is_training", True)
    state.set("training_done", False)
    state.set("elapsed_start", time.time())
    state.set("completed_stages", [])
    state.set("pipelines_state", [])
    state.set("results", [])
    state.set("logs", [])

    state.append_log("🚀 AutoML experiment started...")


def _process_events():
    """Drain the event queue and update session state."""
    q = state.get("event_queue")
    if q is None:
        return

    try:
        while True:
            evt = q.get_nowait()
            _handle_event(evt)
    except queue.Empty:
        pass

    # Check if thread is done
    thread = state.get("training_thread")
    if thread is not None and not thread.is_alive():
        state.set("is_training", False)
        state.set("training_done", True)


def _handle_event(evt: dict):
    etype = evt.get("type")

    if etype == EventType.STAGE:
        stage = evt.get("stage", "")
        stages = state.get("completed_stages", [])
        if stage not in stages:
            stages.append(stage)
        state.set("completed_stages", stages)
        state.set("current_stage", stage)

    elif etype == EventType.PIPELINE_START:
        pid = evt.get("pipeline_id")
        pipelines = state.get("pipelines_state", [])
        existing = [p for p in pipelines if p["pipeline_id"] == pid]
        if not existing:
            pipelines.append({
                "pipeline_id": pid,
                "algorithm": evt.get("algorithm", ""),
                "color": evt.get("color", "#8B5CF6"),
                "nodes_done": 0,
            })
        state.set("pipelines_state", pipelines)

    elif etype == EventType.HPO_PROGRESS:
        pid = evt.get("pipeline_id")
        _update_pipeline_nodes(pid, 2)  # HPO running = node 2

    elif etype == EventType.PIPELINE_DONE:
        pid = evt.get("pipeline_id")
        _update_pipeline_nodes(pid, 4)  # All 4 steps done
        result_dict = evt.get("result")
        if result_dict:
            results = state.get("results", [])
            # Replace if already exists, else append
            idx = next((i for i, r in enumerate(results) if r["pipeline_id"] == pid), None)
            if idx is not None:
                results[idx] = result_dict
            else:
                results.append(result_dict)
            state.set("results", results)
            # Log to MLflow
            try:
                log_pipeline_run(
                    result_dict,
                    dataset_name=state.get("dataset_name", ""),
                    task_type=state.get("task_type", "classification"),
                    n_rows=state.get("df").shape[0] if state.get("df") is not None else 0,
                    n_features=state.get("df").shape[1] - 1 if state.get("df") is not None else 0,
                    n_folds=state.get("n_folds", 3),
                    optimization_metric=state.get("optimization_metric", "roc_auc"),
                    run_name=f"{pid} — {result_dict.get('algorithm', '')}",
                )
            except Exception:
                pass

    elif etype == EventType.PIPELINE_FAILED:
        pid = evt.get("pipeline_id")
        _update_pipeline_nodes(pid, 0)

    elif etype == EventType.LOG:
        state.append_log(evt.get("message", ""))

    elif etype == EventType.DONE:
        state.set("best_result", evt.get("best"))
        # Log experiment summary to MLflow
        try:
            elapsed = time.time() - (state.get("elapsed_start") or time.time())
            log_experiment_summary(
                results=state.get("results", []),
                dataset_name=state.get("dataset_name", ""),
                task_type=state.get("task_type", "classification"),
                elapsed_time=elapsed,
            )
        except Exception:
            pass

    elif etype == EventType.ERROR:
        state.append_log(f"💥 ERROR: {evt.get('error', '')}")


def _update_pipeline_nodes(pipeline_id: str, nodes_done: int):
    pipelines = state.get("pipelines_state", [])
    for p in pipelines:
        if p["pipeline_id"] == pipeline_id:
            p["nodes_done"] = nodes_done
    state.set("pipelines_state", pipelines)


def _get_running_pid() -> str:
    """Return pipeline_id of currently running pipeline (nodes_done < 4)."""
    for p in reversed(state.get("pipelines_state", [])):
        if 0 < p.get("nodes_done", 0) < 4:
            return p["pipeline_id"]
    return None


def _render_header():
    is_training = state.get("is_training")
    is_done = state.get("training_done")
    dataset = state.get("dataset_name", "")
    elapsed_start = state.get("elapsed_start")

    elapsed_str = ""
    if elapsed_start:
        elapsed = int(time.time() - elapsed_start)
        elapsed_str = f"⏱ {elapsed // 60:02d}:{elapsed % 60:02d}"

    if is_training:
        status_html = '<span style="background:#1e3a5f;color:#60a5fa;border-radius:6px;padding:4px 12px;font-size:13px;font-weight:600;">⚡ Experiment running…</span>'
    elif is_done:
        n_done = len(state.get("results", []))
        status_html = f'<span style="background:#064e3b;color:#10B981;border-radius:6px;padding:4px 12px;font-size:13px;font-weight:600;">✅ Experiment complete — {n_done} pipelines</span>'
    else:
        status_html = '<span style="background:#1e293b;color:#94a3b8;border-radius:6px;padding:4px 12px;font-size:13px;">⏳ Initializing…</span>'

    st.markdown(
        f"<div style='display:flex;align-items:center;justify-content:space-between;margin-bottom:8px;'>"
        f"<div><h2 style='margin:0;font-size:22px;font-weight:800;color:#e2e8f0;'>🤖 AutoML Experiment</h2>"
        f"<p style='margin:0;font-size:12px;color:#64748b;'>{dataset}</p></div>"
        f"<div style='display:flex;align-items:center;gap:12px;'>{status_html}"
        f"<span style='color:#64748b;font-size:13px;'>{elapsed_str}</span></div>"
        f"</div>",
        unsafe_allow_html=True,
    )


def _render_status_badge():
    is_done = state.get("training_done")
    n_done = len(state.get("results", []))

    if is_done and n_done > 0:
        best = state.get("best_result") or (state.get("results", [{}])[0] if state.get("results") else {})
        best_algo = best.get("algorithm", "")
        best_score = best.get("primary_metric_cv", 0)
        opt_metric = state.get("optimization_metric", "roc_auc")
        elapsed = int(time.time() - (state.get("elapsed_start") or time.time()))

        st.markdown(f"""
        <div style='background:#064e3b;border:1px solid #10B981;border-radius:12px;padding:16px;margin-top:8px;'>
            <div style='color:#10B981;font-weight:700;font-size:15px;margin-bottom:8px;'>✅ Experiment complete</div>
            <div style='color:#94a3b8;font-size:12px;'>{n_done} pipelines generated</div>
            <div style='color:#e2e8f0;font-size:13px;margin-top:6px;'>
                🏆 <strong>{best.get("pipeline_id","")}</strong> — {best_algo}
            </div>
            <div style='color:#a78bfa;font-size:12px;'>{opt_metric.upper()}: {best_score:.4f}</div>
            <div style='color:#64748b;font-size:11px;margin-top:6px;'>
                ⏱ Time elapsed: {elapsed//60:02d}min {elapsed%60:02d}s
            </div>
        </div>
        """, unsafe_allow_html=True)
    elif state.get("is_training"):
        n_done = len(state.get("results", []))
        st.markdown(f"""
        <div style='background:#1e1b4b;border:1px solid #6366F1;border-radius:12px;padding:16px;margin-top:8px;'>
            <div style='color:#a78bfa;font-weight:700;font-size:14px;'>⚡ Training in progress</div>
            <div style='color:#64748b;font-size:12px;margin-top:4px;'>{n_done} pipelines completed so far</div>
        </div>
        """, unsafe_allow_html=True)
