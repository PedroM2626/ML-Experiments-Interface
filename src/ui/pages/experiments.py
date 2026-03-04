"""
Experiments page — multi-experiment manager dashboard.
Inspired by Azure ML Jobs, SageMaker Canvas, Vertex AI AutoML, and WatsonX AutoAI.
Shows all experiments as cards, allows launching new ones, comparing results, and
drilling into individual experiment training dashboards.
"""

import time
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from src.utils import state
from src.utils import experiment_manager as em
from src.utils.experiment_manager import ExpStatus
from src.ui.components.progress_map import build_progress_map
from src.ui.components.relationship_map import build_relationship_map
from src.ui.components.leaderboard import render_leaderboard
from src.ui.components.pipeline_detail import render_pipeline_detail


# ── Status styles ──────────────────────────────────────────────────────────────
STATUS_STYLES = {
    ExpStatus.RUNNING:  ("⚡", "#3b82f6", "#1e3a5f"),
    ExpStatus.DONE:     ("✅", "#10b981", "#064e3b"),
    ExpStatus.FAILED:   ("❌", "#ef4444", "#450a0a"),
    ExpStatus.STOPPED:  ("⏹️", "#94a3b8", "#1e293b"),
    ExpStatus.PENDING:  ("⏳", "#f59e0b", "#422006"),
}

TASK_ICONS = {
    "classification": "🏷️",
    "regression": "📈",
    "time_series": "📅",
}


def render():
    state.init_state()

    # Process events for all running experiments
    for exp in em.list_experiments():
        if exp["status"] == ExpStatus.RUNNING:
            em.process_events(exp["id"])

    experiments = em.list_experiments()
    active_exp_id = state.get("active_experiment_id")

    # ── Header ──────────────────────────────────────────────────────────────────
    n_running = em.count_running()
    n_done = sum(1 for e in experiments if e["status"] == ExpStatus.DONE)

    st.markdown(f"""
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:16px;">
        <div>
            <h2 style="margin:0;font-size:24px;font-weight:800;color:#e2e8f0;">
                🧪 Experiments
            </h2>
            <p style="margin:0;font-size:12px;color:#64748b;">
                {len(experiments)} total &nbsp;·&nbsp;
                <span style="color:#3b82f6;">{n_running} running</span> &nbsp;·&nbsp;
                <span style="color:#10b981;">{n_done} completed</span>
            </p>
        </div>
        <div style="display:flex;gap:8px;align-items:center;">
            <span style="background:#111827;border:1px solid #1e293b;color:#94a3b8;font-size:11px;border-radius:6px;padding:4px 10px;">
                MLine Platform
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── NEW EXPERIMENT BUTTON ──────────────────────────────────────────────────
    if st.button("➕ New Experiment", type="primary", key="new_exp_btn"):
        state.set("current_page", "upload")
        st.rerun()

    st.divider()

    if not experiments:
        st.markdown("""
        <div style='text-align:center;padding:60px 0;color:#475569;'>
            <div style='font-size:48px;margin-bottom:16px;'>🧪</div>
            <div style='font-size:18px;font-weight:600;color:#64748b;'>No experiments yet</div>
            <div style='font-size:13px;margin-top:8px;'>Configure a dataset and click "Start AutoML Training" to begin</div>
        </div>
        """, unsafe_allow_html=True)
        return

    # ── If an active experiment is selected, show its training view ─────────────
    if active_exp_id and active_exp_id in {e["id"] for e in experiments}:
        active_exp = em.get_experiment(active_exp_id)
        if active_exp:
            _render_active_experiment(active_exp)
            st.divider()
            st.markdown("#### 📋 All Experiments")

    # ── Experiment grid cards ──────────────────────────────────────────────────
    _render_experiment_cards(experiments, active_exp_id)

    # ── Auto-refresh while experiments are running ─────────────────────────────
    if n_running > 0:
        time.sleep(1.5)
        st.rerun()


# ── Active Experiment View ─────────────────────────────────────────────────────

def _render_active_experiment(exp: dict):
    status = exp["status"]
    icon, color, bg = STATUS_STYLES.get(status, ("⏳", "#94a3b8", "#1e293b"))
    task_icon = TASK_ICONS.get(exp["task_type"], "🤖")
    elapsed = ""
    if exp.get("elapsed_start"):
        secs = int(time.time() - exp["elapsed_start"])
        elapsed = f"⏱ {secs//60:02d}:{secs%60:02d}"

    n_done = exp.get("n_pipelines_done", 0)
    best = exp.get("best_result")
    opt_metric = exp.get("optimization_metric", "roc_auc")

    # Header card
    st.markdown(f"""
    <div style="background:{bg};border:1px solid {color};border-radius:12px;padding:16px 20px;margin-bottom:12px;">
        <div style="display:flex;justify-content:space-between;align-items:flex-start;">
            <div>
                <div style="font-size:18px;font-weight:800;color:#e2e8f0;">
                    {task_icon} {exp['name']}
                </div>
                <div style="font-size:12px;color:#94a3b8;margin-top:4px;">
                    {exp['dataset_name']} &nbsp;·&nbsp; Target: <span style="color:#a78bfa;">{exp['target_column']}</span>
                    &nbsp;·&nbsp; {exp['task_type'].replace('_', ' ').title()}
                </div>
            </div>
            <div style="text-align:right;">
                <div style="background:{color}22;color:{color};border-radius:6px;padding:4px 12px;font-size:12px;font-weight:600;display:inline-block;">
                    {icon} {status.title()}
                </div>
                <div style="color:#64748b;font-size:11px;margin-top:4px;">{elapsed}</div>
            </div>
        </div>
        {"" if n_done == 0 else f"<div style='color:#94a3b8;font-size:12px;margin-top:8px;'>📊 {n_done} pipelines completed" +
         (f" &nbsp;·&nbsp; 🏆 Best: <span style='color:#10b981;font-weight:700;'>{best.get('pipeline_id','')} ({best.get('algorithm','')}) — {abs(best.get('primary_metric_cv',0)):.4f} {opt_metric.upper()}</span>" if best else "") + "</div>"}
    </div>
    """, unsafe_allow_html=True)

    # Stop button for running experiments
    if status == ExpStatus.RUNNING:
        col_stop, col_view = st.columns([1, 4])
        with col_stop:
            if st.button("⏹ Stop", key=f"stop_{exp['id']}", use_container_width=True):
                em.stop_experiment(exp["id"])
                st.rerun()

    # Progress map + relationship map
    if exp.get("pipelines_state"):
        map_col, rel_col = st.columns([1.5, 1], gap="medium")
        with map_col:
            st.markdown("##### 🗺️ Progress Map")
            running_pid = _get_running_pid(exp)
            fig_map = build_progress_map(
                completed_stages=exp.get("completed_stages", []),
                pipelines=exp.get("pipelines_state", []),
                running_pipeline_id=running_pid,
            )
            st.plotly_chart(fig_map, use_container_width=True, key=f"map_{exp['id']}")

        with rel_col:
            st.markdown("##### 🌀 Relationship Map")
            fig_rel = build_relationship_map(pipelines=exp.get("pipelines_state", []))
            st.plotly_chart(fig_rel, use_container_width=True, key=f"rel_{exp['id']}")

    # ── Time Series forecast chart ─────────────────────────────────────────────
    if exp.get("task_type") == "time_series" and exp.get("engine") and exp["status"] == ExpStatus.DONE:
        engine = exp.get("engine")
        if engine and engine.forecast_df is not None:
            st.markdown("##### 📈 Forecast vs Actuals")
            from src.ui.components.forecast_chart import build_forecast_chart
            fdf = engine.forecast_df
            df_orig = exp.get("df")
            if df_orig is not None:
                y_all = df_orig[exp["target_column"]]
                cfg = exp.get("config")
                horizon = getattr(cfg, "horizon", 12)
                fig_fc = build_forecast_chart(
                    y_actual=y_all,
                    y_forecast=fdf["forecast"].values,
                    y_lower=fdf["lower"].values,
                    y_upper=fdf["upper"].values,
                    target_col=exp["target_column"],
                    horizon=horizon,
                )
                st.plotly_chart(fig_fc, use_container_width=True, key=f"fc_{exp['id']}")

    # Leaderboard
    results = exp.get("results", [])
    if results:
        st.markdown("##### 📊 Pipeline Leaderboard")
        selected = render_leaderboard(
            results=results,
            optimization_metric=exp.get("optimization_metric", "roc_auc"),
            task_type=exp.get("task_type", "classification"),
            selected_pipeline_id=state.get("selected_pipeline_id"),
        )
        if selected != state.get("selected_pipeline_id"):
            state.set("selected_pipeline_id", selected)

        # Pipeline detail
        if state.get("selected_pipeline_id"):
            sel_id = state.get("selected_pipeline_id")
            result_map = {r["pipeline_id"]: r for r in results}
            if sel_id in result_map:
                render_pipeline_detail(result_map[sel_id],
                                       task_type=exp.get("task_type", "classification"))

    # Logs
    logs = exp.get("logs", [])
    if logs:
        with st.expander("📋 Training Logs", expanded=False):
            st.code("\n".join(reversed(logs[-80:])), language=None)


# ── Experiment Cards Grid ──────────────────────────────────────────────────────

def _render_experiment_cards(experiments: list, active_exp_id: str):
    if not experiments:
        return

    # Show in 2-column grid
    cols = st.columns(2, gap="medium")
    for i, exp in enumerate(experiments):
        col = cols[i % 2]
        with col:
            _render_exp_card(exp, is_active=(exp["id"] == active_exp_id))


def _render_exp_card(exp: dict, is_active: bool):
    """Render a single experiment card."""
    status = exp["status"]
    icon, color, bg = STATUS_STYLES.get(status, ("⏳", "#94a3b8", "#1e293b"))
    task_icon = TASK_ICONS.get(exp["task_type"], "🤖")
    n_done = exp.get("n_pipelines_done", 0)
    best = exp.get("best_result")
    opt_metric = exp.get("optimization_metric", "roc_auc")

    elapsed_str = ""
    if exp.get("elapsed_start"):
        if exp.get("finished_at"):
            elapsed_str = exp.get("finished_at", "")[11:16]
        elif exp["status"] == ExpStatus.RUNNING:
            secs = int(time.time() - exp["elapsed_start"])
            elapsed_str = f"{secs//60:02d}:{secs%60:02d}"

    border = f"2px solid {color}" if is_active else "1px solid #1e293b"

    # Best pipeline row
    best_row = ""
    if best:
        val = abs(best.get("primary_metric_cv", 0))
        best_row = (
            f"<div style='margin-top:8px;background:#0a0a14;border-radius:6px;"
            f"padding:6px 10px;font-size:11px;'>"
            f"🏆 <b style='color:#a78bfa'>{best.get('pipeline_id','')}</b>"
            f" {best.get('algorithm','')} &nbsp;"
            f"<span style='color:#10b981;font-weight:700'>{val:.4f}</span>"
            f" <span style='color:#64748b'>{opt_metric.upper()}</span></div>"
        )

    # Progress bar row (running only)
    progress_row = ""
    if status == ExpStatus.RUNNING:
        cfg = exp.get("config")
        max_p = getattr(cfg, "max_algorithms", 4) * getattr(cfg, "n_estimators_per_algo", 2)
        pct = min(100, int(n_done / max(1, max_p) * 100))
        progress_row = (
            f"<div style='margin-top:8px;background:#0f1929;border-radius:4px;height:4px'>"
            f"<div style='background:linear-gradient(90deg,#8B5CF6,#06B6D4);"
            f"height:4px;width:{pct}%;border-radius:4px'></div></div>"
            f"<div style='font-size:10px;color:#64748b;margin-top:3px'>"
            f"{n_done} pipelines done · {pct}%</div>"
        )

    elapsed_html = f"<span style='color:#64748b;font-size:10px'>⏱ {elapsed_str}</span>" if elapsed_str else ""

    # Render the entire card as raw HTML via st.html() — never touches Markdown parser
    card_html = (
        f"<div style='background:{bg};border:{border};border-radius:12px;"
        f"padding:14px 16px;margin-bottom:6px;font-family:Inter,sans-serif'>"
        f"<div style='display:flex;justify-content:space-between;align-items:flex-start'>"
        f"<div style='font-size:14px;font-weight:700;color:#e2e8f0'>{task_icon} {exp['name']}</div>"
        f"<div>"
        f"<span style='background:{color}22;color:{color};border-radius:5px;"
        f"padding:2px 8px;font-size:11px;font-weight:600'>{icon} {status.title()}</span>"
        f"</div></div>"
        f"<div style='font-size:11px;color:#64748b;margin-top:4px'>"
        f"📋 {exp['dataset_name']} · 🎯 {exp['target_column']} · 🕐 {exp['created_at'][11:16]}"
        f"&nbsp;{elapsed_html}</div>"
        f"{progress_row}"
        f"{best_row}"
        f"</div>"
    )
    st.html(card_html)

    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        if st.button("👁️ View", key=f"view_{exp['id']}", use_container_width=True):
            state.set("active_experiment_id", exp["id"])
            st.rerun()
    with btn_col2:
        if status == ExpStatus.RUNNING:
            if st.button("⏹ Stop", key=f"stop2_{exp['id']}", use_container_width=True):
                em.stop_experiment(exp["id"])
                st.rerun()
        else:
            if st.button("🗑️ Delete", key=f"del_{exp['id']}", use_container_width=True):
                if state.get("active_experiment_id") == exp["id"]:
                    state.set("active_experiment_id", None)
                em.delete_experiment(exp["id"])
                st.rerun()



# ── Compare Experiments ────────────────────────────────────────────────────────

def _render_comparison_chart(experiments: list) -> go.Figure:
    """Radar chart comparing best metrics across completed experiments."""
    done_exps = [e for e in experiments if e["status"] == ExpStatus.DONE and e.get("best_result")]
    if len(done_exps) < 2:
        return None

    fig = go.Figure()
    metrics = ["primary_metric_cv", "primary_metric_holdout"]

    for exp in done_exps[:6]:  # max 6 experiments in comparison
        best = exp["best_result"]
        icon, color, _ = STATUS_STYLES.get(exp["status"], ("", "#8B5CF6", ""))
        vals = [abs(best.get(m, 0)) for m in metrics]
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=["CV Score", "Holdout Score", "CV Score"],
            fill="toself",
            name=exp["name"][:20],
            line_color=color,
            opacity=0.7,
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        paper_bgcolor="#0a0a14",
        plot_bgcolor="#0a0a14",
        font=dict(color="#94a3b8"),
        height=280,
        margin=dict(t=20, b=20, l=20, r=20),
        legend=dict(font=dict(size=10)),
    )
    return fig


def _get_running_pid(exp: dict) -> str:
    for p in reversed(exp.get("pipelines_state", [])):
        if 0 < p.get("nodes_done", 0) < 4:
            return p["pipeline_id"]
    return None
