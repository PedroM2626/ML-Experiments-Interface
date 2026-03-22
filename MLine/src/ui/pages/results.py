"""
Results page — full experiment summary, export, and MLflow link.
"""

import os
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import joblib

from src.utils import state
from src.ui.components.leaderboard import render_leaderboard, ALGO_COLORS
from src.ui.components.pipeline_detail import render_pipeline_detail
from src.utils.report_generator import generate_pdf_report


def render():
    state.init_state()

    results = state.get_active_results()
    active_exp = state.get_active_experiment()
    
    if not results:
        st.warning("No results yet. Please run an experiment first.")
        if st.button("← Back to Training"):
            # Redirect to experiments page if we have an active exp, otherwise training
            state.set("current_page", "experiments" if active_exp else "training")
            st.rerun()
        return

    # Use metadata from experiment if available
    dataset_name = active_exp["dataset_name"] if active_exp else state.get("dataset_name", "")
    task_type = active_exp["task_type"] if active_exp else state.get("task_type", "classification")
    opt_metric = active_exp["optimization_metric"] if active_exp else state.get("optimization_metric", "roc_auc")

    sorted_results = sorted(results, key=lambda r: abs(r.get("primary_metric_cv", 0)), reverse=True)
    best = sorted_results[0]
    
    # ── Sidebar: Model Download ─────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 💾 Export Model")
        exp_id = active_exp["id"] if active_exp else state.get("active_experiment_id", "")
        model_path = os.path.join("exports", exp_id, "best_model.pkl") if exp_id else os.path.join("exports", "best_model.pkl")
        
        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                st.download_button(
                    "📥 Download Best Model (.pkl)",
                    data=f,
                    file_name=f"best_model_{exp_id}.pkl",
                    mime="application/octet-stream",
                    use_container_width=True,
                    help="Download the trained scikit-learn pipeline for offline use."
                )
        else:
            st.warning("💾 Model pipeline not ready for download yet.")
        st.divider()

    # ── Header ─────────────────────────────────────────────────────────────────
    st.markdown("""
    <h2 style='font-size:26px;font-weight:900;background:linear-gradient(135deg,#8B5CF6,#06B6D4);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:4px;'>
        📈 Experiment Results
    </h2>
    """, unsafe_allow_html=True)

    col_nav1, col_nav2, col_nav3 = st.columns([1, 1, 1])
    with col_nav1:
        if st.button("← Back to Training", use_container_width=True):
            state.set("current_page", "experiments" if active_exp else "training")
            st.rerun()
    with col_nav2:
        if st.button("🔄 New Experiment", use_container_width=True):
            state.reset_training_state()
            state.set("current_page", "upload")
            st.rerun()
    with col_nav3:
        # New PDF Report Download
        report_path = os.path.join("exports", "executive_report.pdf")
        if st.button("📄 Generate Executive Report", use_container_width=True):
            with st.spinner("Generating professional PDF report..."):
                try:
                    os.makedirs("exports", exist_ok=True)
                    experiment_data = {
                        "name": active_exp.get("name") if active_exp else state.get("dataset_name", "AutoML Experiment"),
                        "dataset_name": dataset_name,
                        "task_type": task_type,
                        "target_column": active_exp.get("target_column") if active_exp else state.get("target_column", "target")
                    }
                    generate_pdf_report(
                        experiment_data=experiment_data,
                        best_pipeline=best,
                        all_results=sorted_results,
                        output_path=report_path
                    )
                    st.success("✅ Report generated!")
                except Exception as e:
                    st.error(f"Failed to generate report: {e}")

        if os.path.exists(report_path):
            with open(report_path, "rb") as f:
                st.download_button(
                    "⬇️ Download Executive Report (PDF)",
                    data=f,
                    file_name="executive_report.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )

    st.divider()

    # ── Best model card ─────────────────────────────────────────────────────────
    _render_best_card(best, active_exp)

    st.divider()

    # ── Metric comparison chart ─────────────────────────────────────────────────
    _render_comparison_chart(sorted_results, active_exp)

    st.divider()

    # ── Full leaderboard ────────────────────────────────────────────────────────
    st.markdown("### 🏆 Pipeline Leaderboard")
    selected_id = render_leaderboard(
        results=sorted_results,
        optimization_metric=opt_metric,
        task_type=task_type,
        selected_pipeline_id=state.get("selected_pipeline_id"),
    )
    if selected_id != state.get("selected_pipeline_id"):
        state.set("selected_pipeline_id", selected_id)

    # ── Pipeline detail ──────────────────────────────────────────────────────────
    selected_id = state.get("selected_pipeline_id")
    if selected_id:
        st.divider()
        result_map = {r["pipeline_id"]: r for r in results}
        if selected_id in result_map:
            render_pipeline_detail(result_map[selected_id], task_type=task_type)


def _render_best_card(best: dict, active_exp: dict = None):
    pid = best.get("pipeline_id", "")
    algo = best.get("algorithm", "")
    cv_score = best.get("primary_metric_cv", 0)
    hd_score = best.get("primary_metric_holdout", 0)
    opt_metric = (active_exp["optimization_metric"] if active_exp else state.get("optimization_metric", "roc_auc")).upper()
    builds = best.get("build_time", 0)
    transformer = best.get("transformer", "")

    color = ALGO_COLORS.get(algo, "#8B5CF6")

    st.markdown(f"""
    <div style='background:linear-gradient(135deg,#1a0533,#0c1a3a);border:1px solid {color};
    border-radius:16px;padding:24px 28px;margin-bottom:8px;'>
        <div style='display:flex;align-items:center;gap:16px;margin-bottom:14px;'>
            <span style='font-size:36px;'>🏆</span>
            <div>
                <div style='font-size:20px;font-weight:900;color:{color};'>Best Pipeline: {pid}</div>
                <div style='font-size:14px;color:#94a3b8;'>Dataset: {active_exp['dataset_name'] if active_exp else state.get("dataset_name","")} · {(active_exp['task_type'] if active_exp else state.get("task_type","")).title()}</div>
            </div>
        </div>
        <div style='display:grid;grid-template-columns:repeat(4,1fr);gap:12px;'>
            <div style='background:rgba(0,0,0,0.3);border-radius:10px;padding:12px;text-align:center;'>
                <div style='font-size:22px;font-weight:800;color:{color};'>{cv_score:.4f}</div>
                <div style='font-size:11px;color:#64748b;'>CV {opt_metric}</div>
            </div>
            <div style='background:rgba(0,0,0,0.3);border-radius:10px;padding:12px;text-align:center;'>
                <div style='font-size:22px;font-weight:800;color:#06B6D4;'>{hd_score:.4f}</div>
                <div style='font-size:11px;color:#64748b;'>Holdout {opt_metric}</div>
            </div>
            <div style='background:rgba(0,0,0,0.3);border-radius:10px;padding:12px;text-align:center;'>
                <div style='font-size:22px;font-weight:800;color:#e2e8f0;'>{algo}</div>
                <div style='font-size:11px;color:#64748b;'>Algorithm</div>
            </div>
            <div style='background:rgba(0,0,0,0.3);border-radius:10px;padding:12px;text-align:center;'>
                <div style='font-size:22px;font-weight:800;color:#FBBF24;'>{builds:.1f}s</div>
                <div style='font-size:11px;color:#64748b;'>Build Time</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def _render_comparison_chart(sorted_results: list, active_exp: dict = None):
    """Bar chart comparing all pipelines by CV score."""
    st.markdown("### 📉 Pipeline Comparison")
    opt_metric = active_exp["optimization_metric"] if active_exp else state.get("optimization_metric", "roc_auc")

    pids = [r["pipeline_id"] for r in sorted_results]
    cv_scores = [r.get("primary_metric_cv", 0) for r in sorted_results]
    hd_scores = [r.get("primary_metric_holdout", 0) for r in sorted_results]
    colors = [ALGO_COLORS.get(r.get("algorithm", ""), "#8B5CF6") for r in sorted_results]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name=f"CV {opt_metric.upper()}",
        x=pids, y=cv_scores,
        marker=dict(color=colors, opacity=0.9, line=dict(width=0)),
        hovertemplate="%{x}: %{y:.4f}<extra>CV</extra>",
    ))
    fig.add_trace(go.Bar(
        name=f"Holdout {opt_metric.upper()}",
        x=pids, y=hd_scores,
        marker=dict(color=colors, opacity=0.4, line=dict(width=0)),
        hovertemplate="%{x}: %{y:.4f}<extra>Holdout</extra>",
    ))

    fig.update_layout(
        barmode="group",
        paper_bgcolor="#0f0f1a",
        plot_bgcolor="#0f0f1a",
        height=320,
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(
            bgcolor="#0f0f1a", bordercolor="#1e293b",
            font=dict(color="#94a3b8"),
        ),
        xaxis=dict(
            color="#94a3b8", gridcolor="#1e293b", tickfont=dict(size=10),
            title=dict(text="Pipeline", font=dict(color="#64748b")),
        ),
        yaxis=dict(
            color="#94a3b8", gridcolor="#1e293b",
            title=dict(text=opt_metric.upper(), font=dict(color="#64748b")),
        ),
        font=dict(color="#e2e8f0", family="Inter, sans-serif"),
    )

    st.plotly_chart(fig, use_container_width=True)
