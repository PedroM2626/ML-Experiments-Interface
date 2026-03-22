"""
Pipeline Leaderboard — ranked table of all AutoML pipelines.
"""

import streamlit as st
import pandas as pd
from typing import List, Dict, Optional


RANK_BADGES = {1: "🥇", 2: "🥈", 3: "🥉"}

ALGO_COLORS = {
    "GradientBoosting": "#8B5CF6",
    "RandomForest": "#3B82F6",
    "ExtraTrees": "#06B6D4",
    "XGBoost": "#10B981",
    "LightGBM": "#F59E0B",
    "CatBoost": "#EF4444",
    "LogisticRegression": "#6366F1",
    "Ridge": "#6366F1",
    "KNeighbors": "#EC4899",
}


def _badge(algo: str) -> str:
    color = ALGO_COLORS.get(algo, "#6366F1")
    return f'<span style="background:{color};border-radius:4px;padding:2px 8px;font-size:11px;color:white;font-weight:600;">{algo}</span>'


def _enhancements(result: Dict) -> str:
    tags = []
    if result.get("use_pca"):
        tags.append('<span style="background:#1e3a5f;border-radius:4px;padding:1px 6px;font-size:10px;color:#60a5fa;">PCA</span>')
    if result.get("use_select_k"):
        tags.append('<span style="background:#1e3a2c;border-radius:4px;padding:1px 6px;font-size:10px;color:#34d399;">SKB</span>')
    tags.append(f'<span style="background:#2d1b69;border-radius:4px;padding:1px 6px;font-size:10px;color:#a78bfa;">HPO</span>')
    tags.append(f'<span style="background:#1e293b;border-radius:4px;padding:1px 6px;font-size:10px;color:#94a3b8;">{result.get("transformer","")[:4]}FE</span>')
    return " ".join(tags)


def render_leaderboard(
    results: List[Dict],
    optimization_metric: str = "roc_auc",
    task_type: str = "classification",
    selected_pipeline_id: Optional[str] = None,
) -> Optional[str]:
    """
    Render the pipeline leaderboard.
    Returns the pipeline_id that the user clicked on (or None).
    """
    if not results:
        st.info("⏳ Training pipelines… leaderboard will appear as pipelines complete.")
        return selected_pipeline_id

    sorted_results = sorted(results, key=lambda r: r.get("primary_metric_cv", 0), reverse=True)

    clicked = selected_pipeline_id

    st.markdown("""
    <style>
    .lb-row {
        display: flex; align-items: center; gap: 12px;
        padding: 10px 14px; border-radius: 10px; margin-bottom: 6px;
        cursor: pointer; transition: background 0.2s;
        background: #0f1929; border: 1px solid #1e293b;
    }
    .lb-row:hover { background: #1a2744; border-color: #6366F1; }
    .lb-row.selected { background: #1e1b4b; border-color: #8B5CF6; }
    .lb-rank { font-size: 18px; min-width: 28px; }
    .lb-pid { font-size: 12px; color: #94a3b8; min-width: 28px; }
    .lb-metric { font-size: 14px; font-weight: 700; min-width: 54px; }
    .lb-holdout { font-size: 12px; color: #94a3b8; min-width: 54px; }
    .lb-time { font-size: 11px; color: #64748b; min-width: 46px; }
    .lb-enhancements { flex: 1; }
    </style>
    """, unsafe_allow_html=True)

    # Header
    col_h = st.columns([0.8, 0.6, 2.0, 1.2, 1.2, 1.5, 1.0])
    headers = ["Rank", "ID", "Algorithm", f"CV {optimization_metric.upper()}", "Holdout", "Enhancements", "Time"]
    for col, h in zip(col_h, headers):
        col.markdown(f"<span style='font-size:11px;color:#64748b;font-weight:600;'>{h}</span>", unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#1e293b;margin:4px 0 8px 0;'>", unsafe_allow_html=True)

    selected = selected_pipeline_id
    for rank, result in enumerate(sorted_results, 1):
        pid = result.get("pipeline_id", "")
        is_selected = pid == selected_pipeline_id
        cv_val = result.get("primary_metric_cv", 0)
        hd_val = result.get("primary_metric_holdout", 0)
        algo = result.get("algorithm", "")
        build_t = result.get("build_time", 0)

        badge = RANK_BADGES.get(rank, f"#{rank}")
        row_bg = "#1e1b4b" if is_selected else "#0f1929"
        border_col = "#8B5CF6" if is_selected else "#1e293b"

        cols = st.columns([0.8, 0.6, 2.0, 1.2, 1.2, 1.5, 1.0])
        cols[0].markdown(f"<span style='font-size:18px;'>{badge}</span>", unsafe_allow_html=True)
        cols[1].markdown(f"<span style='font-size:12px;color:#94a3b8;'>{pid}</span>", unsafe_allow_html=True)
        cols[2].markdown(_badge(algo), unsafe_allow_html=True)
        cols[3].markdown(f"<span style='font-size:14px;font-weight:700;color:#e2e8f0;'>{cv_val:.4f}</span>", unsafe_allow_html=True)
        cols[4].markdown(f"<span style='font-size:12px;color:#94a3b8;'>{hd_val:.4f}</span>", unsafe_allow_html=True)
        cols[5].markdown(_enhancements(result), unsafe_allow_html=True)
        cols[6].markdown(f"<span style='font-size:11px;color:#64748b;'>{build_t:.1f}s</span>", unsafe_allow_html=True)

        if cols[1].button("▶", key=f"_lb_select_{pid}", help=f"View {pid} details"):
            selected = pid

        st.markdown("<div style='height:2px;'></div>", unsafe_allow_html=True)

    return selected
