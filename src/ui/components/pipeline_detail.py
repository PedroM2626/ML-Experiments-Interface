"""
Pipeline Detail — feature importance chart, metrics table, hyperparameter view.
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Any
from .shap_charts import render_shap_summary, render_shap_waterfall
from .eval_charts import (
    render_calibration_curve, 
    render_residuals_plot, 
    render_learning_curve, 
    render_per_class_metrics
)


def render_pipeline_detail(result: Dict, task_type: str = "classification"):
    """Render a detailed view for a single pipeline result."""
    pid = result.get("pipeline_id", "")
    algo = result.get("algorithm", "")
    transformer = result.get("transformer", "")

    st.markdown(f"""
    <div style='background:#0f1929;border:1px solid #1e293b;border-radius:12px;padding:16px 20px;margin-bottom:16px;'>
        <div style='display:flex;align-items:center;gap:12px;'>
            <span style='font-size:22px;font-weight:800;color:#e2e8f0;'>{pid}</span>
            <span style='background:#2d1b69;border-radius:6px;padding:4px 12px;font-size:13px;color:#a78bfa;font-weight:600;'>{algo}</span>
            <span style='background:#1e293b;border-radius:6px;padding:4px 10px;font-size:11px;color:#64748b;'>{transformer}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1.2, 1])

    with col1:
        _render_feature_importance(result)
        
        # --- Advanced Evaluation Charts ---
        st.divider()
        if task_type == "classification":
            cal_data = result.get("calibration_data")
            if cal_data:
                st.markdown("**🎯 Calibration Curve**")
                render_calibration_curve(*cal_data)
            
        elif task_type == "regression":
            res_data = result.get("residuals_data")
            if res_data:
                st.markdown("**📉 Residuals Analysis**")
                render_residuals_plot(*res_data)

        lc_data = result.get("learning_curve_data")
        if lc_data:
            st.markdown("**📈 Learning Curve**")
            render_learning_curve(*lc_data)

    with col2:
        _render_metrics_table(result, task_type)
        _render_hyperparams(result)
        
        # --- SHAP Explanations ---
        st.divider()
        st.markdown("**✨ Model Explainability (SHAP)**")
        
        shap_values = result.get("shap_values")
        feature_names = result.get("feature_names")
        base_value = result.get("shap_base_value", 0.0)
        
        if shap_values is not None and feature_names is not None:
            tab_global, tab_local = st.tabs(["Global Importance", "Local Explanation"])
            
            with tab_global:
                render_shap_summary(shap_values, feature_names)
                st.caption("SHAP values distribution showing feature impact on model output.")
            
            with tab_local:
                st.markdown("###### Local Explanation (Waterfall)")
                # Sample selection
                idx = st.slider("Select sample index", 0, len(shap_values)-1, 0, key=f"shap_idx_{pid}")
                render_shap_waterfall(shap_values[idx], feature_names, float(base_value))
                st.caption(f"Detailed breakdown for sample {idx}.")
        else:
            st.info("SHAP values not available. Ensure 'Explainability' was enabled.")


def _render_feature_importance(result: Dict):
    fi_data = result.get("feature_importance", [])
    if not fi_data:
        st.info("Feature importance not available for this algorithm.")
        return

    fi_df = pd.DataFrame(fi_data).head(15)
    fig = go.Figure()

    colors = [
        f"rgba(139, 92, 246, {0.4 + 0.6 * (1 - i / len(fi_df))})"
        for i in range(len(fi_df))
    ]

    fig.add_trace(go.Bar(
        x=fi_df["importance"][::-1],
        y=fi_df["feature"][::-1],
        orientation="h",
        marker=dict(color=colors[::-1], line=dict(width=0)),
        hovertemplate="%{y}: %{x:.4f}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(text="Feature Importance (Top 15)", font=dict(size=13, color="#e2e8f0")),
        paper_bgcolor="#0f1929",
        plot_bgcolor="#0f1929",
        height=350,
        margin=dict(l=10, r=10, t=40, b=20),
        xaxis=dict(
            color="#64748b", gridcolor="#1e293b", zeroline=False,
            title=dict(text="Importance", font=dict(color="#64748b", size=10)),
        ),
        yaxis=dict(color="#94a3b8", tickfont=dict(size=9)),
        font=dict(color="#e2e8f0", family="Inter, sans-serif"),
    )

    st.plotly_chart(fig, use_container_width=True)


def _render_metrics_table(result: Dict, task_type: str):
    st.markdown("**📊 Metrics**")
    cv = result.get("cv_scores", {})
    hd = result.get("holdout_scores", {})

    all_keys = list({**cv, **hd}.keys())
    rows = []
    for k in all_keys:
        if "neg_" in k:
            continue
        cv_val = cv.get(k)
        hd_val = hd.get(k)
        rows.append({
            "Metric": k.replace("_", " ").title(),
            "CV": f"{cv_val:.4f}" if cv_val is not None else "—",
            "Holdout": f"{hd_val:.4f}" if hd_val is not None else "—",
        })

    if rows:
        df_metrics = pd.DataFrame(rows)
        st.dataframe(
            df_metrics,
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.info("No metrics available.")


def _render_hyperparams(result: Dict):
    hparams = result.get("hyperparams", {})
    if not hparams:
        return

    st.markdown("**⚙️ Hyperparameters**")
    hp_rows = [{"Parameter": k, "Value": str(v)} for k, v in hparams.items()]
    st.dataframe(
        pd.DataFrame(hp_rows),
        hide_index=True,
        use_container_width=True,
        height=200,
    )
