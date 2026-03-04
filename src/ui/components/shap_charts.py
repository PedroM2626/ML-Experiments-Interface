"""
SHAP-based interactive charts for MLine.
Beeswarm (global importance), Waterfall (local explanation), Bar summary.
"""

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from typing import List, Optional

_BG = "#0a0a14"
_GRID = "#1e293b"
_TEXT = "#94a3b8"
_PURPLE = "#8B5CF6"
_CYAN = "#06B6D4"
_GREEN = "#10b981"
_RED = "#ef4444"
_ORANGE = "#f59e0b"


def _base_layout(title: str = "", height: int = 400) -> dict:
    return dict(
        paper_bgcolor=_BG, plot_bgcolor=_BG,
        font=dict(color=_TEXT, family="Inter, sans-serif", size=11),
        title=dict(text=title, font=dict(size=13, color="#e2e8f0")),
        height=height,
        margin=dict(l=150, r=30, t=40, b=40),
        xaxis=dict(gridcolor=_GRID, zerolinecolor=_GRID),
        yaxis=dict(gridcolor=_GRID),
    )


# ── Bar Summary (Global importance) ───────────────────────────────────────────

def build_shap_bar_summary(
    shap_values: np.ndarray,
    feature_names: List[str],
    top_n: int = 15,
) -> go.Figure:
    """
    Horizontal bar chart of mean |SHAP| per feature (global importance).
    """
    mean_abs = np.abs(shap_values).mean(axis=0)
    df_idx = np.argsort(mean_abs)[-top_n:]
    feats = [feature_names[i] for i in df_idx]
    vals = [float(mean_abs[i]) for i in df_idx]

    # Color gradient from gray to purple based on importance
    colors = [
        f"rgba(139,92,246,{min(1.0, 0.3 + 0.7*(v/max(vals))):.2f})"
        for v in vals
    ]

    fig = go.Figure(go.Bar(
        x=vals, y=feats,
        orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v:.4f}" for v in vals],
        textposition="outside",
        textfont=dict(size=9),
    ))
    fig.update_layout(
        **_base_layout("🔍 SHAP — Global Feature Importance (mean |SHAP|)", 380),
        xaxis_title="Mean |SHAP value|",
        bargap=0.3,
    )
    return fig


# ── Beeswarm (Global, dot plot per feature × sample) ──────────────────────────

def build_shap_beeswarm(
    shap_values: np.ndarray,
    feature_names: List[str],
    top_n: int = 12,
) -> go.Figure:
    """
    Beeswarm-style scatter: y = feature, x = SHAP value, colour = raw value rank.
    Approximated in Plotly as a scatter with jitter.
    """
    mean_abs = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_abs)[-top_n:]

    fig = go.Figure()

    rng = np.random.default_rng(42)
    for rank, fi in enumerate(top_idx):
        sv = shap_values[:, fi]
        jitter = rng.uniform(-0.25, 0.25, size=len(sv))

        # Colour by normalised SHAP sign
        colors = [_PURPLE if v > 0 else _CYAN for v in sv]

        fig.add_trace(go.Scatter(
            x=sv.tolist(),
            y=[rank + j for j in jitter],
            mode="markers",
            name=feature_names[fi],
            showlegend=False,
            marker=dict(
                size=4, color=colors, opacity=0.65,
                line=dict(width=0),
            ),
            hovertemplate=f"<b>{feature_names[fi]}</b><br>SHAP: %{{x:.4f}}<extra></extra>",
        ))

    tick_labels = [feature_names[i] for i in top_idx]
    tick_vals = list(range(len(top_idx)))

    fig.add_vline(x=0, line_color=_GRID, line_dash="dash", line_width=1)
    fig.update_layout(
        **_base_layout("🐝 SHAP Beeswarm — Feature Impact Distribution", 420),
        xaxis_title="SHAP Value (impact on model output)",
        yaxis=dict(
            tickmode="array", tickvals=tick_vals, ticktext=tick_labels,
            gridcolor=_GRID,
        ),
        margin=dict(l=160, r=30, t=50, b=40),
    )
    return fig


# ── Waterfall (Local, single prediction) ──────────────────────────────────────

def build_shap_waterfall(
    shap_values_1d: np.ndarray,
    feature_names: List[str],
    feature_values: Optional[List] = None,
    base_value: float = 0.0,
    top_n: int = 12,
    prediction_label: str = "",
) -> go.Figure:
    """
    Waterfall chart: shows how each feature pushes the prediction
    above/below the base value.
    """
    sv = np.array(shap_values_1d)

    # Sort by |SHAP| descending, take top_n
    order = np.argsort(np.abs(sv))[-top_n:][::-1]
    selected_sv = sv[order]
    selected_feat = [feature_names[i] for i in order]
    selected_vals = [feature_values[i] if feature_values else "" for i in order] if feature_values else [""] * len(order)

    cumsum = np.cumsum(selected_sv)
    start = [base_value] + (base_value + cumsum[:-1]).tolist()
    end = (base_value + cumsum).tolist()
    colors = [_PURPLE if v > 0 else _CYAN for v in selected_sv]

    # Build waterfall
    labels = [
        f"{f}<br><span style='font-size:9px;color:#64748b;'>{v}</span>"
        if v != "" else f"{f}"
        for f, v in zip(selected_feat, selected_vals)
    ]

    fig = go.Figure(go.Waterfall(
        orientation="h",
        measure=["relative"] * len(selected_sv),
        x=selected_sv.tolist(),
        y=labels,
        base=start,
        connector=dict(line=dict(color=_GRID, width=1, dash="dot")),
        increasing=dict(marker=dict(color=_PURPLE)),
        decreasing=dict(marker=dict(color=_CYAN)),
        text=[f"+{v:.3f}" if v > 0 else f"{v:.3f}" for v in selected_sv],
        textposition="outside",
        textfont=dict(size=9),
    ))

    # Base value line
    fig.add_vline(x=base_value, line_color="#64748b", line_dash="dash", line_width=1,
                  annotation_text=f"Base {base_value:.3f}",
                  annotation_font=dict(size=9, color="#64748b"))

    title = f"💧 SHAP Waterfall — Local Explanation"
    if prediction_label:
        title += f" ({prediction_label})"

    fig.update_layout(
        **_base_layout(title, 380),
        xaxis_title="SHAP Value",
        margin=dict(l=200, r=60, t=50, b=40),
    )
    return fig


# ── Render Wrappers for Streamlit ─────────────────────────────────────────────

def render_shap_summary(shap_values: np.ndarray, feature_names: List[str]):
    """Render SHAP summary (bar and beeswarm) in tabs."""
    tab1, tab2 = st.tabs(["Bar Summary", "Beeswarm"])
    with tab1:
        fig1 = build_shap_bar_summary(shap_values, feature_names)
        st.plotly_chart(fig1, use_container_width=True)
    with tab2:
        fig2 = build_shap_beeswarm(shap_values, feature_names)
        st.plotly_chart(fig2, use_container_width=True)


def render_shap_waterfall(
    shap_value_1d: np.ndarray, 
    feature_names: List[str], 
    base_value: float = 0.0,
    prediction_label: str = ""
):
    """Render SHAP waterfall for a single prediction."""
    fig = build_shap_waterfall(
        shap_value_1d, 
        feature_names, 
        base_value=base_value,
        prediction_label=prediction_label
    )
    st.plotly_chart(fig, use_container_width=True)
