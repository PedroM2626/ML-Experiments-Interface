"""
Visualization utilities – Plotly charts for AutoML Studio.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Theme ───────────────────────────────────────────────────────────────────────
PALETTE = [
    "#7C3AED", "#2563EB", "#059669", "#D97706",
    "#DC2626", "#0891B2", "#7C3AED", "#DB2777",
]

LAYOUT_DEFAULTS = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#94A3B8", size=11),
    margin=dict(l=0, r=0, t=30, b=0),
    showlegend=False,
)


def _apply_theme(fig: go.Figure) -> go.Figure:
    fig.update_layout(**LAYOUT_DEFAULTS)
    fig.update_xaxes(gridcolor="#1E293B", showline=False, zeroline=False)
    fig.update_yaxes(gridcolor="#1E293B", showline=False, zeroline=False)
    return fig


# ── Column Distribution ─────────────────────────────────────────────────────────

def column_distribution_chart(series: pd.Series, col_name: str, col_type: str) -> go.Figure:
    """Mini distribution chart for a column (histogram or bar)."""
    clean = series.dropna()
    if len(clean) == 0:
        fig = go.Figure()
        fig.update_layout(title="No data", **LAYOUT_DEFAULTS)
        return fig

    if col_type in ("Numeric",):
        fig = go.Figure(go.Histogram(
            x=clean,
            nbinsx=30,
            marker_color=PALETTE[0],
            opacity=0.85,
        ))
        fig.add_vline(x=clean.mean(), line_dash="dash", line_color="#A78BFA", line_width=1)
    elif col_type in ("Binary", "Categorical"):
        vc = clean.value_counts().head(15)
        fig = go.Figure(go.Bar(
            x=vc.values.tolist(),
            y=vc.index.astype(str).tolist(),
            orientation="h",
            marker_color=PALETTE[0],
        ))
    elif col_type == "Datetime":
        parsed = pd.to_datetime(clean, errors="coerce").dropna()
        by_month = parsed.dt.to_period("M").value_counts().sort_index()
        fig = go.Figure(go.Bar(
            x=[str(p) for p in by_month.index],
            y=by_month.values.tolist(),
            marker_color=PALETTE[1],
        ))
    else:
        # Text / ID – just word count distribution
        lengths = clean.astype(str).str.len()
        fig = go.Figure(go.Histogram(x=lengths, nbinsx=20, marker_color=PALETTE[2]))

    fig.update_layout(title=dict(text=col_name, font=dict(size=12, color="#F1F5F9")),
                      height=120, **LAYOUT_DEFAULTS)
    return _apply_theme(fig)


# ── Feature Importance ──────────────────────────────────────────────────────────

def feature_importance_chart(fi: dict[str, float], top_n: int = 15) -> go.Figure:
    """Horizontal bar chart of feature importance."""
    # Sort descending, take top N
    sorted_fi = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:top_n]
    cols, vals = zip(*sorted_fi) if sorted_fi else ([], [])

    # Color gradient
    n = len(vals)
    colors = [
        f"rgba(124,58,237,{0.4 + 0.6 * (i / max(n - 1, 1))})"
        for i in range(n - 1, -1, -1)
    ]

    fig = go.Figure(go.Bar(
        x=list(vals),
        y=list(cols),
        orientation="h",
        marker_color=colors,
        text=[f"{v:.2%}" for v in vals],
        textposition="auto",
        textfont=dict(color="#F1F5F9", size=11),
    ))
    fig.update_layout(
        title="Column Impact",
        height=max(250, n * 28),
        xaxis=dict(tickformat=".0%", title=""),
        yaxis=dict(autorange="reversed"),
        **LAYOUT_DEFAULTS,
    )
    return _apply_theme(fig)


# ── Confusion Matrix ────────────────────────────────────────────────────────────

def confusion_matrix_chart(cm: np.ndarray, labels: list[str]) -> go.Figure:
    """Annotated heatmap for confusion matrix."""
    # Normalize for color scale
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)
    annotations = []
    for i in range(len(labels)):
        for j in range(len(labels)):
            annotations.append(dict(
                x=labels[j], y=labels[i],
                text=f"{cm[i, j]}<br>({cm_norm[i, j]:.1%})",
                showarrow=False,
                font=dict(color="#F1F5F9", size=11),
            ))

    fig = go.Figure(go.Heatmap(
        z=cm_norm,
        x=labels, y=labels,
        colorscale=[[0, "#0F172A"], [0.5, "#4C1D95"], [1, "#7C3AED"]],
        showscale=False,
    ))
    fig.update_layout(
        title="Confusion Matrix",
        annotations=annotations,
        xaxis=dict(title="Predicted"),
        yaxis=dict(title="Actual", autorange="reversed"),
        height=350,
        **LAYOUT_DEFAULTS,
    )
    return _apply_theme(fig)


# ── ROC Curve ──────────────────────────────────────────────────────────────────

def roc_curve_chart(fpr: list, tpr: list, auc: float) -> go.Figure:
    """ROC curve with AUC."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode="lines",
        name=f"ROC (AUC = {auc:.3f})",
        line=dict(color=PALETTE[0], width=2),
        fill="tozeroy",
        fillcolor="rgba(124,58,237,0.1)",
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        line=dict(color="#475569", width=1, dash="dash"),
        name="Random",
    ))
    fig.update_layout(
        title=f"ROC Curve — AUC = {auc:.3f}",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=350,
        showlegend=True,
        legend=dict(x=0.6, y=0.1, font=dict(color="#94A3B8")),
        **LAYOUT_DEFAULTS,
    )
    return _apply_theme(fig)


# ── Residuals Chart ────────────────────────────────────────────────────────────

def residuals_chart(y_true: pd.Series, y_pred: pd.Series) -> go.Figure:
    """Scatter plot of residuals for regression."""
    residuals = pd.Series(y_true).values - pd.Series(y_pred).values
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Actual vs Predicted", "Residuals"))

    fig.add_trace(go.Scatter(
        x=y_true, y=y_pred, mode="markers",
        marker=dict(color=PALETTE[0], opacity=0.6, size=5),
        name="Predictions",
    ), row=1, col=1)

    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    fig.add_trace(go.Scatter(
        x=[min_val, max_val], y=[min_val, max_val],
        mode="lines", line=dict(color="#475569", dash="dash"),
        name="Perfect fit",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=y_pred, y=residuals, mode="markers",
        marker=dict(color=PALETTE[3], opacity=0.6, size=5),
        name="Residuals",
    ), row=1, col=2)
    fig.add_hline(y=0, line_dash="dash", line_color="#475569", row=1, col=2)

    fig.update_layout(height=350, showlegend=False, **LAYOUT_DEFAULTS)
    return _apply_theme(fig)


# ── Precision-Recall ───────────────────────────────────────────────────────────

def precision_recall_chart(precision: list, recall: list, ap: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=recall, y=precision,
        mode="lines",
        line=dict(color=PALETTE[1], width=2),
        fill="tozeroy",
        fillcolor="rgba(37,99,235,0.1)",
        name=f"AP = {ap:.3f}",
    ))
    fig.update_layout(
        title=f"Precision-Recall — AP = {ap:.3f}",
        xaxis_title="Recall",
        yaxis_title="Precision",
        height=350,
        **LAYOUT_DEFAULTS,
    )
    return _apply_theme(fig)


# ── Model Leaderboard Bar ───────────────────────────────────────────────────────

def leaderboard_chart(leaderboard_df: pd.DataFrame, metric: str) -> go.Figure:
    """Horizontal bar chart of model leaderboard scores."""
    df = leaderboard_df.sort_values(metric, ascending=True).tail(10)
    model_names = df["model"].astype(str).tolist() if "model" in df.columns else df.index.astype(str).tolist()
    scores = df[metric].tolist()

    fig = go.Figure(go.Bar(
        x=scores,
        y=model_names,
        orientation="h",
        marker=dict(
            color=scores,
            colorscale=[[0, "#1E3A5F"], [1, "#7C3AED"]],
        ),
        text=[f"{s:.4f}" for s in scores],
        textposition="auto",
        textfont=dict(color="#F1F5F9"),
    ))
    fig.update_layout(
        title=f"Model Leaderboard — {metric.upper()}",
        height=max(200, len(model_names) * 30),
        **LAYOUT_DEFAULTS,
    )
    return _apply_theme(fig)


# ── Value Distribution Bar ─────────────────────────────────────────────────────

def target_distribution_chart(series: pd.Series, col_name: str) -> go.Figure:
    """Large distribution bar for target column like Canvas."""
    clean = series.dropna()

    if pd.api.types.is_numeric_dtype(clean) and clean.nunique() > 10:
        fig = go.Figure(go.Histogram(
            x=clean, nbinsx=30,
            marker_color=PALETTE[0], opacity=0.85,
        ))
    else:
        vc = clean.value_counts()
        colors = PALETTE[: len(vc)]
        fig = go.Figure(go.Bar(
            x=vc.index.astype(str).tolist(),
            y=vc.values.tolist(),
            marker_color=colors,
        ))

    fig.update_layout(
        title=f"Value Distribution — {col_name}",
        height=200, **LAYOUT_DEFAULTS,
    )
    return _apply_theme(fig)
