"""
Evaluation charts for MLine.
Calibration curves, learning curves, residual analysis, per-class reports.
All return Plotly figures matching the dark theme.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.figure_factory as ff
import streamlit as st
from typing import List, Dict, Optional, Tuple
from sklearn.calibration import calibration_curve
from sklearn.model_selection import learning_curve, StratifiedKFold, KFold
from sklearn.metrics import classification_report

import warnings
warnings.filterwarnings("ignore")

_BG = "#0a0a14"
_GRID = "#1e293b"
_TEXT = "#94a3b8"
_PURPLE = "#8B5CF6"
_CYAN = "#06B6D4"
_GREEN = "#10b981"
_RED = "#ef4444"


def _base_layout(title: str = "", height: int = 360) -> dict:
    return dict(
        paper_bgcolor=_BG, plot_bgcolor=_BG,
        font=dict(color=_TEXT, family="Inter, sans-serif", size=11),
        title=dict(text=title, font=dict(size=13, color="#e2e8f0")),
        height=height,
        margin=dict(l=50, r=20, t=40, b=50),
        xaxis=dict(gridcolor=_GRID, zerolinecolor=_GRID),
        yaxis=dict(gridcolor=_GRID, zerolinecolor=_GRID),
    )


# ── Calibration Curve ──────────────────────────────────────────────────────────

def compute_calibration_data(pipeline, X_holdout, y_holdout, n_bins: int = 10):
    """
    Compute calibration curve data.
    Returns (fraction_of_positives, mean_predicted_value) or (None, None).
    """
    try:
        if not hasattr(pipeline, "predict_proba"):
            return None, None
        proba = pipeline.predict_proba(X_holdout)
        if proba.shape[1] == 2:
            y_prob = proba[:, 1]
        else:
            return None, None
        fop, mpv = calibration_curve(y_holdout, y_prob, n_bins=n_bins)
        return fop.tolist(), mpv.tolist()
    except Exception:
        return None, None


def build_calibration_chart(
    fraction_of_positives: List[float],
    mean_predicted: List[float],
    pipeline_id: str = "Model",
) -> go.Figure:
    fig = go.Figure()

    # Perfect calibration reference
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        line=dict(color=_GRID, dash="dash", width=1),
        name="Perfect",
        showlegend=True,
    ))

    # Model calibration
    fig.add_trace(go.Scatter(
        x=mean_predicted, y=fraction_of_positives,
        mode="lines+markers",
        line=dict(color=_PURPLE, width=2),
        marker=dict(size=7, color=_PURPLE),
        name=pipeline_id,
        showlegend=True,
    ))

    fig.update_layout(
        **_base_layout("📊 Calibration Curve", 320),
        xaxis_title="Mean Predicted Probability",
        yaxis_title="Fraction of Positives",
        showlegend=True,
        legend=dict(bgcolor="#0a0a14", bordercolor=_GRID, font=dict(size=10)),
    )
    return fig


# ── Learning Curve ─────────────────────────────────────────────────────────────

def compute_learning_curve_data(
    pipeline,
    X,
    y,
    task_type: str = "classification",
    n_splits: int = 3,
    train_sizes: Optional[List[float]] = None,
):
    """
    Compute learning curve (train size vs CV score).
    Returns (train_sizes_abs, train_mean, train_std, val_mean, val_std) or all-None on fail.
    """
    try:
        if train_sizes is None:
            train_sizes = [0.2, 0.4, 0.6, 0.8, 1.0]
        scoring = "roc_auc" if task_type == "classification" else "r2"
        cv = (StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
              if task_type == "classification"
              else KFold(n_splits=n_splits, shuffle=True, random_state=42))
        ts, tr_scores, val_scores = learning_curve(
            pipeline, X, y,
            train_sizes=train_sizes,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
        )
        return (
            ts.tolist(),
            np.mean(tr_scores, axis=1).tolist(),
            np.std(tr_scores, axis=1).tolist(),
            np.mean(val_scores, axis=1).tolist(),
            np.std(val_scores, axis=1).tolist(),
        )
    except Exception:
        return None, None, None, None, None


def build_learning_curve_chart(
    train_sizes: List[int],
    train_mean: List[float],
    train_std: List[float],
    val_mean: List[float],
    val_std: List[float],
) -> go.Figure:
    fig = go.Figure()

    ts = np.array(train_sizes)
    tr_m = np.array(train_mean)
    tr_s = np.array(train_std)
    v_m = np.array(val_mean)
    v_s = np.array(val_std)

    # Train band
    fig.add_trace(go.Scatter(
        x=np.concatenate([ts, ts[::-1]]).tolist(),
        y=np.concatenate([tr_m + tr_s, (tr_m - tr_s)[::-1]]).tolist(),
        fill="toself", fillcolor=f"rgba(139,92,246,0.15)",
        line=dict(color="rgba(0,0,0,0)"), showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=ts.tolist(), y=tr_m.tolist(),
        mode="lines+markers", name="Train",
        line=dict(color=_PURPLE, width=2), marker=dict(size=5),
    ))

    # Validation band
    fig.add_trace(go.Scatter(
        x=np.concatenate([ts, ts[::-1]]).tolist(),
        y=np.concatenate([v_m + v_s, (v_m - v_s)[::-1]]).tolist(),
        fill="toself", fillcolor=f"rgba(6,182,212,0.15)",
        line=dict(color="rgba(0,0,0,0)"), showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=ts.tolist(), y=v_m.tolist(),
        mode="lines+markers", name="Validation",
        line=dict(color=_CYAN, width=2), marker=dict(size=5),
    ))

    fig.update_layout(
        **_base_layout("📈 Learning Curve", 320),
        xaxis_title="Training Samples",
        yaxis_title="Score",
        showlegend=True,
        legend=dict(bgcolor="#0a0a14", bordercolor=_GRID, font=dict(size=10)),
    )
    return fig


# ── Residual Analysis (Regression) ─────────────────────────────────────────────

def compute_residuals(pipeline, X_holdout, y_holdout):
    """Returns (y_pred, residuals) or (None, None)."""
    try:
        y_pred = pipeline.predict(X_holdout)
        residuals = np.array(y_holdout) - y_pred
        return y_pred.tolist(), residuals.tolist()
    except Exception:
        return None, None


def build_residuals_chart(y_pred: List[float], residuals: List[float]) -> go.Figure:
    """Residuals vs Predicted scatter plot."""
    fig = go.Figure()
    fig.add_hline(y=0, line_color=_GRID, line_dash="dash")
    fig.add_trace(go.Scatter(
        x=y_pred, y=residuals,
        mode="markers",
        marker=dict(color=_PURPLE, size=5, opacity=0.6,
                    line=dict(width=0.5, color=_CYAN)),
        name="Residuals",
    ))
    fig.update_layout(
        **_base_layout("🔍 Residuals vs Predicted", 320),
        xaxis_title="Predicted Value",
        yaxis_title="Residual (Actual − Predicted)",
    )
    return fig


def build_qq_chart(residuals: List[float]) -> go.Figure:
    """QQ-plot of residuals vs theoretical normal quantiles."""
    from scipy import stats
    res = np.array(residuals)
    (osm, osr), (slope, intercept, _) = stats.probplot(res)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(osm), y=list(osr),
        mode="markers",
        marker=dict(color=_PURPLE, size=5, opacity=0.7),
        name="Residuals",
    ))
    # Reference line
    x_line = [float(min(osm)), float(max(osm))]
    y_line = [slope * x + intercept for x in x_line]
    fig.add_trace(go.Scatter(
        x=x_line, y=y_line, mode="lines",
        line=dict(color=_CYAN, dash="dash", width=1),
        name="Normal",
    ))
    fig.update_layout(
        **_base_layout("📐 QQ Plot (Residuals)", 320),
        xaxis_title="Theoretical Quantiles",
        yaxis_title="Sample Quantiles",
        showlegend=True,
    )
    return fig


# ── Per-Class Report ───────────────────────────────────────────────────────────

def build_per_class_report_chart(
    y_true, y_pred,
    labels: Optional[List] = None,
) -> go.Figure:
    """Heatmap-style per-class precision/recall/f1 report."""
    try:
        report = classification_report(y_true, y_pred, output_dict=True,
                                       zero_division=0, labels=labels)
        classes = [k for k in report if k not in ("accuracy", "macro avg", "weighted avg")]
        metrics = ["precision", "recall", "f1-score"]

        z = [[report[cls][m] for cls in classes] for m in metrics]
        text_z = [[f"{report[cls][m]:.3f}" for cls in classes] for m in metrics]

        fig = go.Figure(go.Heatmap(
            z=z, x=classes, y=metrics,
            text=text_z, texttemplate="%{text}",
            colorscale=[[0, "#1e293b"], [0.5, "#4f46e5"], [1.0, "#10b981"]],
            zmin=0, zmax=1,
            showscale=True,
        ))
        fig.update_layout(
            **_base_layout("📋 Per-Class Metrics", 260),
            xaxis_title="Class",
        )
        return fig
    except Exception:
        return go.Figure()


# ── Imbalance detection helper ─────────────────────────────────────────────────

def check_imbalance(y) -> Tuple[bool, float]:
    """Returns (is_imbalanced, minority_ratio)."""
    try:
        from collections import Counter
        counts = Counter(y)
        if len(counts) < 2:
            return False, 1.0
        vals = sorted(counts.values())
        ratio = vals[0] / vals[-1]
        return ratio < 0.3, round(ratio, 4)
    except Exception:
        return False, 1.0


# ── Render Wrappers for Streamlit ─────────────────────────────────────────────

def render_calibration_curve(fraction_of_positives, mean_predicted, pipeline_id="Model"):
    """Render calibration curve chart."""
    fig = build_calibration_chart(fraction_of_positives, mean_predicted, pipeline_id)
    st.plotly_chart(fig, use_container_width=True)


def render_learning_curve(train_sizes, train_mean, train_std, val_mean, val_std):
    """Render learning curve chart."""
    fig = build_learning_curve_chart(train_sizes, train_mean, train_std, val_mean, val_std)
    st.plotly_chart(fig, use_container_width=True)


def render_residuals_plot(y_pred, residuals):
    """Render residuals analysis charts (scatter + QQ)."""
    col1, col2 = st.columns(2)
    with col1:
        fig1 = build_residuals_chart(y_pred, residuals)
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        fig2 = build_qq_chart(residuals)
        st.plotly_chart(fig2, use_container_width=True)


def render_per_class_metrics(y_true, y_pred, labels=None):
    """Render per-class report heatmap."""
    fig = build_per_class_report_chart(y_true, y_pred, labels)
    st.plotly_chart(fig, use_container_width=True)
