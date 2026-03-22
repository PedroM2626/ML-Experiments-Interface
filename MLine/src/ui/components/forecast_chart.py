"""
Forecast Chart — interactive Plotly visualization for time series experiments.
Shows actual vs forecast with 95% confidence bands.
"""

import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import Optional, List


DARK_BG = "#0a0a14"
GRID_COLOR = "#1e293b"
ACTUAL_COLOR = "#06B6D4"
FORECAST_COLOR = "#8B5CF6"
CI_COLOR = "rgba(139, 92, 246, 0.15)"


def build_forecast_chart(
    y_actual: pd.Series,
    y_forecast: Optional[np.ndarray] = None,
    y_lower: Optional[np.ndarray] = None,
    y_upper: Optional[np.ndarray] = None,
    date_index: Optional[pd.Series] = None,
    target_col: str = "target",
    horizon: int = 12,
) -> go.Figure:
    """
    Build a forecast vs actuals chart.

    Parameters
    ----------
    y_actual : full historical series
    y_forecast : forecast values (last `horizon` steps)
    y_lower/upper : confidence interval bounds
    date_index : optional datetime index for x-axis labels
    """
    fig = go.Figure()

    n = len(y_actual)
    x_all = date_index.tolist() if date_index is not None else list(range(n))

    # --- Historical actuals ---
    fig.add_trace(go.Scatter(
        x=x_all,
        y=y_actual.values,
        mode="lines",
        name="Actual",
        line=dict(color=ACTUAL_COLOR, width=2),
    ))

    # --- Forecast ---
    if y_forecast is not None:
        x_fc = x_all[-horizon:] if len(x_all) >= horizon else x_all
        n_fc = len(x_fc)
        y_fc = y_forecast[:n_fc] if len(y_forecast) >= n_fc else y_forecast

        # Confidence band
        if y_lower is not None and y_upper is not None:
            y_lo = y_lower[:n_fc]
            y_hi = y_upper[:n_fc]
            fig.add_trace(go.Scatter(
                x=x_fc + x_fc[::-1],
                y=list(y_hi) + list(y_lo[::-1]),
                fill="toself",
                fillcolor=CI_COLOR,
                line=dict(color="rgba(255,255,255,0)"),
                name="95% CI",
                showlegend=True,
                hoverinfo="skip",
            ))

        fig.add_trace(go.Scatter(
            x=x_fc,
            y=y_fc,
            mode="lines+markers",
            name="Forecast",
            line=dict(color=FORECAST_COLOR, width=2.5, dash="dot"),
            marker=dict(size=5, color=FORECAST_COLOR, symbol="diamond"),
        ))

        # Vertical marker separating history from forecast
        x_sep = x_fc[0]
        fig.add_vline(
            x=x_sep if not isinstance(x_sep, int) else x_sep,
            line=dict(color="#334155", width=1.5, dash="dash"),
            annotation_text="Forecast start",
            annotation_font=dict(color="#64748b", size=10),
        )

    fig.update_layout(
        plot_bgcolor=DARK_BG,
        paper_bgcolor=DARK_BG,
        font=dict(color="#94a3b8"),
        margin=dict(t=30, b=40, l=40, r=20),
        height=320,
        legend=dict(
            bgcolor="#0f1929",
            bordercolor="#1e293b",
            borderwidth=1,
            font=dict(size=11),
            orientation="h",
            y=1.05,
        ),
        xaxis=dict(
            gridcolor=GRID_COLOR,
            showgrid=True,
            title=dict(text="Time", font=dict(size=11, color="#64748b")),
            color="#64748b",
        ),
        yaxis=dict(
            gridcolor=GRID_COLOR,
            showgrid=True,
            title=dict(text=target_col, font=dict(size=11, color="#64748b")),
            color="#64748b",
        ),
        hovermode="x unified",
    )

    return fig


def build_ts_leaderboard_chart(results: list, metric_key: str = "rmse") -> go.Figure:
    """Bar chart of pipelines sorted by TS metric (ascending for error metrics)."""
    if not results:
        fig = go.Figure()
        fig.update_layout(
            plot_bgcolor=DARK_BG, paper_bgcolor=DARK_BG,
            annotations=[dict(text="No results yet", x=0.5, y=0.5,
                              font=dict(color="#475569", size=14), showarrow=False)],
            height=200,
        )
        return fig

    sorted_results = sorted(results, key=lambda r: abs(r.get("primary_metric_cv", 0)), reverse=True)
    labels = [r["pipeline_id"] for r in sorted_results]
    values = [abs(r.get("primary_metric_cv", 0)) for r in sorted_results]
    colors = [r.get("color", "#8B5CF6") for r in sorted_results]

    fig = go.Figure(go.Bar(
        y=labels, x=values,
        orientation="h",
        marker=dict(color=colors, opacity=0.85),
        text=[f"{v:.4f}" for v in values],
        textposition="outside",
        textfont=dict(size=10, color="#94a3b8"),
    ))
    fig.update_layout(
        plot_bgcolor=DARK_BG, paper_bgcolor=DARK_BG,
        font=dict(color="#94a3b8"),
        margin=dict(t=10, b=10, l=50, r=60),
        height=max(180, len(results) * 32),
        xaxis=dict(gridcolor=GRID_COLOR, showgrid=True, color="#64748b",
                   title=dict(text=metric_key.upper(), font=dict(size=10))),
        yaxis=dict(color="#64748b"),
    )
    return fig
