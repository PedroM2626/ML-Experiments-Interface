"""
Progress Map — dynamic per task-type, per-algorithm branching graph.
Adapts stages and node labels based on task type (classification / regression / time_series).
Algorithm-specific node symbols distinguish tree, linear, ensemble, and TS models.
"""

import math
import plotly.graph_objects as go
import numpy as np
from typing import List, Dict, Optional

# ── Task-specific stage definitions ───────────────────────────────────────────

_STAGES = {
    "classification": [
        "Read data", "Split holdout", "Preprocessing", "Feature sel.", "Train",
    ],
    "regression": [
        "Read data", "Split holdout", "Preprocessing", "Target transf.", "Train",
    ],
    "time_series": [
        "Read data", "Date features", "Walk-fwd split", "Model select.", "Forecast",
    ],
}

_NODE_STEPS = {
    "classification": ["Algorithm", "HPO", "Threshold", "Evaluate"],
    "regression":     ["Algorithm", "HPO", "Transform",  "Evaluate"],
    "time_series":    ["Algorithm", "Walk-fwd CV", "HPO",       "Forecast"],
}

# ── Algorithm → Plotly marker symbol ──────────────────────────────────────────

_TREE_ALGOS  = {"XGBoost", "LightGBM", "GradientBoosting", "RandomForest", "ExtraTrees", "CatBoost"}
_LINEAR_ALGOS = {"LogisticRegression", "Ridge", "ElasticNet", "Lasso", "LinearSVC"}
_ENSEMBLE_ALGOS = {"StackedEnsemble", "P_Ens", "Stacking"}
_TS_ALGOS = {"ARIMA", "SARIMA", "Prophet"}

def _algo_symbol(algo: str) -> str:
    if algo in _ENSEMBLE_ALGOS or "Ens" in algo:
        return "diamond"
    if algo in _TREE_ALGOS:
        return "triangle-up"
    if algo in _LINEAR_ALGOS:
        return "circle"
    if algo in _TS_ALGOS:
        return "star"
    return "circle"


def _algo_label(algo: str) -> str:
    """Short label for the pipeline branch header."""
    _SHORTS = {
        "XGBoost": "XGB", "LightGBM": "LGB", "GradientBoosting": "GB",
        "RandomForest": "RF", "ExtraTrees": "ET", "LogisticRegression": "LR",
        "Ridge": "Ridge", "ElasticNet": "EN", "LinearSVC": "LSVC",
        "StackedEnsemble": "Stack", "ARIMA": "ARIMA", "Prophet": "Prophet",
        "CatBoost": "CB",
    }
    for k, v in _SHORTS.items():
        if k in algo:
            return v
    return algo[:6]


STAGE_X = [0, 1.5, 3.0, 4.5, 6.0]
BASE_Y = 0.0
PIPELINE_Y_OFFSET = 1.8
BRANCH_X_OFFSET = 0.5


def _pipeline_y(idx: int) -> float:
    if idx % 2 == 0:
        return PIPELINE_Y_OFFSET + (idx // 2) * 0.9
    else:
        return -(PIPELINE_Y_OFFSET + (idx // 2) * 0.9)


def build_progress_map(
    completed_stages: List[str],
    pipelines: List[Dict],
    running_pipeline_id: Optional[str] = None,
    best_pipeline_id: Optional[str] = None,
    task_type: str = "classification",
    theme: str = "dark",
) -> go.Figure:
    """
    Dynamic AutoAI-style progress map that adapts to task type and algorithm.

    pipelines: list of dicts with keys:
        pipeline_id, algorithm, color, nodes_done (0-4)
    """
    bg_color   = "#0f0f1a" if theme == "dark" else "#ffffff"
    text_color = "#e2e8f0" if theme == "dark" else "#1a202c"
    line_color = "#334155" if theme == "dark" else "#cbd5e0"
    done_node  = "#1e293b" if theme == "dark" else "#f1f5f9"

    # Resolve stages for this task type
    base_stages = _STAGES.get(task_type, _STAGES["classification"])
    node_steps  = _NODE_STEPS.get(task_type, _NODE_STEPS["classification"])

    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor=bg_color, plot_bgcolor=bg_color,
        height=460,
        margin=dict(l=20, r=20, t=30, b=20),
        showlegend=False,
        xaxis=dict(visible=False, range=[-0.5, STAGE_X[-1] + 4.5]),
        yaxis=dict(visible=False, range=[-3.8, 3.8]),
        font=dict(color=text_color, family="Inter, sans-serif"),
    )

    # ── Base pipeline line ─────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=STAGE_X, y=[BASE_Y] * len(STAGE_X),
        mode="lines", line=dict(color=line_color, width=2),
    ))

    # ── Base stage nodes ───────────────────────────────────────────────────────
    for i, (stage, sx) in enumerate(zip(base_stages, STAGE_X)):
        done = stage.replace(" ", "").lower() in [s.replace(" ", "").lower() for s in completed_stages]
        fill    = "#6366F1" if done else done_node
        outline = "#6366F1" if done else line_color

        fig.add_trace(go.Scatter(
            x=[sx], y=[BASE_Y],
            mode="markers+text",
            marker=dict(size=16, color=fill, line=dict(color=outline, width=2), symbol="circle"),
            text=[stage], textposition="bottom center",
            textfont=dict(size=9, color=text_color),
        ))

    if not pipelines:
        return fig

    branch_x = STAGE_X[-1]

    for p_idx, pipeline in enumerate(pipelines):
        pid        = pipeline.get("pipeline_id", f"P{p_idx+1}")
        algo       = pipeline.get("algorithm", "")
        color      = pipeline.get("color", "#8B5CF6")
        nodes_done = pipeline.get("nodes_done", 0)
        is_running = pid == running_pipeline_id
        is_best    = (pid == best_pipeline_id) if best_pipeline_id else (p_idx == 0 and nodes_done == 4)
        symbol     = _algo_symbol(algo)
        short_algo = _algo_label(algo)

        py = _pipeline_y(p_idx)

        # Bezier-like branch curve
        ctrl_x = [branch_x, branch_x + BRANCH_X_OFFSET, branch_x + BRANCH_X_OFFSET * 2]
        ctrl_y = [BASE_Y, BASE_Y + (py - BASE_Y) * 0.4, py]
        fig.add_trace(go.Scatter(
            x=ctrl_x, y=ctrl_y, mode="lines",
            line=dict(color=color if nodes_done > 0 else line_color,
                      width=2, dash="dot" if nodes_done == 0 else "solid"),
        ))

        node_xs = [branch_x + BRANCH_X_OFFSET * 2 + i * 1.3 for i in range(4)]
        node_ys = [py] * 4

        # Horizontal connector between done nodes
        if nodes_done > 1:
            done_end_x = node_xs[min(nodes_done - 1, 3)]
            fig.add_trace(go.Scatter(
                x=[node_xs[0], done_end_x], y=[py, py],
                mode="lines", line=dict(color=color, width=2),
            ))

        for n_idx, (nx, ny, label) in enumerate(zip(node_xs, node_ys, node_steps)):
            is_done_node    = n_idx < nodes_done
            is_current_node = n_idx == nodes_done and is_running

            # Pulsing ring for running node
            if is_current_node:
                for sz, alpha in [(28, 0.1), (22, 0.2)]:
                    fig.add_trace(go.Scatter(
                        x=[nx], y=[ny], mode="markers",
                        marker=dict(size=sz, color=color, opacity=alpha),
                    ))

            node_fill = color if is_done_node else bg_color
            node_line = color if (is_done_node or is_current_node) else line_color
            lw = 3 if is_current_node else 2

            fig.add_trace(go.Scatter(
                x=[nx], y=[ny],
                mode="markers+text",
                marker=dict(
                    size=16, color=node_fill,
                    line=dict(color=node_line, width=lw),
                    symbol=symbol,
                ),
                text=[pid if n_idx == 0 else ""],
                textposition="top center",
                textfont=dict(size=9, color=color if (is_done_node or is_current_node) else text_color),
                hovertext=f"{pid} — {algo} — {label}",
                hoverinfo="text",
            ))

            # Sub-label below node
            sub = short_algo if n_idx == 0 else label
            fig.add_trace(go.Scatter(
                x=[nx], y=[ny - 0.3], mode="text",
                text=[sub],
                textfont=dict(size=8, color=color if (is_done_node or is_current_node) else "#64748b"),
            ))

        # ⭐ Best badge — dynamic: highlights the actual best pipeline
        if is_best and nodes_done == 4:
            fig.add_annotation(
                x=node_xs[-1] + 0.4, y=py + 0.3,
                text="⭐ Best",
                showarrow=False,
                font=dict(size=10, color="#FBBF24"),
                bgcolor="#1e293b", bordercolor="#FBBF24",
                borderwidth=1, borderpad=4,
            )

    # Ensemble special badge
    ens = next((p for p in pipelines if "Ens" in p.get("pipeline_id", "")), None)
    if ens and ens.get("nodes_done", 0) == 4:
        fig.add_annotation(
            x=0.98, y=0.05, xref="paper", yref="paper",
            text="🥇 Ensemble Ready",
            showarrow=False,
            font=dict(size=11, color="#FBBF24"),
            bgcolor="#1e293b", bordercolor="#FBBF24",
            borderwidth=1, borderpad=6,
        )

    # "Experiment complete" annotation
    all_done = all(p.get("nodes_done", 0) == 4 for p in pipelines) and len(pipelines) > 0
    if all_done:
        fig.add_annotation(
            x=0.98, y=0.98, xref="paper", yref="paper",
            text="✅ Experiment complete",
            showarrow=False,
            font=dict(size=13, color="#10B981"),
            bgcolor="#064e3b", bordercolor="#10B981",
            borderwidth=1, borderpad=6,
        )

    return fig
