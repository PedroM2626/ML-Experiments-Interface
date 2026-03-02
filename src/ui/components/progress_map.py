"""
Progress Map — animated pipeline graph à la IBM AutoAI.
"""

import math
import plotly.graph_objects as go
import numpy as np
from typing import List, Dict, Optional

BASE_STAGES = [
    "Read dataset",
    "Split holdout",
    "Read training",
    "Preprocessing",
    "Model selection",
]

STAGE_X = [0, 1.5, 3.0, 4.5, 6.0]
BASE_Y = 0.0
PIPELINE_Y_OFFSET = 1.8
BRANCH_X_OFFSET = 0.5
NODE_STEPS = ["Algorithm", "HPO", "Feature Eng.", "HPO+"]


def _pipeline_y(idx: int) -> float:
    """Alternate pipelines above/below center for a branching visual."""
    if idx % 2 == 0:
        return PIPELINE_Y_OFFSET + (idx // 2) * 0.9
    else:
        return -(PIPELINE_Y_OFFSET + (idx // 2) * 0.9)


def build_progress_map(
    completed_stages: List[str],
    pipelines: List[Dict],
    running_pipeline_id: Optional[str] = None,
    theme: str = "dark",
) -> go.Figure:
    """
    Build the AutoAI-style progress map.

    pipelines: list of dicts with keys:
        pipeline_id, algorithm, color, nodes_done (0-4)
    """
    bg_color = "#0f0f1a" if theme == "dark" else "#ffffff"
    text_color = "#e2e8f0" if theme == "dark" else "#1a202c"
    line_color = "#334155" if theme == "dark" else "#cbd5e0"
    done_node_color = "#1e293b" if theme == "dark" else "#f1f5f9"

    fig = go.Figure()

    fig.update_layout(
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        height=460,
        margin=dict(l=20, r=20, t=30, b=20),
        showlegend=False,
        xaxis=dict(visible=False, range=[-0.5, STAGE_X[-1] + 4.5]),
        yaxis=dict(visible=False, range=[-3.8, 3.8]),
        font=dict(color=text_color, family="Inter, sans-serif"),
    )

    # ── Base pipeline horizontal line ──────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=STAGE_X, y=[BASE_Y] * len(STAGE_X),
        mode="lines",
        line=dict(color=line_color, width=2),
    ))

    # ── Base stage nodes + labels ──────────────────────────────────────────────
    for i, (stage, sx) in enumerate(zip(BASE_STAGES, STAGE_X)):
        done = stage.replace(" ", "").lower() in [s.replace(" ", "").lower() for s in completed_stages]
        fill = "#6366F1" if done else done_node_color
        outline = "#6366F1" if done else line_color

        fig.add_trace(go.Scatter(
            x=[sx], y=[BASE_Y],
            mode="markers+text",
            marker=dict(size=16, color=fill, line=dict(color=outline, width=2)),
            text=[stage], textposition="bottom center",
            textfont=dict(size=10, color=text_color),
        ))

    if not pipelines:
        return fig

    # ── Pipeline branches ──────────────────────────────────────────────────────
    branch_x = STAGE_X[-1]  # where pipelines branch off

    for p_idx, pipeline in enumerate(pipelines):
        pid = pipeline.get("pipeline_id", f"P{p_idx+1}")
        algo = pipeline.get("algorithm", "")
        color = pipeline.get("color", "#8B5CF6")
        nodes_done = pipeline.get("nodes_done", 0)  # 0-4
        is_running = pid == running_pipeline_id

        py = _pipeline_y(p_idx)

        # Curve from base to first node
        # Bezier-like with intermediate points
        ctrl_x = [branch_x, branch_x + BRANCH_X_OFFSET, branch_x + BRANCH_X_OFFSET * 2]
        ctrl_y = [BASE_Y, BASE_Y + (py - BASE_Y) * 0.4, py]

        fig.add_trace(go.Scatter(
            x=ctrl_x, y=ctrl_y,
            mode="lines",
            line=dict(color=color if nodes_done > 0 else line_color, width=2, dash="dot" if nodes_done == 0 else "solid"),
        ))

        # Node positions along the branch
        node_xs = [branch_x + BRANCH_X_OFFSET * 2 + i * 1.3 for i in range(4)]
        node_ys = [py] * 4

        # Horizontal line connecting nodes
        if nodes_done > 1:
            done_end_x = node_xs[min(nodes_done - 1, 3)]
            fig.add_trace(go.Scatter(
                x=[node_xs[0], done_end_x], y=[py, py],
                mode="lines",
                line=dict(color=color, width=2),
            ))

        # Draw each step node
        for n_idx, (nx, ny, label) in enumerate(zip(node_xs, node_ys, NODE_STEPS)):
            step_label = f"{pid}" if n_idx == 0 else ""
            sub_label = algo if n_idx == 0 else label

            is_done_node = n_idx < nodes_done
            is_current_node = n_idx == nodes_done and is_running

            if is_current_node:
                # Pulsing ring effect — multiple circles
                for sz, alpha in [(28, 0.1), (22, 0.2)]:
                    fig.add_trace(go.Scatter(
                        x=[nx], y=[ny],
                        mode="markers",
                        marker=dict(size=sz, color=color, opacity=alpha),
                    ))

            node_fill = color if is_done_node else (bg_color if not is_current_node else bg_color)
            node_line_color = color if (is_done_node or is_current_node) else line_color
            node_size = 16

            fig.add_trace(go.Scatter(
                x=[nx], y=[ny],
                mode="markers+text",
                marker=dict(
                    size=node_size,
                    color=node_fill,
                    line=dict(color=node_line_color, width=2 if not is_current_node else 3),
                    symbol="circle",
                ),
                text=[step_label],
                textposition="top center",
                textfont=dict(size=9, color=color if is_done_node or is_current_node else text_color),
                hovertext=f"{pid} — {sub_label} — {label}",
                hoverinfo="text",
            ))

            # Sub-label below node
            fig.add_trace(go.Scatter(
                x=[nx], y=[ny - 0.3],
                mode="text",
                text=[sub_label if n_idx == 0 else label],
                textfont=dict(size=8, color=color if is_done_node or is_current_node else "#64748b"),
            ))

        # Best badge if first pipeline
        if p_idx == 0 and nodes_done == 4:
            fig.add_annotation(
                x=node_xs[-1] + 0.4, y=py + 0.3,
                text="⭐ Best",
                showarrow=False,
                font=dict(size=10, color="#FBBF24"),
                bgcolor="#1e293b",
                bordercolor="#FBBF24",
                borderwidth=1,
                borderpad=4,
            )

    # "Experiment complete" annotation
    all_done = all(p.get("nodes_done", 0) == 4 for p in pipelines) and len(pipelines) > 0
    if all_done:
        fig.add_annotation(
            x=0.98, y=0.98,
            xref="paper", yref="paper",
            text="✅ Experiment complete",
            showarrow=False,
            font=dict(size=13, color="#10B981"),
            bgcolor="#064e3b",
            bordercolor="#10B981",
            borderwidth=1,
            borderpad=6,
        )

    return fig
