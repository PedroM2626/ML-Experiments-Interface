"""
Relationship Map — semicircle visualization of algorithms, pipelines, and transformers.
Inspired by IBM AutoAI's relationship map.
"""

import math
import numpy as np
import plotly.graph_objects as go
from typing import List, Dict


# Predefined positions
N_OUTER = 20   # feature transformer slots
N_MID = 14     # pipeline slots
N_INNER = 8    # algorithm slots

OUTER_RADIUS = 1.0
MID_RADIUS = 0.67
INNER_RADIUS = 0.38


def _semicircle_positions(n: int, radius: float, start_angle=0, end_angle=180):
    """Generate (x, y) positions along a semicircle arc."""
    angles = np.linspace(math.radians(start_angle), math.radians(end_angle), n)
    xs = [radius * math.cos(a) for a in angles]
    ys = [radius * math.sin(a) for a in angles]
    return list(zip(xs, ys))


def build_relationship_map(
    pipelines: List[Dict],
    theme: str = "dark",
) -> go.Figure:
    """
    Build the relationship map semicircle.
    pipelines: list of dicts with pipeline_id, algorithm, color, nodes_done
    """
    bg_color = "#0f0f1a" if theme == "dark" else "#ffffff"
    text_color = "#e2e8f0" if theme == "dark" else "#1a202c"
    inactive_color = "#1e293b" if theme == "dark" else "#e2e8f0"
    arc_color = "#1e293b" if theme == "dark" else "#e2e8f0"

    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        height=320,
        margin=dict(l=10, r=10, t=40, b=10),
        showlegend=False,
        xaxis=dict(visible=False, range=[-1.3, 1.3]),
        yaxis=dict(visible=False, range=[-0.2, 1.25]),
        font=dict(color=text_color, family="Inter, sans-serif"),
    )

    # ── Draw arc backgrounds ───────────────────────────────────────────────────
    for radius, label, r_frac in [
        (OUTER_RADIUS, "FEATURE TRANSFORMERS", 1.15),
        (MID_RADIUS, "PIPELINES", 0.80),
        (INNER_RADIUS, "TOP ALGORITHMS", 0.48),
    ]:
        theta = np.linspace(0, math.pi, 60)
        arc_x = [radius * math.cos(t) for t in theta]
        arc_y = [radius * math.sin(t) for t in theta]
        fig.add_trace(go.Scatter(
            x=arc_x, y=arc_y,
            mode="lines",
            line=dict(color=arc_color, width=1, dash="dot"),
        ))
        # Label
        fig.add_annotation(
            x=0, y=r_frac * OUTER_RADIUS,
            text=label,
            showarrow=False,
            font=dict(size=8, color="#64748b"),
        )

    # ── Outer ring — feature transformers ─────────────────────────────────────
    outer_positions = _semicircle_positions(N_OUTER, OUTER_RADIUS)
    for i, (ox, oy) in enumerate(outer_positions):
        fig.add_trace(go.Scatter(
            x=[ox], y=[oy],
            mode="markers",
            marker=dict(size=8, color=inactive_color, line=dict(color="#334155", width=1)),
        ))

    # ── Middle ring — pipelines ────────────────────────────────────────────────
    mid_positions = _semicircle_positions(N_MID, MID_RADIUS)
    # Fill slots matching completed pipelines
    done_pipelines = [p for p in pipelines if p.get("nodes_done", 0) >= 2]

    for i, (mx, my) in enumerate(mid_positions):
        if i < len(done_pipelines):
            p = done_pipelines[i]
            color = p.get("color", "#8B5CF6")
            size = 14
            # Best pipeline gets a ring
            is_best = i == 0 and p.get("nodes_done", 0) == 4
            if is_best:
                fig.add_trace(go.Scatter(
                    x=[mx], y=[my],
                    mode="markers",
                    marker=dict(size=22, color=color, opacity=0.2),
                ))
            fig.add_trace(go.Scatter(
                x=[mx], y=[my],
                mode="markers",
                marker=dict(
                    size=size,
                    color=color,
                    line=dict(color="white", width=2 if is_best else 0),
                ),
                hovertext=f"{p['pipeline_id']} — {p['algorithm']}",
                hoverinfo="text",
            ))
            # Connect to inner ring
            fig.add_trace(go.Scatter(
                x=[mx, 0], y=[my, 0],
                mode="lines",
                line=dict(color=color, width=0.5, dash="dot"),
            ))
        else:
            fig.add_trace(go.Scatter(
                x=[mx], y=[my],
                mode="markers",
                marker=dict(size=10, color=inactive_color, line=dict(color="#334155", width=1)),
            ))

    # ── Inner ring — algorithms ────────────────────────────────────────────────
    inner_positions = _semicircle_positions(N_INNER, INNER_RADIUS)
    # Unique algorithms seen
    seen_algos = {}
    for p in pipelines:
        algo = p.get("algorithm", "")
        if algo and algo not in seen_algos:
            seen_algos[algo] = p.get("color", "#8B5CF6")

    algo_list = list(seen_algos.items())
    for i, (ix, iy) in enumerate(inner_positions):
        if i < len(algo_list):
            algo_name, color = algo_list[i]
            fig.add_trace(go.Scatter(
                x=[ix], y=[iy],
                mode="markers+text",
                marker=dict(size=18, color=color, line=dict(color="white", width=1)),
                text=[algo_name[:4]],
                textposition="bottom center",
                textfont=dict(size=7, color="white"),
                hovertext=algo_name,
                hoverinfo="text",
            ))
        else:
            fig.add_trace(go.Scatter(
                x=[ix], y=[iy],
                mode="markers",
                marker=dict(size=12, color=inactive_color, line=dict(color="#334155", width=1)),
            ))

    # Center connector dot
    fig.add_trace(go.Scatter(
        x=[0], y=[0],
        mode="markers",
        marker=dict(size=6, color="#334155"),
    ))

    return fig
