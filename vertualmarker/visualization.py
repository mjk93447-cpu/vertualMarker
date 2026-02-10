"""Visualization module for Strategy 2 outputs.

ver7 focus:
- Improve overlap readability by shrinking critical markers and using transparency.
- Keep a clean, professional style with subtle gray grid lines for coordinate comparison.
- Minimize non-essential decoration while preserving interpretability.
"""
from typing import List

import matplotlib.pyplot as plt

from .geometry import Point
from .strategy2 import Strategy2Result


def visualize_result(
    original_points: List[Point],
    result: Strategy2Result,
    out_path: str,
) -> None:
    """Visualize turtle-line geometry and indexed bending trajectory."""
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_facecolor("#FAFAFB")

    # Original edge points: keep very light so structure remains visible.
    if original_points:
        ox = [p[0] for p in original_points]
        oy = [p[1] for p in original_points]
        ax.scatter(
            ox,
            oy,
            color="#B6BBC6",
            s=0.55,
            alpha=0.18,
            linewidths=0.0,
            label="Original edge points",
            zorder=1,
        )

    # Turtle line: thin but crisp path guide.
    tl = result.turtle_line_path
    if len(tl) >= 2:
        xs = [p[0] for p in tl]
        ys = [p[1] for p in tl]
        ax.plot(
            xs,
            ys,
            color="#2B6CB0",
            linewidth=0.72,
            label="Turtle line",
            alpha=0.88,
            zorder=3,
        )

    # Bending points: use modern high-contrast spectral map with transparent markers.
    if result.bending_points:
        bx = [p[0] for p in result.bending_points]
        by = [p[1] for p in result.bending_points]
        indices = list(range(1, len(result.bending_points) + 1))
        sc = ax.scatter(
            bx,
            by,
            c=indices,
            cmap="turbo",
            s=4.5,
            alpha=0.72,
            label="Bending points (1..PBL)",
            linewidths=0.0,
            zorder=5,
            rasterized=True,
        )
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("Bending index")
        cbar.ax.tick_params(labelsize=8, colors="#4B5563")

    # Critical key points: make symbols compact so lines remain visible on overlap.
    ax.scatter(
        [result.tlsp[0]],
        [result.tlsp[1]],
        color="#C62828",
        s=19,
        alpha=0.82,
        marker="v",
        label="TLSP",
        zorder=10,
        edgecolors="none",
        linewidths=0.0,
    )
    ax.annotate(
        "TLSP",
        xy=result.tlsp,
        xytext=(8, -10),
        textcoords="offset points",
        fontsize=7,
        color="#A61B1B",
        weight="semibold",
        alpha=0.92,
        bbox={"boxstyle": "round,pad=0.12", "fc": "#FAFAFB", "ec": "none", "alpha": 0.72},
    )

    ax.scatter(
        [result.bsp[0]],
        [result.bsp[1]],
        color="#DD6B20",
        s=19,
        alpha=0.82,
        marker="D",
        label="BSP",
        zorder=10,
        edgecolors="none",
        linewidths=0.0,
    )
    ax.annotate(
        "BSP",
        xy=result.bsp,
        xytext=(12, -1),
        textcoords="offset points",
        fontsize=7,
        color="#B45309",
        weight="semibold",
        alpha=0.92,
        bbox={"boxstyle": "round,pad=0.12", "fc": "#FAFAFB", "ec": "none", "alpha": 0.72},
    )

    ax.scatter(
        [result.mv[0]],
        [result.mv[1]],
        color="#6B46C1",
        s=18,
        alpha=0.82,
        marker="P",
        label="Mv",
        zorder=10,
        edgecolors="none",
        linewidths=0.0,
    )
    ax.annotate(
        "Mv",
        xy=result.mv,
        xytext=(8, 10),
        textcoords="offset points",
        fontsize=7,
        color="#5B21B6",
        weight="semibold",
        alpha=0.92,
        bbox={"boxstyle": "round,pad=0.12", "fc": "#FAFAFB", "ec": "none", "alpha": 0.72},
    )

    ax.scatter(
        [result.mv_shifted[0]],
        [result.mv_shifted[1]],
        color="#C026D3",
        s=15,
        alpha=0.78,
        marker="x",
        label="Mv shifted",
        zorder=10,
        linewidths=0.95,
    )
    ax.annotate(
        "Mv'",
        xy=result.mv_shifted,
        xytext=(-24, 8),
        textcoords="offset points",
        fontsize=7,
        color="#A21CAF",
        weight="semibold",
        alpha=0.9,
        bbox={"boxstyle": "round,pad=0.12", "fc": "#FAFAFB", "ec": "none", "alpha": 0.72},
    )

    # FH/UH detected runs: keep visible but not dominant.
    if result.front_head_run:
        fh_x = [p[0] for p in result.front_head_run]
        fh_y = [p[1] for p in result.front_head_run]
        ax.plot(
            fh_x,
            fh_y,
            color="#2E8B57",
            linewidth=0.98,
            alpha=0.9,
            label="Front head (FH)",
            zorder=8,
        )

    if result.upper_head_run:
        uh_x = [p[0] for p in result.upper_head_run]
        uh_y = [p[1] for p in result.upper_head_run]
        ax.plot(
            uh_x,
            uh_y,
            color="#6D4C41",
            linewidth=0.98,
            alpha=0.9,
            label="Upper head (UH)",
            zorder=8,
        )

    ax.set_aspect("equal", adjustable="datalim")
    ax.invert_yaxis()  # Image coordinates: y increases downward.
    ax.set_xlabel("X (pixel)")
    ax.set_ylabel("Y (pixel)")
    ax.set_title("Strategy 2 - Turtle Geometry and Indexed Trajectory", fontsize=11)
    ax.minorticks_on()
    ax.grid(
        which="major",
        color="#D7DBE3",
        linestyle="-",
        linewidth=0.55,
        alpha=0.38,
    )
    ax.grid(
        which="minor",
        color="#E8EBF1",
        linestyle="-",
        linewidth=0.42,
        alpha=0.26,
    )
    ax.legend(
        loc="best",
        fontsize=7.5,
        framealpha=0.88,
        facecolor="#FCFCFD",
        edgecolor="#D3D7DE",
    )
    for spine in ax.spines.values():
        spine.set_color("#C5CAD3")
        spine.set_linewidth(0.8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
