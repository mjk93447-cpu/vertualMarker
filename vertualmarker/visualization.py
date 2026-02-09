"""시각화 모듈."""
from typing import List

import matplotlib.pyplot as plt

from .geometry import Point
from .strategy2 import Strategy2Result


def visualize_result(
    original_points: List[Point],
    result: Strategy2Result,
    out_path: str,
) -> None:
    """거북이 선과 bending 포인트를 시각화.

    - Original points: 점으로 표시 (연한 회색)
    - Turtle line: 파란색 선
    - Bending points: 인덱스별 색상
    - TLSP, BSP, Mv, Mv_shifted, 앞머리/윗머리 구간: 하이라이트
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # 원본 점들 (연한 회색)
    if original_points:
        ox = [p[0] for p in original_points]
        oy = [p[1] for p in original_points]
        ax.scatter(ox, oy, color="lightgray", s=1, alpha=0.3, label="Original points")

    # 거북이 선
    tl = result.turtle_line_path
    if len(tl) >= 2:
        xs = [p[0] for p in tl]
        ys = [p[1] for p in tl]
        ax.plot(xs, ys, color="blue", linewidth=2, label="Turtle line", alpha=0.7)

    # Bending points (인덱스별 색상)
    if result.bending_points:
        bx = [p[0] for p in result.bending_points]
        by = [p[1] for p in result.bending_points]
        indices = list(range(1, len(result.bending_points) + 1))
        sc = ax.scatter(
            bx,
            by,
            c=indices,
            cmap="viridis",
            s=20,
            label="Bending points (1..PBL)",
            zorder=5,
        )
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("Index")

    # 핵심 포인트들
    ax.scatter(
        [result.tlsp[0]],
        [result.tlsp[1]],
        color="red",
        s=100,
        marker="*",
        label="TLSP",
        zorder=10,
        edgecolors="black",
        linewidths=1,
    )
    ax.scatter(
        [result.bsp[0]],
        [result.bsp[1]],
        color="orange",
        s=100,
        marker="*",
        label="BSP",
        zorder=10,
        edgecolors="black",
        linewidths=1,
    )
    ax.scatter(
        [result.mv[0]],
        [result.mv[1]],
        color="purple",
        s=80,
        marker="o",
        label="Mv",
        zorder=10,
        edgecolors="black",
        linewidths=1,
    )
    ax.scatter(
        [result.mv_shifted[0]],
        [result.mv_shifted[1]],
        color="magenta",
        s=80,
        marker="x",
        label="Mv shifted",
        zorder=10,
        linewidths=2,
    )

    # 앞머리 구간 (세로)
    if result.front_head_run:
        fh_x = [p[0] for p in result.front_head_run]
        fh_y = [p[1] for p in result.front_head_run]
        ax.plot(fh_x, fh_y, color="green", linewidth=3, label="Front head (FH)", zorder=8)

    # 윗머리 구간 (가로)
    if result.upper_head_run:
        uh_x = [p[0] for p in result.upper_head_run]
        uh_y = [p[1] for p in result.upper_head_run]
        ax.plot(uh_x, uh_y, color="brown", linewidth=3, label="Upper head (UH)", zorder=8)

    ax.set_aspect("equal", adjustable="datalim")
    ax.invert_yaxis()  # 이미지 좌표계 (y가 아래로 증가)
    ax.set_xlabel("X (pixel)")
    ax.set_ylabel("Y (pixel)")
    ax.set_title("Strategy 2 - Turtle Line & Bending Points")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
