"""시각화 모듈.

ver5 방향:
- 선/점 두께를 얇게 유지해 edge 형태를 더 정확히 관찰
- 색상은 고대비 스펙트럼(turbo) 기반으로 인덱스 구분 강화
- 핵심 포인트(TLSP/BSP/Mv/Mv_shifted)에 도형 + 태그를 명확히 표시
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
    """거북이 선과 bending 포인트를 시각화."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # 원본 edge 점: 얇고 투명하게 표시하여 배경 윤곽을 보존
    if original_points:
        ox = [p[0] for p in original_points]
        oy = [p[1] for p in original_points]
        ax.scatter(
            ox,
            oy,
            color="#BDBDBD",
            s=0.8,
            alpha=0.22,
            linewidths=0.0,
            label="Original edge points",
            zorder=1,
        )

    # 거북이 선: 원본 대비 살짝 강조하되 과도하게 두껍지 않게 표시
    tl = result.turtle_line_path
    if len(tl) >= 2:
        xs = [p[0] for p in tl]
        ys = [p[1] for p in tl]
        ax.plot(
            xs,
            ys,
            color="#1F77B4",
            linewidth=0.85,
            label="Turtle line",
            alpha=0.95,
            zorder=3,
        )

    # Bending points: turbo 스펙트럼으로 연속 순번의 변화량 가시화
    if result.bending_points:
        bx = [p[0] for p in result.bending_points]
        by = [p[1] for p in result.bending_points]
        indices = list(range(1, len(result.bending_points) + 1))
        sc = ax.scatter(
            bx,
            by,
            c=indices,
            cmap="turbo",
            s=8,
            label="Bending points (1..PBL)",
            linewidths=0.0,
            zorder=5,
        )
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("Bending index")

    # 핵심 포인트: 기호 + 태그를 같이 그려 사용자 해석성을 높인다.
    ax.scatter(
        [result.tlsp[0]],
        [result.tlsp[1]],
        color="#D62728",
        s=62,
        marker="v",
        label="TLSP",
        zorder=10,
        edgecolors="white",
        linewidths=0.8,
    )
    ax.annotate(
        "TLSP",
        xy=result.tlsp,
        xytext=(6, -10),
        textcoords="offset points",
        fontsize=8,
        color="#D62728",
        weight="bold",
    )

    ax.scatter(
        [result.bsp[0]],
        [result.bsp[1]],
        color="#FF7F0E",
        s=62,
        marker="D",
        label="BSP",
        zorder=10,
        edgecolors="white",
        linewidths=0.8,
    )
    ax.annotate(
        "BSP",
        xy=result.bsp,
        xytext=(6, 8),
        textcoords="offset points",
        fontsize=8,
        color="#FF7F0E",
        weight="bold",
    )

    ax.scatter(
        [result.mv[0]],
        [result.mv[1]],
        color="#9467BD",
        s=58,
        marker="P",
        label="Mv",
        zorder=10,
        edgecolors="white",
        linewidths=0.8,
    )
    ax.annotate(
        "Mv",
        xy=result.mv,
        xytext=(7, -8),
        textcoords="offset points",
        fontsize=8,
        color="#9467BD",
        weight="bold",
    )

    ax.scatter(
        [result.mv_shifted[0]],
        [result.mv_shifted[1]],
        color="#E377C2",
        s=54,
        marker="x",
        label="Mv shifted",
        zorder=10,
        linewidths=1.2,
    )
    ax.annotate(
        "Mv'",
        xy=result.mv_shifted,
        xytext=(7, 8),
        textcoords="offset points",
        fontsize=8,
        color="#E377C2",
        weight="bold",
    )

    # 앞머리/윗머리 검출 구간
    if result.front_head_run:
        fh_x = [p[0] for p in result.front_head_run]
        fh_y = [p[1] for p in result.front_head_run]
        ax.plot(
            fh_x,
            fh_y,
            color="#2CA02C",
            linewidth=1.25,
            label="Front head (FH)",
            zorder=8,
        )

    if result.upper_head_run:
        uh_x = [p[0] for p in result.upper_head_run]
        uh_y = [p[1] for p in result.upper_head_run]
        ax.plot(
            uh_x,
            uh_y,
            color="#8C564B",
            linewidth=1.25,
            label="Upper head (UH)",
            zorder=8,
        )

    ax.set_aspect("equal", adjustable="datalim")
    ax.invert_yaxis()  # 이미지 좌표계 (y가 아래로 증가)
    ax.set_xlabel("X (pixel)")
    ax.set_ylabel("Y (pixel)")
    ax.set_title("Strategy 2 - Turtle Line and Bending Trajectory")
    ax.legend(loc="best", fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.15, linewidth=0.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
