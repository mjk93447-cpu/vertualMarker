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

    # 거북이선 외 긴 선들을 희미한 회색으로 표시
    if result.support_line_paths:
        for i, aux in enumerate(result.support_line_paths, start=1):
            if len(aux) < 2:
                continue
            ax.plot(
                [p[0] for p in aux],
                [p[1] for p in aux],
                color="#A9B0BE",
                linewidth=0.7,
                alpha=0.45,
                zorder=2,
                label="Other long lines" if i == 1 else None,
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
        xytext=(6, 10),
        textcoords="offset points",
        fontsize=8,
        color="#FF7F0E",
        weight="bold",
    )
    ax.annotate(
        f"({result.bsp[0]}, {result.bsp[1]})",
        xy=result.bsp,
        xytext=(6, 21),
        textcoords="offset points",
        fontsize=7,
        color="#FFB866",
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
    ax.annotate(
        f"({result.mv[0]}, {result.mv[1]})",
        xy=result.mv,
        xytext=(7, -20),
        textcoords="offset points",
        fontsize=7,
        color="#B9A4D8",
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
        xytext=(7, 10),
        textcoords="offset points",
        fontsize=8,
        color="#E377C2",
        weight="bold",
    )
    ax.annotate(
        f"({result.mv_shifted[0]}, {result.mv_shifted[1]})",
        xy=result.mv_shifted,
        xytext=(7, 21),
        textcoords="offset points",
        fontsize=7,
        color="#F0A7DA",
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

    # Mv' -> BSP 오차 벡터(초기 오차) 표시
    ax.annotate(
        "",
        xy=result.bsp,
        xytext=result.mv_shifted,
        arrowprops=dict(
            arrowstyle="->",
            color="#FFD166",
            lw=1.3,
            alpha=0.95,
            shrinkA=2,
            shrinkB=2,
        ),
        zorder=11,
    )
    mid_x = (result.mv_shifted[0] + result.bsp[0]) / 2
    mid_y = (result.mv_shifted[1] + result.bsp[1]) / 2
    ax.text(
        mid_x,
        mid_y,
        f"dX={result.mv_bsp_dx}, dY={result.mv_bsp_dy}, dist={result.mv_bsp_distance:.2f}",
        fontsize=8,
        color="#FFD166",
        bbox=dict(boxstyle="round,pad=0.25", fc="#2A2A2A", ec="#FFD166", alpha=0.82),
        zorder=12,
    )

    # 핵심 좌표/오차 정보를 우측 상단 패널로 제공
    info_text = (
        f"TLSP: ({result.tlsp[0]}, {result.tlsp[1]})\n"
        f"Mv: ({result.mv[0]}, {result.mv[1]})\n"
        f"Mv': ({result.mv_shifted[0]}, {result.mv_shifted[1]})\n"
        f"BSP: ({result.bsp[0]}, {result.bsp[1]})\n"
        f"Offset (BSP-Mv'): ({result.mv_bsp_dx}, {result.mv_bsp_dy})\n"
        f"Offset distance: {result.mv_bsp_distance:.3f}"
    )
    ax.text(
        0.99,
        0.99,
        info_text,
        transform=ax.transAxes,
        va="top",
        ha="right",
        fontsize=8,
        color="#EDEFF5",
        bbox=dict(boxstyle="round,pad=0.35", fc="#1F2430", ec="#586174", alpha=0.9),
        zorder=20,
    )

    if result.warnings:
        visible = result.warnings[:5]
        warn_lines = [f"- {w}" for w in visible]
        if len(result.warnings) > len(visible):
            warn_lines.append(f"- ... and {len(result.warnings) - len(visible)} more")
        warn_text = "Auto-correction / diagnostics:\n" + "\n".join(warn_lines)
        ax.text(
            0.01,
            0.99,
            warn_text,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=7.5,
            color="#FFD6A5",
            bbox=dict(boxstyle="round,pad=0.35", fc="#3B2B20", ec="#C89E63", alpha=0.85),
            zorder=20,
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
