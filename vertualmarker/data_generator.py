"""
합성 데이터 생성기 - 점 기반.

실제 OLED 패널 PCBL 측면 윤곽과 유사한 점들의 집합을 생성.
8-이웃 연결성을 유지하면서 두 개의 긴 선을 만든다.
"""
from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from typing import List, Tuple

Point = Tuple[int, int]  # 픽셀 좌표는 정수


@dataclass
class SyntheticParams:
    length: int = 2000  # 전체 x 범위 (픽셀)
    base_y: int = 1000  # 거북이 선 기본 y (아래쪽)
    offset_y: int = -150  # 위쪽 선과의 y 오프셋
    head_height: int = 200  # 앞머리(세로) 구간 높이 (점 개수)
    head_width: int = 300  # 윗머리(가로) 구간 길이 (점 개수)
    bend_center_x: int = 1000  # 굽힘 중심 x
    bend_amplitude: int = 50  # 굽힘 진폭 (픽셀)
    bend_width: int = 800  # 굽힘이 일어나는 x 구간 폭
    noise: float = 1.0  # 무작위 노이즈 (픽셀 단위, 작은 값)
    num_points: int = 2000  # 본체 구간 점 개수


def generate_turtle_and_partner(params: SyntheticParams) -> List[Point]:
    """점 기반으로 거북이 선과 파트너 선을 생성.

    반환: 모든 점들의 리스트 (두 선이 연결되어 있지 않음)
    """
    points: List[Point] = []

    # 거북이 선 생성 (연결된 하나의 component로)
    turtle_points: List[Point] = []

    # 1) 앞머리 세로 구간 (x=0에서 head_height 만큼 세로)
    x0 = 0
    y_bottom = params.base_y + params.head_height // 2
    y_top = params.base_y - params.head_height // 2

    for i in range(params.head_height + 1):
        y = y_bottom - i
        turtle_points.append((x0, y))

    # 2) 윗머리 가로 구간 (y_top 위치에서 head_width 만큼 가로)
    # 마지막 세로 점(y_top)에서 시작해서 가로로 연결
    for i in range(1, params.head_width + 1):
        x = x0 + i
        turtle_points.append((x, y_top))

    # 3) 본체 bending 구간
    # 마지막 가로 점에서 y 방향으로 연결된 다음 가로로 진행
    x_start = x0 + params.head_width
    y_start = y_top
    
    # y_top에서 base_y로 연결 (대각선 또는 세로)
    if y_start != params.base_y:
        # 대각선으로 연결
        dy = params.base_y - y_start
        steps = abs(dy)
        for i in range(1, steps + 1):
            y = y_start + (dy // steps) * i if steps > 0 else y_start
            turtle_points.append((x_start, y))
    
    # 본체 가로 구간
    for i in range(1, params.num_points + 1):
        x = x_start + i

        # 굽힘 프로파일
        dx = x - params.bend_center_x
        if abs(dx) <= params.bend_width // 2:
            t = dx / (params.bend_width / 2)
            bend = int(round(-params.bend_amplitude * (1 - t * t)))
        else:
            bend = 0

        y = params.base_y + bend
        # 작은 노이즈 추가 (연결성 유지)
        noise_x = int(round((random.random() - 0.5) * params.noise))
        noise_y = int(round((random.random() - 0.5) * params.noise))
        # 노이즈로 인해 연결이 끊어지지 않도록 제한
        noise_x = max(-1, min(1, noise_x))
        noise_y = max(-1, min(1, noise_y))
        turtle_points.append((x + noise_x, y + noise_y))

    # 파트너 선 생성 (위쪽에 거의 평행)
    partner_points: List[Point] = []
    for x, y in turtle_points:
        # 거의 같은 곡률을 유지하면서 위로 이동
        partner_y = y + params.offset_y
        # 작은 노이즈
        noise_x = int(round((random.random() - 0.5) * params.noise))
        noise_y = int(round((random.random() - 0.5) * params.noise))
        partner_points.append((x + noise_x, partner_y + noise_y))

    # 두 선을 분리하기 위해 빈 줄 추가하지 않고 그냥 합침
    # (connected component가 자동으로 분리됨)
    points.extend(turtle_points)
    points.extend(partner_points)

    return points


def save_points_txt(path: str, points: List[Point]) -> None:
    """점들을 TXT 파일로 저장."""
    with open(path, "w", encoding="utf-8") as f:
        f.write("# x,y\n")
        for x, y in points:
            f.write(f"{x},{y}\n")


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic TXT example for vertualMarker Strategy 2."
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="example_turtle.txt",
        help="Output TXT file path (default: example_turtle.txt)",
    )
    parser.add_argument(
        "--length", type=int, default=2000, help="Total x-length of curves (pixels)",
    )
    parser.add_argument(
        "--base-y", type=int, default=1000, help="Base y of turtle line (pixels)",
    )
    parser.add_argument(
        "--offset-y",
        type=int,
        default=-150,
        help="Vertical offset between turtle and partner lines (pixels)",
    )
    parser.add_argument(
        "--head-height",
        type=int,
        default=200,
        help="Front head vertical segment height (number of points)",
    )
    parser.add_argument(
        "--head-width",
        type=int,
        default=300,
        help="Upper head horizontal segment width (number of points)",
    )
    parser.add_argument(
        "--bend-center-x",
        type=int,
        default=1000,
        help="X position of bending center (pixels)",
    )
    parser.add_argument(
        "--bend-amplitude",
        type=int,
        default=50,
        help="Bending amplitude (pixels)",
    )
    parser.add_argument(
        "--bend-width",
        type=int,
        default=800,
        help="Width of x-region where bending occurs (pixels)",
    )
    parser.add_argument(
        "--noise",
        type=float,
        default=1.0,
        help="Random noise amplitude (pixels)",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=2000,
        help="Number of points in main body segment",
    )

    args = parser.parse_args(argv)

    params = SyntheticParams(
        length=args.length,
        base_y=args.base_y,
        offset_y=args.offset_y,
        head_height=args.head_height,
        head_width=args.head_width,
        bend_center_x=args.bend_center_x,
        bend_amplitude=args.bend_amplitude,
        bend_width=args.bend_width,
        noise=args.noise,
        num_points=args.num_points,
    )
    points = generate_turtle_and_partner(params)
    save_points_txt(args.output, points)
    print(f"Synthetic example saved to {args.output} ({len(points)} points)")


if __name__ == "__main__":
    main()
