"""
전략 2: 거북이 머리 더듬기 알고리즘.

점 기반 입력에서 connected component를 복원하고,
거북이 선을 찾아 가상 마커와 bending 포인트를 계산.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from .geometry import (
    Point,
    compute_line_intersection,
    distance,
    find_connected_components,
    find_endpoints,
    find_first_horizontal_run,
    find_first_vertical_run,
    find_longest_path_with_branching,
    sample_path_at_intervals,
)


@dataclass
class Strategy2Config:
    FH: float  # Forehead vertical length threshold (점 개수)
    UH: float  # Upper head horizontal length threshold (점 개수)
    SX: float  # Shift in x for virtual marker
    SY: float  # Shift in y for virtual marker
    PBL: int  # Panel bending length (출력 포인트 개수)
    sample_step: float = 1.0  # 샘플링 간격 (픽셀)


@dataclass
class Strategy2Result:
    tlsp: Point
    turtle_line_path: List[Point]  # 거북이 선의 경로 (순서대로)
    front_head_run: List[Point]  # 거북이 앞머리끝 직선 구간
    upper_head_run: List[Point]  # 거북이 윗머리끝 직선 구간
    mv: Point  # 가상 마커
    mv_shifted: Point
    bsp: Point
    bending_points: List[Point]  # 순서번호 1..PBL에 해당하는 점들


class Strategy2Error(Exception):
    """Domain-specific error for Strategy 2 processing."""


def parse_txt_points(path: str) -> List[Point]:
    """TXT 파일에서 점 좌표들을 읽어온다.

    - 한 줄에 한 점: x,y 또는 x y
    - # 로 시작하는 줄은 주석으로 무시
    - 빈 줄은 무시
    """
    points: List[Point] = []

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            # 주석 줄 무시
            if line.startswith("#"):
                continue

            # x,y 또는 x y 형식 파싱
            if "," in line:
                parts = line.split(",")
            else:
                parts = line.split()

            if len(parts) < 2:
                continue

            try:
                x = int(round(float(parts[0])))
                y = int(round(float(parts[1])))
                points.append((x, y))
            except ValueError:
                continue

    if not points:
        raise Strategy2Error("입력 txt 파일에서 좌표를 찾을 수 없습니다.")

    return points


def pick_two_longest_lines(components: List[List[Point]]) -> Tuple[List[Point], List[Point]]:
    """Connected component들 중 점 개수가 가장 많은 상위 2개를 선택."""
    if len(components) < 2:
        raise Strategy2Error("최소 두 개의 연결된 선이 필요합니다.")

    # 점 개수로 정렬
    lengths = [(len(comp), i) for i, comp in enumerate(components)]
    lengths.sort(reverse=True)

    idx1 = lengths[0][1]
    idx2 = lengths[1][1]
    return components[idx1], components[idx2]


def find_turtle_line(comp1: List[Point], comp2: List[Point]) -> List[Point]:
    """두 component 중 거북이 선을 찾는다.

    거북이 선: 가장 아래(y 최대)인 점을 포함하는 component.
    """
    all_points = [(p, 1) for p in comp1] + [(p, 2) for p in comp2]
    lowest_point, which = max(all_points, key=lambda t: t[0][1])
    return comp1 if which == 1 else comp2


def find_tlsp(turtle_component: List[Point]) -> Tuple[Point, List[Point]]:
    """거북이 선의 TLSP를 찾고 경로를 정렬한다.

    TLSP: 끝점 중 더 아래(y 최대)인 점.
    순환 구조나 분기점이 있으면 예외처리.
    """
    endpoints = find_endpoints(turtle_component)

    if not endpoints:
        raise Strategy2Error("거북이 선에서 끝점을 찾을 수 없습니다.")

    # 더 아래에 있는 끝점 선택
    tlsp = max(endpoints, key=lambda p: p[1])

    # TLSP에서 가장 긴 경로 찾기
    path = find_longest_path_with_branching(turtle_component, tlsp)

    return tlsp, path


def find_front_head_and_upper_head(
    path: List[Point], fh: int, uh: int
) -> Tuple[List[Point], List[Point]]:
    """거북이 선 경로에서 앞머리끝과 윗머리끝 직선 구간을 찾는다."""
    # 앞머리끝: 세로 직선 구간 (점 개수 >= FH)
    front_head = find_first_vertical_run(path, int(fh))
    if not front_head:
        raise Strategy2Error(f"조건을 만족하는 세로(FH={fh}) 구간을 찾지 못했습니다.")

    # 앞머리끝 이후 경로에서 윗머리끝 찾기
    front_head_end_idx = path.index(front_head[-1])
    remaining_path = path[front_head_end_idx + 1 :]

    upper_head = find_first_horizontal_run(remaining_path, int(uh))
    if not upper_head:
        raise Strategy2Error(f"조건을 만족하는 가로(UH={uh}) 구간을 찾지 못했습니다.")

    return front_head, upper_head


def compute_mv(front_head: List[Point], upper_head: List[Point]) -> Point:
    """가상 마커 Mv 계산.

    FH 직선(x=상수)과 UH 직선(y=상수)의 교차점.
    """
    return compute_line_intersection(front_head, upper_head)


def find_bsp(turtle_path: List[Point], mv_shifted: Point) -> Point:
    """BSP 찾기: 거북이 선의 점 중 mv_shifted에 가장 가까운 점."""
    if not turtle_path:
        raise Strategy2Error("거북이 선 경로가 비어있습니다.")

    return min(turtle_path, key=lambda p: distance(p, mv_shifted))


def run_strategy2_on_points(
    points: List[Point], config: Strategy2Config
) -> Strategy2Result:
    """점 집합에 대해 전략 2를 실행."""
    # 1. Connected component 찾기
    components = find_connected_components(points)

    # 2. 가장 긴 두 개의 선 선택
    comp1, comp2 = pick_two_longest_lines(components)

    # 3. 거북이 선 찾기
    turtle_component = find_turtle_line(comp1, comp2)

    # 4. TLSP 찾기 및 경로 정렬
    tlsp, turtle_path = find_tlsp(turtle_component)

    # 5 & 6. 앞머리끝과 윗머리끝 찾기
    front_head, upper_head = find_front_head_and_upper_head(
        turtle_path, config.FH, config.UH
    )

    # 7. 가상 마커 Mv 계산
    mv = compute_mv(front_head, upper_head)

    # 8. Mv 평행이동 및 BSP 찾기
    mv_shifted = (int(round(mv[0] + config.SX)), int(round(mv[1] + config.SY)))
    bsp = find_bsp(turtle_path, mv_shifted)

    # 9. BSP에서 TLSP 방향의 반대 방향으로 경로 탐색
    bsp_idx = turtle_path.index(bsp)
    tlsp_idx = turtle_path.index(tlsp)

    # TLSP 방향의 반대 = BSP에서 멀어지는 방향
    # BSP가 TLSP보다 앞에 있으면 뒤로, 뒤에 있으면 앞으로
    if bsp_idx < tlsp_idx:
        # BSP -> ... -> TLSP 방향이므로, 반대는 BSP에서 앞으로
        sampling_path = turtle_path[bsp_idx:]
    else:
        # TLSP -> ... -> BSP 방향이므로, 반대는 BSP에서 뒤로
        sampling_path = turtle_path[bsp_idx:]

    # PBL개 점 샘플링 (경로의 모든 점 순회 후 1픽셀 간격)
    bending_points = sample_path_at_intervals(
        sampling_path, 0, config.PBL, config.sample_step
    )

    return Strategy2Result(
        tlsp=tlsp,
        turtle_line_path=turtle_path,
        front_head_run=front_head,
        upper_head_run=upper_head,
        mv=mv,
        mv_shifted=mv_shifted,
        bsp=bsp,
        bending_points=bending_points,
    )


def run_strategy2_on_file(path: str, config: Strategy2Config) -> Strategy2Result:
    """TXT 파일에 대해 전략 2를 실행."""
    points = parse_txt_points(path)
    return run_strategy2_on_points(points, config)


def save_result_points_txt(path: str, result: Strategy2Result) -> None:
    """결과를 TXT 파일로 저장.

    포맷: x,y (줄 순서가 곧 순서번호 1..PBL)
    """
    with open(path, "w", encoding="utf-8") as f:
        f.write("# x,y\n")
        for p in result.bending_points:
            f.write(f"{p[0]},{p[1]}\n")
