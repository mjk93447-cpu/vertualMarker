"""
전략 2: 거북이 머리 더듬기 알고리즘.

점 기반 입력에서 connected component를 복원하고,
거북이 선을 찾아 가상 마커와 bending 포인트를 계산.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .geometry import (
    Point,
    compute_line_intersection,
    distance,
    find_connected_components,
    find_endpoints,
    find_first_horizontal_run,
    find_first_vertical_run,
    find_path_bfs,
    get_neighbors,
    sample_path_at_intervals,
)

logger = logging.getLogger(__name__)


@dataclass
class Strategy2Config:
    FH: float  # Forehead vertical length threshold (점 개수)
    UH: float  # Upper head horizontal length threshold (점 개수)
    SX: float  # Shift in x for virtual marker
    SY: float  # Shift in y for virtual marker
    PBL: int  # Panel bending length (출력 포인트 개수)
    sample_step: float = 1.0  # 샘플링 간격 (픽셀)
    vertical_angle_tolerance: float = 5.0  # 세로 각도 허용 (도)
    horizontal_angle_tolerance: float = 5.0  # 가로 각도 허용 (도)


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
    diagnostics: Dict[str, Any] = field(default_factory=dict)


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


def pick_two_longest_lines(
    components: List[List[Point]],
) -> Tuple[List[Point], List[Point], Dict[str, Any]]:
    """Connected component들 중 점 개수가 가장 많은 상위 2개를 선택.

    반환: (comp1, comp2, diagnostics)
    diagnostics에 상위 component 정보 포함.
    """
    if len(components) < 2:
        raise Strategy2Error(
            f"최소 두 개의 연결된 선이 필요합니다. (발견: {len(components)}개)"
        )

    # 점 개수로 정렬 (내림차순)
    sorted_comps = sorted(
        enumerate(components), key=lambda t: len(t[1]), reverse=True
    )

    idx1, comp1 = sorted_comps[0]
    idx2, comp2 = sorted_comps[1]

    # 진단 정보: 상위 component들의 정보
    diag: Dict[str, Any] = {
        "num_components": len(components),
        "top_components": [],
    }
    for rank, (orig_idx, comp) in enumerate(sorted_comps[:5]):
        bottom_point = max(comp, key=lambda p: p[1])
        diag["top_components"].append(
            {
                "rank": rank + 1,
                "original_index": orig_idx,
                "length": len(comp),
                "bottom_point": bottom_point,
            }
        )

    return comp1, comp2, diag


def find_turtle_line(
    comp1: List[Point], comp2: List[Point]
) -> Tuple[List[Point], List[Point]]:
    """두 component 중 거북이 선을 찾는다.

    거북이 선: y축 방향에서 가장 아래(y 최대)인 점 두개 중 더 아래인
    점을 포함하고 있는 선.

    반환: (turtle_component, other_component)
    """
    bottom1 = max(comp1, key=lambda p: p[1])
    bottom2 = max(comp2, key=lambda p: p[1])

    if bottom1[1] >= bottom2[1]:
        return comp1, comp2
    else:
        return comp2, comp1


def find_tlsp(
    turtle_component: List[Point],
) -> Tuple[Point, List[Point], Dict[str, Any]]:
    """거북이 선의 TLSP를 찾고 경로를 정렬한다.

    TLSP: 끝점 중 더 아래(y 최대)인 점.

    BFS를 사용해 TLSP에서 가장 먼 점까지의 경로를 찾는다.
    이 방식은 분기점이 있어도 올바르게 동작한다.

    반환: (tlsp, path, diagnostics)
    """
    diag: Dict[str, Any] = {}

    # 끝점 찾기
    endpoints = find_endpoints(turtle_component)
    diag["num_endpoints"] = len(endpoints)

    if endpoints:
        # 끝점 중 가장 아래(y 최대)인 것을 TLSP로 선택
        tlsp = max(endpoints, key=lambda p: p[1])
    else:
        # 순환 구조: 가장 아래 점 선택
        tlsp = max(turtle_component, key=lambda p: p[1])

    diag["tlsp"] = tlsp

    # BFS로 TLSP에서 가장 먼 점까지의 경로를 찾는다.
    # 이것이 거북이 선의 "주 경로(main chain)"이 된다.
    path = find_path_bfs(turtle_component, tlsp, end=None)
    diag["path_length"] = len(path)
    diag["path_end_point"] = path[-1] if path else None

    if len(path) < 2:
        # 혹시 BFS가 실패하면 (거의 불가능하지만), 그리디 방식 시도
        path = _build_ordered_path(turtle_component, tlsp)
        diag["path_method"] = "greedy_fallback"
        diag["path_length"] = len(path)
    else:
        diag["path_method"] = "bfs"

    if len(path) < 2:
        raise Strategy2Error(
            f"거북이 선 경로가 너무 짧습니다. "
            f"(component 점 수: {len(turtle_component)}, "
            f"경로 길이: {len(path)}, TLSP: {tlsp})"
        )

    return tlsp, path, diag


def _build_ordered_path(
    component: List[Point], start: Point
) -> List[Point]:
    """순서대로 경로 생성: 시작점에서 연결된 점들을 순차적으로 따라가며 경로 생성.

    분기점에서는 방향 연속성을 우선하고, 그래도 안 되면
    미방문 이웃 중 가장 아래(y 최대)쪽으로 이동.
    """
    from typing import Set

    point_set = set(component)
    path: List[Point] = [start]
    visited: Set[Point] = {start}
    current = start

    while True:
        neighbors = get_neighbors(current, point_set)
        unvisited_neighbors = [n for n in neighbors if n not in visited]

        if not unvisited_neighbors:
            break

        if len(unvisited_neighbors) == 1:
            next_point = unvisited_neighbors[0]
        else:
            # 현재 경로의 방향 추정
            if len(path) >= 2:
                prev_dir = (
                    path[-1][0] - path[-2][0],
                    path[-1][1] - path[-2][1],
                )
                # 같은 방향의 이웃 찾기
                same_dir_neighbors = [
                    n
                    for n in unvisited_neighbors
                    if (
                        n[0] - current[0],
                        n[1] - current[1],
                    )
                    == prev_dir
                ]
                if same_dir_neighbors:
                    next_point = same_dir_neighbors[0]
                else:
                    # 비슷한 방향 (같은 부호)의 이웃 찾기
                    similar_dir_neighbors = []
                    for n in unvisited_neighbors:
                        nd = (n[0] - current[0], n[1] - current[1])
                        # 방향 유사도 점수
                        score = (
                            (1 if (nd[0] > 0) == (prev_dir[0] > 0) and prev_dir[0] != 0 else 0)
                            + (1 if (nd[1] > 0) == (prev_dir[1] > 0) and prev_dir[1] != 0 else 0)
                        )
                        similar_dir_neighbors.append((score, n))
                    similar_dir_neighbors.sort(key=lambda t: t[0], reverse=True)
                    next_point = similar_dir_neighbors[0][1]
            else:
                # 경로가 짧으면 y가 작아지는 방향 (위로 올라가는 방향) 우선
                next_point = min(unvisited_neighbors, key=lambda p: p[1])

        visited.add(next_point)
        path.append(next_point)
        current = next_point

    return path


def find_front_head_and_upper_head(
    path: List[Point],
    fh: int,
    uh: int,
    vertical_angle_tolerance: float = 5.0,
    horizontal_angle_tolerance: float = 5.0,
) -> Tuple[List[Point], List[Point]]:
    """거북이 선 경로에서 앞머리끝과 윗머리끝 직선 구간을 찾는다.

    각도 허용 범위를 적용하여 near-vertical/near-horizontal 구간을 인식.
    """
    # 앞머리끝: 세로 직선 구간 (점 개수 >= FH)
    front_head = find_first_vertical_run(
        path, int(fh), angle_tolerance_deg=vertical_angle_tolerance
    )
    if not front_head:
        # 허용 각도를 넓혀서 재시도
        front_head = find_first_vertical_run(
            path, int(fh), angle_tolerance_deg=max(vertical_angle_tolerance * 2, 15.0)
        )
    if not front_head:
        raise Strategy2Error(
            f"조건을 만족하는 세로(FH={fh}) 구간을 찾지 못했습니다. "
            f"(경로 길이: {len(path)}, 각도 허용: {vertical_angle_tolerance}도)"
        )

    # 앞머리끝 이후 경로에서 윗머리끝 찾기
    # front_head의 마지막 점이 경로에서 어디에 있는지 찾기
    front_head_end_idx = -1
    front_head_last = front_head[-1]
    for idx, p in enumerate(path):
        if p == front_head_last:
            front_head_end_idx = idx
            break

    if front_head_end_idx < 0:
        # 정확히 일치하는 점이 없으면, 가장 가까운 점 사용
        min_dist = float("inf")
        for idx, p in enumerate(path):
            d = distance(p, front_head_last)
            if d < min_dist:
                min_dist = d
                front_head_end_idx = idx

    remaining_path = path[front_head_end_idx + 1:]

    upper_head = find_first_horizontal_run(
        remaining_path, int(uh), angle_tolerance_deg=horizontal_angle_tolerance
    )
    if not upper_head:
        # 허용 각도를 넓혀서 재시도
        upper_head = find_first_horizontal_run(
            remaining_path, int(uh), angle_tolerance_deg=max(horizontal_angle_tolerance * 2, 15.0)
        )
    if not upper_head:
        raise Strategy2Error(
            f"조건을 만족하는 가로(UH={uh}) 구간을 찾지 못했습니다. "
            f"(앞머리 이후 경로 길이: {len(remaining_path)}, "
            f"각도 허용: {horizontal_angle_tolerance}도)"
        )

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
    all_diagnostics: Dict[str, Any] = {}

    # 1. Connected component 찾기
    components = find_connected_components(points)
    all_diagnostics["num_components"] = len(components)

    # 2. 가장 긴 두 개의 선 선택
    comp1, comp2, pick_diag = pick_two_longest_lines(components)
    all_diagnostics["pick_lines"] = pick_diag

    # 가장 긴 두 선 정보 로그
    comp1_bottom = max(comp1, key=lambda p: p[1])
    comp2_bottom = max(comp2, key=lambda p: p[1])
    all_diagnostics["line1_length"] = len(comp1)
    all_diagnostics["line1_bottom"] = comp1_bottom
    all_diagnostics["line2_length"] = len(comp2)
    all_diagnostics["line2_bottom"] = comp2_bottom

    logger.info(
        "가장 긴 선 #1: 길이=%d, 최하단=(%d,%d)",
        len(comp1), comp1_bottom[0], comp1_bottom[1],
    )
    logger.info(
        "가장 긴 선 #2: 길이=%d, 최하단=(%d,%d)",
        len(comp2), comp2_bottom[0], comp2_bottom[1],
    )

    # 3. 거북이 선 찾기
    turtle_component, other_component = find_turtle_line(comp1, comp2)
    turtle_bottom = max(turtle_component, key=lambda p: p[1])
    all_diagnostics["turtle_component_length"] = len(turtle_component)
    all_diagnostics["turtle_bottom"] = turtle_bottom
    logger.info(
        "거북이 선 선택: 길이=%d, 최하단=(%d,%d)",
        len(turtle_component), turtle_bottom[0], turtle_bottom[1],
    )

    # 4. TLSP 찾기 및 경로 정렬
    tlsp, turtle_path, tlsp_diag = find_tlsp(turtle_component)
    all_diagnostics["tlsp"] = tlsp_diag

    logger.info(
        "거북이선 특정완료: TLSP=(%d,%d), 전체 경로 길이=%d픽셀, "
        "경로 끝점=(%d,%d), 탐색방법=%s",
        tlsp[0], tlsp[1],
        len(turtle_path),
        turtle_path[-1][0], turtle_path[-1][1],
        tlsp_diag.get("path_method", "unknown"),
    )

    # 5 & 6. 앞머리끝과 윗머리끝 찾기
    front_head, upper_head = find_front_head_and_upper_head(
        turtle_path,
        config.FH,
        config.UH,
        vertical_angle_tolerance=config.vertical_angle_tolerance,
        horizontal_angle_tolerance=config.horizontal_angle_tolerance,
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
    if bsp_idx < tlsp_idx:
        # BSP -> ... -> TLSP 방향이므로, 반대는 BSP에서 앞으로 (인덱스 감소 방향)
        # 하지만 경로는 TLSP에서 시작하므로 bsp_idx가 TLSP보다 뒤에 있음
        # 반대 방향 = bsp_idx에서 경로 끝 방향
        sampling_path = turtle_path[bsp_idx:]
    else:
        # TLSP -> ... -> BSP 방향이므로, 반대는 BSP에서 경로 끝 방향
        sampling_path = turtle_path[bsp_idx:]

    # PBL개 점 샘플링
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
        diagnostics=all_diagnostics,
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
