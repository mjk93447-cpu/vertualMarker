"""
전략 2: 거북이 머리 더듬기 알고리즘.

점 기반 입력에서 connected component를 복원하고,
거북이 선을 찾아 가상 마커와 bending 포인트를 계산.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import List, Tuple

from .geometry import (
    Point,
    compute_line_intersection,
    detect_straight_runs,
    distance,
    find_connected_components,
    find_endpoints,
    find_first_horizontal_run,
    find_first_vertical_run,
    get_neighbors,
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
    heal_gap: int = 2  # line-break auto-heal max gap (pixels)


@dataclass
class Strategy2Result:
    tlsp: Point
    turtle_line_path: List[Point]  # 거북이 선의 경로 (TLSP 기준 순서)
    front_head_run: List[Point]  # 거북이 앞머리끝 직선 구간
    upper_head_run: List[Point]  # 거북이 윗머리끝 직선 구간
    mv: Point  # 가상 마커
    mv_shifted: Point
    bsp: Point
    bending_points: List[Point]  # 순서번호 1..PBL에 해당하는 점들
    turtle_lowest_point: Point  # 거북이 선에서 y가 가장 큰 점
    turtle_line_length: int  # 거북이 선 경로 길이(점 개수 기준)
    longest_two_lines_info: List[Tuple[Point, int]]  # [(최하단점, 길이), ...]
    support_line_paths: List[List[Point]]  # 거북이선 외 긴 선(시각화용)
    warnings: List[str]  # 자동보정/예외 대응 피드백
    mv_bsp_dx: int  # BSP - Mv' 의 x 오차
    mv_bsp_dy: int  # BSP - Mv' 의 y 오차
    mv_bsp_distance: float  # BSP와 Mv' 사이 거리


class Strategy2Error(Exception):
    """Domain-specific error for Strategy 2 processing."""


def parse_txt_points_with_report(path: str) -> Tuple[List[Point], int]:
    """TXT 파일에서 점 좌표를 읽고 skip 라인 수를 함께 반환한다.

    - 한 줄에 한 점: x,y 또는 x y
    - # 로 시작하는 줄은 주석으로 무시
    - 빈 줄은 무시
    """
    points: List[Point] = []
    skipped = 0

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
                skipped += 1
                continue

            try:
                x = int(round(float(parts[0])))
                y = int(round(float(parts[1])))
                points.append((x, y))
            except ValueError:
                skipped += 1
                continue

    if not points:
        raise Strategy2Error(
            f"No valid coordinates were found in the input TXT file. "
            f"Skipped malformed lines: {skipped}."
        )

    return points, skipped


def parse_txt_points(path: str) -> List[Point]:
    """Backward-compatible parser that returns points only."""
    points, _skipped = parse_txt_points_with_report(path)
    return points


def pick_two_longest_lines(components: List[List[Point]]) -> Tuple[List[Point], List[Point]]:
    """Connected component들 중 점 개수가 가장 많은 상위 2개를 선택.
    
    만약 거북이 선이 여러 component로 분리되어 있으면 (예: 앞머리+본체),
    가장 긴 component와 그와 가까운 component를 합쳐서 하나의 선으로 처리.
    """
    if len(components) < 2:
        raise Strategy2Error("At least two connected line components are required.")

    # 점 개수로 정렬
    lengths = [(len(comp), i) for i, comp in enumerate(components)]
    lengths.sort(reverse=True)

    idx1 = lengths[0][1]
    idx2 = lengths[1][1]
    
    comp1 = components[idx1]
    comp2 = components[idx2]
    
    # 두 component가 가까이 있으면 합치기 (거북이 선이 분리된 경우 처리)
    # 간단한 휴리스틱: 두 component의 최소 거리가 작으면 합침
    min_dist = min(
        distance(p1, p2) for p1 in comp1 for p2 in comp2
    )
    if min_dist < 10:  # 10픽셀 이내면 합침
        merged = comp1 + comp2
        # 나머지 component 중 가장 긴 것 찾기
        remaining = [components[idx] for _, idx in lengths if idx not in [idx1, idx2]]
        if remaining:
            comp2 = max(remaining, key=len)
        else:
            # 두 번째 component가 없으면 첫 번째만 반환 (하지만 함수 시그니처상 두 개 필요)
            comp2 = comp1[:]
        return merged, comp2
    
    return comp1, comp2


def summarize_longest_two_lines(components: List[List[Point]]) -> List[Tuple[Point, int]]:
    """가장 긴 2개 선의 (최하단 점, 길이) 정보를 반환한다."""
    if not components:
        return []
    ranked = sorted(components, key=len, reverse=True)[:2]
    return [(max(comp, key=lambda p: p[1]), len(comp)) for comp in ranked]


def _interpolate_points(p1: Point, p2: Point) -> List[Point]:
    """p1-p2 사이를 직선 보간한 정수 격자 점들을 반환한다."""
    steps = max(abs(p2[0] - p1[0]), abs(p2[1] - p1[1]))
    if steps <= 1:
        return []
    points: List[Point] = []
    for i in range(1, steps):
        t = i / steps
        points.append(
            (
                int(round(p1[0] + (p2[0] - p1[0]) * t)),
                int(round(p1[1] + (p2[1] - p1[1]) * t)),
            )
        )
    return points


def heal_broken_lines(points: List[Point], max_gap: int = 2) -> Tuple[List[Point], int]:
    """작은 선 끊어짐(끝점 간 근접)을 자동 보정해 연결점을 추가한다."""
    components = find_connected_components(points)
    if len(components) <= 1:
        return points, 0

    added: set[Point] = set()
    endpoint_map: List[Tuple[int, Point]] = []
    for idx, comp in enumerate(components):
        for ep in find_endpoints(comp):
            endpoint_map.append((idx, ep))

    # 서로 다른 component의 끝점이 max_gap 이내면 연결선 삽입
    for i in range(len(endpoint_map)):
        ci, p1 = endpoint_map[i]
        for j in range(i + 1, len(endpoint_map)):
            cj, p2 = endpoint_map[j]
            if ci == cj:
                continue
            if max(abs(p1[0] - p2[0]), abs(p1[1] - p2[1])) <= max_gap:
                for bridge in _interpolate_points(p1, p2):
                    added.add(bridge)

    if not added:
        return points, 0

    merged = list(set(points).union(added))
    return merged, len(added)


def find_turtle_line(comp1: List[Point], comp2: List[Point]) -> List[Point]:
    """두 component 중 거북이 선을 찾는다.

    거북이 선: 가장 아래(y 최대)인 점을 포함하는 component.
    """
    all_points = [(p, 1) for p in comp1] + [(p, 2) for p in comp2]
    lowest_point, which = max(all_points, key=lambda t: t[0][1])
    return comp1 if which == 1 else comp2


def _shortest_path_in_component(
    component: List[Point], start: Point, end: Point
) -> List[Point]:
    """component 내부에서 start->end 최단 경로(BFS)를 반환."""
    if start == end:
        return [start]
    point_set = set(component)
    if start not in point_set or end not in point_set:
        return []

    q = deque([start])
    parent: dict[Point, Point | None] = {start: None}
    while q:
        cur = q.popleft()
        if cur == end:
            break
        for nxt in get_neighbors(cur, point_set):
            if nxt in parent:
                continue
            parent[nxt] = cur
            q.append(nxt)

    if end not in parent:
        return []

    path: List[Point] = []
    cur: Point | None = end
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path


def _farthest_point_by_steps(component: List[Point], start: Point) -> Point:
    """BFS hop 기준으로 start에서 가장 먼 점을 찾는다."""
    point_set = set(component)
    if start not in point_set:
        return start

    q = deque([start])
    dist: dict[Point, int] = {start: 0}
    farthest = start
    while q:
        cur = q.popleft()
        if dist[cur] > dist[farthest]:
            farthest = cur
        elif dist[cur] == dist[farthest]:
            # 같은 거리면 더 아래쪽(y 큰 점)을 우선
            if cur[1] > farthest[1]:
                farthest = cur
        for nxt in get_neighbors(cur, point_set):
            if nxt in dist:
                continue
            dist[nxt] = dist[cur] + 1
            q.append(nxt)

    return farthest


def find_tlsp(turtle_component: List[Point]) -> Tuple[Point, List[Point]]:
    """거북이 선의 TLSP를 찾고 경로를 정렬한다.

    TLSP: 끝점 중 더 아래(y 최대)인 점.
    순환 구조나 분기점이 있으면 예외처리.
    """
    # 가장 아래(y 최대)인 점 찾기
    max_y_point = max(turtle_component, key=lambda p: p[1])
    max_y = max_y_point[1]
    
    # 같은 y값을 가진 점들 중에서 끝점 찾기
    candidates = [p for p in turtle_component if p[1] == max_y]
    
    endpoints = find_endpoints(turtle_component)
    if endpoints:
        # 끝점 중에서 가장 아래인 것 선택
        tlsp_candidates = [ep for ep in endpoints if ep[1] == max_y]
        if tlsp_candidates:
            tlsp = tlsp_candidates[0]
        else:
            # 끝점 중 가장 아래
            tlsp = max(endpoints, key=lambda p: p[1])
    else:
        # 순환 구조: 가장 아래 점 선택
        tlsp = max_y_point
    
    # 다른 끝점 중 TLSP에서 hop 기준 가장 먼 점을 선택
    other_endpoints = [ep for ep in endpoints if ep != tlsp]
    if other_endpoints:
        # BFS 거리 기반으로 실제 연결 경로가 긴 끝점을 선택
        best_path: List[Point] = []
        for ep in other_endpoints:
            candidate = _shortest_path_in_component(turtle_component, tlsp, ep)
            if len(candidate) > len(best_path):
                best_path = candidate
        path = best_path
    else:
        # 끝점이 명확하지 않으면 TLSP에서 가장 먼 점까지 경로 사용
        end_point = _farthest_point_by_steps(turtle_component, tlsp)
        path = _shortest_path_in_component(turtle_component, tlsp, end_point)

    # 경로 복원 실패 시 연결 그래프 순회 기반 fallback
    if len(path) < 2:
        path = _build_ordered_path(turtle_component, tlsp)

    if len(path) < 2:
        raise Strategy2Error("Turtle-line path is too short.")

    return tlsp, path


def _build_ordered_path(component: List[Point], start: Point) -> List[Point]:
    """순서대로 경로 생성: 시작점에서 연결된 점들을 순차적으로 따라가며 경로 생성."""
    from typing import Set

    point_set = set(component)
    path: List[Point] = [start]
    visited: Set[Point] = {start}
    current = start

    # 시작점에서 끝점까지 순차적으로 따라가기
    while True:
        neighbors = get_neighbors(current, point_set)
        unvisited_neighbors = [n for n in neighbors if n not in visited]
        
        if not unvisited_neighbors:
            break
        
        # 다음 점 선택 전략:
        # 1. 이웃이 1개면 그대로 선택
        # 2. 여러 이웃이 있으면:
        #    - 현재 방향과 일치하는 이웃 우선
        #    - 없으면 가장 가까운 이웃
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
                    ) == prev_dir
                ]
                if same_dir_neighbors:
                    next_point = same_dir_neighbors[0]
                else:
                    # 방향이 다르면 첫 번째 이웃 선택
                    next_point = unvisited_neighbors[0]
            else:
                next_point = unvisited_neighbors[0]
        
        visited.add(next_point)
        path.append(next_point)
        current = next_point

    return path


def _build_component_plot_path(component: List[Point]) -> List[Point]:
    """component를 시각화용 연속 경로로 정렬한다."""
    if not component:
        return []
    endpoints = find_endpoints(component)
    start = endpoints[0] if endpoints else component[0]
    path = _build_ordered_path(component, start)
    if len(path) < 2 and len(component) >= 2:
        point_set = set(component)
        start = component[0]
        end = max(component[1:], key=lambda p: distance(start, p))
        path = _shortest_path_in_component(list(point_set), start, end)
    return path if path else component[:]


def find_front_head_and_upper_head(
    path: List[Point], fh: int, uh: int
) -> Tuple[List[Point], List[Point], List[str]]:
    """거북이 선 경로에서 앞머리/윗머리 구간을 탐색하고 자동 보정한다."""
    warnings: List[str] = []

    # 1) strict 탐색
    front_head = find_first_vertical_run(path, int(fh))
    used_fh = int(fh)

    # 2) 자동 보정: 임계값 완화
    if not front_head:
        for ratio in (0.85, 0.7):
            relaxed = max(3, int(round(fh * ratio)))
            front_head = find_first_vertical_run(path, relaxed)
            if front_head:
                used_fh = relaxed
                warnings.append(
                    f"Auto-correction applied: FH threshold relaxed from {int(fh)} to {relaxed}."
                )
                break
    if not front_head:
        # 마지막 fallback: 가장 긴 세로 구간 선택
        vertical_runs = [pts for d, pts in detect_straight_runs(path) if d == "vertical"]
        if vertical_runs:
            front_head = max(vertical_runs, key=len)
            used_fh = len(front_head)
            warnings.append(
                "Auto-correction applied: strict FH matching failed; selected the longest vertical run."
            )
        else:
            raise Strategy2Error(f"Unable to find a vertical segment satisfying FH={fh}.")

    # 분기/애매한 구간 진단
    runs = detect_straight_runs(path)
    candidate_v = [pts for d, pts in runs if d == "vertical" and len(pts) >= max(3, int(used_fh * 0.8))]
    if len(candidate_v) > 1:
        warnings.append(
            f"Ambiguous branch detected: {len(candidate_v)} vertical candidates found near FH."
        )

    front_head_end_idx = path.index(front_head[-1])
    remaining_path = path[front_head_end_idx + 1 :]
    if len(remaining_path) < 2:
        raise Strategy2Error("Path after front-head segment is too short.")

    upper_head = find_first_horizontal_run(remaining_path, int(uh))
    used_uh = int(uh)
    if not upper_head:
        for ratio in (0.85, 0.7):
            relaxed = max(3, int(round(uh * ratio)))
            upper_head = find_first_horizontal_run(remaining_path, relaxed)
            if upper_head:
                used_uh = relaxed
                warnings.append(
                    f"Auto-correction applied: UH threshold relaxed from {int(uh)} to {relaxed}."
                )
                break
    if not upper_head:
        horizontal_runs = [
            pts for d, pts in detect_straight_runs(remaining_path) if d == "horizontal"
        ]
        if horizontal_runs:
            upper_head = max(horizontal_runs, key=len)
            used_uh = len(upper_head)
            warnings.append(
                "Auto-correction applied: strict UH matching failed; selected the longest horizontal run."
            )
        else:
            raise Strategy2Error(
                f"Unable to find a horizontal segment satisfying UH={uh}."
            )

    candidate_h = [
        pts
        for d, pts in detect_straight_runs(remaining_path)
        if d == "horizontal" and len(pts) >= max(3, int(used_uh * 0.8))
    ]
    if len(candidate_h) > 1:
        warnings.append(
            f"Ambiguous branch detected: {len(candidate_h)} horizontal candidates found near UH."
        )

    return front_head, upper_head, warnings


def compute_mv(front_head: List[Point], upper_head: List[Point]) -> Point:
    """가상 마커 Mv 계산.

    FH 직선(x=상수)과 UH 직선(y=상수)의 교차점.
    """
    return compute_line_intersection(front_head, upper_head)


def find_bsp(turtle_path: List[Point], mv_shifted: Point) -> Point:
    """BSP 찾기: 거북이 선의 점 중 mv_shifted에 가장 가까운 점."""
    if not turtle_path:
        raise Strategy2Error("Turtle-line path is empty.")

    return min(turtle_path, key=lambda p: distance(p, mv_shifted))


def run_strategy2_on_points(
    points: List[Point], config: Strategy2Config
) -> Strategy2Result:
    """점 집합에 대해 전략 2를 실행."""
    warnings: List[str] = []

    # 1. Connected component 찾기 + 선 끊어짐 자동 보정
    healed_points, added_bridge_points = heal_broken_lines(points, max_gap=max(1, int(config.heal_gap)))
    if added_bridge_points > 0:
        warnings.append(
            f"Auto-correction applied: healed line breaks by inserting {added_bridge_points} bridge points."
        )
    components = find_connected_components(healed_points)

    longest_two_info = summarize_longest_two_lines(components)

    # 2. 가장 긴 두 개의 선 선택
    comp1, comp2 = pick_two_longest_lines(components)

    # 3. 거북이 선 찾기
    turtle_component = find_turtle_line(comp1, comp2)
    turtle_lowest_point = max(turtle_component, key=lambda p: p[1])

    # 4. TLSP 찾기 및 경로 정렬
    tlsp, turtle_path = find_tlsp(turtle_component)

    # 5 & 6. 앞머리끝과 윗머리끝 찾기
    front_head, upper_head, run_warnings = find_front_head_and_upper_head(
        turtle_path, config.FH, config.UH
    )
    warnings.extend(run_warnings)

    # 7. 가상 마커 Mv 계산
    mv = compute_mv(front_head, upper_head)

    # 8. Mv 평행이동 및 BSP 찾기
    mv_shifted = (int(round(mv[0] + config.SX)), int(round(mv[1] + config.SY)))
    bsp = find_bsp(turtle_path, mv_shifted)
    mv_bsp_dx = bsp[0] - mv_shifted[0]
    mv_bsp_dy = bsp[1] - mv_shifted[1]
    mv_bsp_dist = distance(mv_shifted, bsp)

    # 9. BSP에서 TLSP 방향의 반대 방향으로 경로 탐색
    bsp_idx = turtle_path.index(bsp)
    tlsp_idx = turtle_path.index(tlsp)

    # TLSP로 가는 방향의 반대 방향으로 진행
    if bsp_idx < tlsp_idx:
        # TLSP가 오른쪽(인덱스 증가 방향)이므로 반대는 왼쪽으로 이동
        sampling_path = list(reversed(turtle_path[: bsp_idx + 1]))
    elif bsp_idx > tlsp_idx:
        # TLSP가 왼쪽(인덱스 감소 방향)이므로 반대는 오른쪽으로 이동
        sampling_path = turtle_path[bsp_idx:]
    else:
        # BSP==TLSP면 더 긴 쪽으로 진행
        left_len = bsp_idx + 1
        right_len = len(turtle_path) - bsp_idx
        if right_len >= left_len:
            sampling_path = turtle_path[bsp_idx:]
        else:
            sampling_path = list(reversed(turtle_path[: bsp_idx + 1]))

    # PBL개 점 샘플링 (경로의 모든 점 순회 후 1픽셀 간격)
    bending_points = sample_path_at_intervals(
        sampling_path, 0, config.PBL, config.sample_step
    )

    # 거북이선 외 상위 긴 선들(최대 3개)을 시각화용 경로로 준비
    support_line_paths: List[List[Point]] = []
    turtle_set = set(turtle_component)
    ranked = sorted(components, key=len, reverse=True)
    for comp in ranked:
        if set(comp) == turtle_set:
            continue
        support_line_paths.append(_build_component_plot_path(comp))
        if len(support_line_paths) >= 3:
            break

    return Strategy2Result(
        tlsp=tlsp,
        turtle_line_path=turtle_path,
        front_head_run=front_head,
        upper_head_run=upper_head,
        mv=mv,
        mv_shifted=mv_shifted,
        bsp=bsp,
        bending_points=bending_points,
        turtle_lowest_point=turtle_lowest_point,
        turtle_line_length=len(turtle_path),
        longest_two_lines_info=longest_two_info,
        support_line_paths=support_line_paths,
        warnings=warnings,
        mv_bsp_dx=mv_bsp_dx,
        mv_bsp_dy=mv_bsp_dy,
        mv_bsp_distance=mv_bsp_dist,
    )


def run_strategy2_on_file(path: str, config: Strategy2Config) -> Strategy2Result:
    """TXT 파일에 대해 전략 2를 실행."""
    points = parse_txt_points(path)
    return run_strategy2_on_points(points, config)


def save_result_points_txt(path: str, result: Strategy2Result) -> None:
    """결과를 TXT 파일로 저장.

    포맷: x,y,index
    - index는 1..PBL 순서번호를 명시적으로 저장
    - 비디오/시계열 후처리에서 frame 간 motion 변화 추적에 사용 가능
    """
    with open(path, "w", encoding="utf-8") as f:
        f.write("# x,y,index\n")
        for idx, p in enumerate(result.bending_points, start=1):
            f.write(f"{p[0]},{p[1]},{idx}\n")
