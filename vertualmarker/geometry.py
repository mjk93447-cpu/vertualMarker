"""
기하학적 유틸리티 함수들.

점 기반 그래프 구조에서의 연결성, 경로 탐색, 직선 구간 탐지 등을 제공.
"""
import math
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

Point = Tuple[int, int]  # 픽셀 좌표는 정수


def distance(p1: Point, p2: Point) -> float:
    """두 점 사이의 유클리드 거리."""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def are_adjacent(p1: Point, p2: Point) -> bool:
    """8-이웃 인접성 확인.

    두 점이 가로/세로/대각선으로 인접한지 확인.
    """
    dx = abs(p1[0] - p2[0])
    dy = abs(p1[1] - p2[1])
    return dx <= 1 and dy <= 1 and (dx + dy > 0)


def get_neighbors(p: Point, point_set: Set[Point]) -> List[Point]:
    """점 p의 8-이웃 중 point_set에 포함된 점들을 반환."""
    neighbors = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            neighbor = (p[0] + dx, p[1] + dy)
            if neighbor in point_set:
                neighbors.append(neighbor)
    return neighbors


def find_connected_components(points: List[Point]) -> List[List[Point]]:
    """점 집합을 8-이웃 연결성으로 connected component로 분할."""
    point_set = set(points)
    visited: Set[Point] = set()
    components: List[List[Point]] = []

    for start in points:
        if start in visited:
            continue

        # BFS로 연결된 모든 점 찾기
        component: List[Point] = []
        queue = deque([start])
        visited.add(start)

        while queue:
            p = queue.popleft()
            component.append(p)

            for neighbor in get_neighbors(p, point_set):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        if component:
            components.append(component)

    return components


def get_degree(p: Point, point_set: Set[Point]) -> int:
    """점 p의 차수(인접한 점의 개수)."""
    return len(get_neighbors(p, point_set))


def find_endpoints(component: List[Point]) -> List[Point]:
    """Connected component에서 끝점들을 찾는다.

    기본적으로 차수 1인 점을 끝점으로 정의.
    순환 구조나 분기점이 있는 경우 예외처리:
    - 순환 구조: 모든 점의 차수가 2 이상인 경우, 가장 아래(y 최대)인 점을 끝점으로 선택
    - 분기점: 차수 3 이상인 점은 무시하고, 가장 긴 경로만 고려
    """
    point_set = set(component)
    endpoints: List[Point] = []

    # 차수 1인 점들 찾기
    for p in component:
        if get_degree(p, point_set) == 1:
            endpoints.append(p)

    # 순환 구조 처리: 차수 1인 점이 없으면 가장 아래 점 선택
    if not endpoints:
        endpoints = [max(component, key=lambda p: p[1])]

    return endpoints


def find_path_bfs(
    component: List[Point],
    start: Point,
    end: Optional[Point] = None,
) -> List[Point]:
    """BFS로 start에서 end까지의 경로를 찾는다.

    end가 None이면 start에서 가장 먼(hop count) 점을 찾아 그 경로를 반환.
    parent 추적을 사용해 O(n) 메모리/시간으로 동작.
    """
    point_set = set(component)
    visited: Set[Point] = {start}
    parent: Dict[Point, Optional[Point]] = {start: None}
    queue = deque([start])
    farthest = start
    max_hops = 0
    hop_count: Dict[Point, int] = {start: 0}

    while queue:
        current = queue.popleft()
        hops = hop_count[current]

        if end is not None and current == end:
            break

        if hops > max_hops:
            max_hops = hops
            farthest = current

        for neighbor in get_neighbors(current, point_set):
            if neighbor not in visited:
                visited.add(neighbor)
                parent[neighbor] = current
                hop_count[neighbor] = hops + 1
                queue.append(neighbor)

    # Determine target
    target = end if (end is not None and end in parent) else farthest

    # Reconstruct path
    path: List[Point] = []
    p: Optional[Point] = target
    while p is not None:
        path.append(p)
        p = parent[p]
    return path[::-1]


def find_longest_path(
    component: List[Point], start: Point, end: Optional[Point] = None
) -> List[Point]:
    """시작점에서 끝점까지의 경로를 BFS로 찾는다.

    end가 None이면 start에서 가장 먼 점까지의 경로를 반환.
    """
    return find_path_bfs(component, start, end)


def find_longest_path_with_branching(
    component: List[Point], start: Point, end: Optional[Point] = None
) -> List[Point]:
    """BFS를 사용해 start에서 end까지의 경로를 찾는다.

    end가 None이면 start에서 가장 먼 점까지의 경로를 반환.
    분기점이 있어도 BFS로 올바른 경로를 찾는다.
    """
    return find_path_bfs(component, start, end)


def detect_straight_runs(
    path: List[Point],
    angle_tolerance_deg: float = 0.0,
) -> List[Tuple[str, List[Point]]]:
    """경로에서 직선 구간들을 탐지.

    반환: [(direction, points), ...]
    direction: 'horizontal', 'vertical', 'diagonal'

    angle_tolerance_deg > 0 이면 각도 허용 범위 내에서 방향을 결정.
    예: tolerance=10이면, y축에서 10도 이내는 vertical로 인정.
    """
    if len(path) < 2:
        return []

    tolerance_rad = math.radians(angle_tolerance_deg) if angle_tolerance_deg > 0 else 0.0

    def classify_direction(dx: int, dy: int) -> str:
        if dx == 0 and dy == 0:
            return "none"

        if tolerance_rad > 0:
            # 각도 기반 판정
            angle = math.atan2(abs(dy), abs(dx))  # 0 = horizontal, pi/2 = vertical
            if angle >= (math.pi / 2 - tolerance_rad):
                return "vertical"
            elif angle <= tolerance_rad:
                return "horizontal"
            else:
                return "diagonal"
        else:
            # 정확한 판정
            if dx == 0:
                return "vertical"
            elif dy == 0:
                return "horizontal"
            else:
                return "diagonal"

    runs: List[Tuple[str, List[Point]]] = []
    current_run: List[Point] = [path[0]]
    current_dir: Optional[str] = None

    for i in range(1, len(path)):
        p1 = path[i - 1]
        p2 = path[i]
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]

        direction = classify_direction(dx, dy)

        if direction == "none":
            # 같은 점 — 현재 구간에 추가
            current_run.append(p2)
            continue

        if direction == current_dir:
            current_run.append(p2)
        else:
            if current_dir and current_dir != "none" and len(current_run) >= 2:
                runs.append((current_dir, current_run[:]))
            current_run = [p1, p2]
            current_dir = direction

    # 마지막 구간 추가
    if current_dir and current_dir != "none" and len(current_run) >= 2:
        runs.append((current_dir, current_run[:]))

    return runs


def find_first_vertical_run(
    path: List[Point],
    min_length: int,
    angle_tolerance_deg: float = 0.0,
) -> Optional[List[Point]]:
    """경로에서 첫 번째 (거의) 세로 직선 구간을 찾는다.

    min_length: 최소 점 개수
    angle_tolerance_deg: 각도 허용 범위 (0이면 정확한 세로만)
    """
    # 각도 허용이 있으면 직접 스캔 (더 정확하고 연속된 near-vertical 구간을 잡음)
    if angle_tolerance_deg > 0:
        return _find_first_near_direction_run(
            path, min_length, "vertical", angle_tolerance_deg
        )
    # 정확한 매칭
    runs = detect_straight_runs(path, angle_tolerance_deg=0.0)
    for direction, points in runs:
        if direction == "vertical" and len(points) >= min_length:
            return points
    return None


def find_first_horizontal_run(
    path: List[Point],
    min_length: int,
    angle_tolerance_deg: float = 0.0,
) -> Optional[List[Point]]:
    """경로에서 첫 번째 (거의) 가로 직선 구간을 찾는다.

    min_length: 최소 점 개수
    angle_tolerance_deg: 각도 허용 범위 (0이면 정확한 가로만)
    """
    if angle_tolerance_deg > 0:
        return _find_first_near_direction_run(
            path, min_length, "horizontal", angle_tolerance_deg
        )
    runs = detect_straight_runs(path, angle_tolerance_deg=0.0)
    for direction, points in runs:
        if direction == "horizontal" and len(points) >= min_length:
            return points
    return None


def _find_first_near_direction_run(
    path: List[Point],
    min_length: int,
    target_direction: str,
    angle_tolerance_deg: float,
) -> Optional[List[Point]]:
    """경로에서 첫 번째 near-vertical 또는 near-horizontal 연속 구간을 찾는다.

    개별 스텝의 각도를 확인하여 허용 범위 내이면 해당 방향으로 누적.
    연속으로 누적된 점의 수가 min_length 이상이면 반환.
    """
    if len(path) < 2:
        return None

    tolerance_rad = math.radians(angle_tolerance_deg)

    current_run: List[Point] = [path[0]]

    for i in range(1, len(path)):
        dx = path[i][0] - path[i - 1][0]
        dy = path[i][1] - path[i - 1][1]

        if dx == 0 and dy == 0:
            current_run.append(path[i])
            continue

        is_match = False
        if target_direction == "vertical":
            # near-vertical: angle from y-axis <= tolerance
            if dy != 0:
                angle_from_y = abs(math.atan2(abs(dx), abs(dy)))
                is_match = angle_from_y <= tolerance_rad
            # dy == 0 이면 완전 수평이므로 vertical이 아님
        elif target_direction == "horizontal":
            # near-horizontal: angle from x-axis <= tolerance
            if dx != 0:
                angle_from_x = abs(math.atan2(abs(dy), abs(dx)))
                is_match = angle_from_x <= tolerance_rad
            # dx == 0 이면 완전 수직이므로 horizontal이 아님

        if is_match:
            current_run.append(path[i])
        else:
            if len(current_run) >= min_length:
                return current_run
            current_run = [path[i]]

    if len(current_run) >= min_length:
        return current_run
    return None


def compute_line_intersection(
    vertical_points: List[Point], horizontal_points: List[Point]
) -> Point:
    """세로 직선과 가로 직선의 교차점 계산.

    세로 직선: x = constant
    가로 직선: y = constant
    교차점: (vertical_x, horizontal_y)
    """
    # 세로 직선의 x값 (평균)
    vx = sum(p[0] for p in vertical_points) / len(vertical_points)

    # 가로 직선의 y값 (평균)
    hy = sum(p[1] for p in horizontal_points) / len(horizontal_points)

    return (int(round(vx)), int(round(hy)))


def sample_path_at_intervals(
    path: List[Point], start_index: int, num_samples: int, step: float
) -> List[Point]:
    """경로를 따라가면서 일정 간격으로 샘플링.

    경로의 모든 점을 순회한 후, 1픽셀 간격으로 샘플링.
    """
    if not path or num_samples <= 0:
        return []

    # start_index부터 경로 끝까지
    remaining_path = path[start_index:]
    if not remaining_path:
        return []

    # 경로의 모든 점을 순회하면서 거리 누적
    sampled: List[Point] = []
    accumulated_dist = 0.0
    target_dist = 0.0

    for i in range(len(remaining_path)):
        if i > 0:
            accumulated_dist += distance(remaining_path[i - 1], remaining_path[i])

        while accumulated_dist >= target_dist and len(sampled) < num_samples:
            # 선형 보간
            if i == 0:
                sampled.append(remaining_path[0])
            else:
                seg_len = distance(remaining_path[i - 1], remaining_path[i])
                if seg_len > 0:
                    t = (target_dist - (accumulated_dist - seg_len)) / seg_len
                    t = max(0.0, min(1.0, t))
                    p1 = remaining_path[i - 1]
                    p2 = remaining_path[i]
                    interp = (
                        int(round(p1[0] + (p2[0] - p1[0]) * t)),
                        int(round(p1[1] + (p2[1] - p1[1]) * t)),
                    )
                    sampled.append(interp)
                else:
                    sampled.append(remaining_path[i])
            target_dist += step

        if len(sampled) >= num_samples:
            break

    # 부족하면 마지막 점 반복
    while len(sampled) < num_samples:
        sampled.append(remaining_path[-1])

    return sampled
