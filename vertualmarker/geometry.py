"""
기하학적 유틸리티 함수들.

점 기반 그래프 구조에서의 연결성, 경로 탐색, 직선 구간 탐지 등을 제공.
"""
import math
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

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


def find_longest_path(
    component: List[Point], start: Point, end: Point | None = None
) -> List[Point]:
    """시작점에서 끝점까지의 가장 긴 경로를 찾는다.

    DFS로 모든 가능한 경로를 탐색하고 가장 긴 것을 선택.
    end가 None이면 start에서 가장 먼 점을 찾는다.
    """
    point_set = set(component)

    def dfs(current: Point, visited: Set[Point], path: List[Point]) -> List[Point]:
        if end is not None and current == end:
            return path[:]

        longest_path = path[:]
        for neighbor in get_neighbors(current, point_set):
            if neighbor not in visited:
                visited.add(neighbor)
                candidate = dfs(neighbor, visited, path + [neighbor])
                visited.remove(neighbor)
                if len(candidate) > len(longest_path):
                    longest_path = candidate

        return longest_path

    if end is None:
        # start에서 가장 먼 점 찾기
        max_dist = -1
        farthest = start
        for p in component:
            if p != start:
                d = distance(start, p)
                if d > max_dist:
                    max_dist = d
                    farthest = p
        end = farthest

    visited = {start}
    path = dfs(start, visited, [start])
    return path


def find_longest_path_with_branching(
    component: List[Point], start: Point, end: Point | None = None
) -> List[Point]:
    """분기점을 고려한 가장 긴 경로 찾기.

    간단한 접근: 끝점이 있으면 끝점까지, 없으면 가장 먼 점까지.
    분기점에서는 가장 긴 경로로 이어지는 분기만 선택.
    """
    point_set = set(component)

    if end is None:
        # start에서 가장 먼 점 찾기
        max_dist = -1
        farthest = start
        for p in component:
            if p != start:
                d = distance(start, p)
                if d > max_dist:
                    max_dist = d
                    farthest = p
        end = farthest

    # 간단한 BFS로 경로 찾기 (visited 체크로 순환 방지)
    parent: Dict[Point, Point | None] = {start: None}
    queue = deque([start])
    visited: Set[Point] = {start}

    while queue:
        current = queue.popleft()

        if current == end:
            # 경로 복원
            path = []
            node = end
            while node is not None:
                path.append(node)
                node = parent[node]
            return list(reversed(path))

        neighbors = get_neighbors(current, point_set)
        # 분기점이면 가장 긴 경로로 이어지는 이웃만 선택
        if get_degree(current, point_set) >= 3:
            # 각 이웃으로 가는 경로의 잠재적 길이 추정
            neighbor_scores = []
            for neighbor in neighbors:
                if neighbor not in visited:
                    # 간단한 휴리스틱: end까지의 거리
                    score = -distance(neighbor, end)
                    neighbor_scores.append((score, neighbor))
            if neighbor_scores:
                neighbors = [n for _, n in sorted(neighbor_scores, reverse=True)[:1]]

        for neighbor in neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                parent[neighbor] = current
                queue.append(neighbor)

    # end에 도달하지 못한 경우, start에서 가장 먼 점까지의 경로 반환
    if visited:
        farthest = max(visited, key=lambda p: distance(start, p))
        path = []
        node = farthest
        while node is not None:
            path.append(node)
            node = parent[node]
        return list(reversed(path))

    return [start]


def detect_straight_runs(path: List[Point]) -> List[Tuple[str, List[Point]]]:
    """경로에서 직선 구간들을 탐지.

    반환: [(direction, points), ...]
    direction: 'horizontal', 'vertical', 'diagonal'
    """
    if len(path) < 2:
        return []

    runs: List[Tuple[str, List[Point]]] = []
    current_run: List[Point] = [path[0]]
    current_dir: str | None = None

    for i in range(1, len(path)):
        p1 = path[i - 1]
        p2 = path[i]
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]

        # 방향 결정
        if dx == 0:
            direction = "vertical"
        elif dy == 0:
            direction = "horizontal"
        else:
            direction = "diagonal"

        if direction == current_dir:
            current_run.append(p2)
        else:
            if current_dir and len(current_run) >= 2:
                runs.append((current_dir, current_run[:]))
            current_run = [p1, p2]
            current_dir = direction

    # 마지막 구간 추가
    if current_dir and len(current_run) >= 2:
        runs.append((current_dir, current_run[:]))

    return runs


def find_first_vertical_run(
    path: List[Point], min_length: int
) -> List[Point] | None:
    """경로에서 첫 번째 세로 직선 구간을 찾는다.

    min_length: 최소 점 개수
    """
    runs = detect_straight_runs(path)
    for direction, points in runs:
        if direction == "vertical" and len(points) >= min_length:
            return points
    return None


def find_first_horizontal_run(
    path: List[Point], min_length: int
) -> List[Point] | None:
    """경로에서 첫 번째 가로 직선 구간을 찾는다.

    min_length: 최소 점 개수
    """
    runs = detect_straight_runs(path)
    for direction, points in runs:
        if direction == "horizontal" and len(points) >= min_length:
            return points
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
