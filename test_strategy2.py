"""
Test script for Strategy 2 turtle line detection.
Tests with synthetic data to verify the algorithm works correctly.
"""
import os
import sys
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(name)s - %(message)s")

from vertualmarker.data_generator import SyntheticParams, generate_turtle_and_partner, save_points_txt
from vertualmarker.strategy2 import (
    Strategy2Config,
    Strategy2Error,
    run_strategy2_on_file,
    run_strategy2_on_points,
    parse_txt_points,
    pick_two_longest_lines,
    find_turtle_line,
    find_tlsp,
)
from vertualmarker.geometry import (
    find_connected_components,
    find_endpoints,
    find_path_bfs,
)


def make_connected_line(segments):
    """Create a connected line from segment descriptions.
    Each segment: ('v', x, y_start, y_end) for vertical,
                  ('h', y, x_start, x_end) for horizontal,
                  ('d', x_start, y_start, x_end, y_end) for diagonal.
    Ensures 8-connectivity between consecutive segments.
    """
    points = []
    for seg in segments:
        if seg[0] == 'v':
            _, x, y_start, y_end = seg
            step = 1 if y_end >= y_start else -1
            for y in range(y_start, y_end + step, step):
                p = (x, y)
                if p not in points:
                    points.append(p)
        elif seg[0] == 'h':
            _, y, x_start, x_end = seg
            step = 1 if x_end >= x_start else -1
            for x in range(x_start, x_end + step, step):
                p = (x, y)
                if p not in points:
                    points.append(p)
        elif seg[0] == 'd':
            _, x_start, y_start, x_end, y_end = seg
            dx = x_end - x_start
            dy = y_end - y_start
            steps = max(abs(dx), abs(dy))
            for i in range(steps + 1):
                x = x_start + round(dx * i / steps)
                y = y_start + round(dy * i / steps)
                p = (x, y)
                if p not in points:
                    points.append(p)
    return points


def test_bfs_path_finding():
    """Test BFS path finding on an L-shaped path with a branch."""
    print("=" * 60)
    print("TEST 1: BFS path finding on L-shape")
    print("=" * 60)

    # L-shape: vertical (100, 0..50), horizontal (100..200, 50), branch (101..109, 25)
    points = []
    for y in range(51):
        points.append((100, y))
    for x in range(101, 201):
        points.append((x, 50))
    for x in range(101, 110):
        points.append((x, 25))

    print(f"Created {len(points)} points (L-shape with branch)")

    components = find_connected_components(points)
    assert len(components) == 1, f"Expected 1 component, got {len(components)}"
    comp = components[0]

    endpoints = find_endpoints(comp)
    print(f"Endpoints: {endpoints}")

    # Start from endpoint with highest y
    tlsp = max(endpoints, key=lambda p: p[1])
    print(f"TLSP: ({tlsp[0]},{tlsp[1]})")

    path = find_path_bfs(comp, tlsp, end=None)
    print(f"BFS path: length={len(path)}, start=({path[0][0]},{path[0][1]}), end=({path[-1][0]},{path[-1][1]})")

    # Path from (200,50) → (100,50) → (100,0), ~151 points
    if len(path) >= 140:
        print(f"  Path length {len(path)} >= 140: OK")
    else:
        print(f"  Path length {len(path)} < 140: PROBLEM!")
        return False

    print("\nTEST 1 PASSED")
    return True


def test_well_separated_lines():
    """Test with two clearly separated lines."""
    print("\n" + "=" * 60)
    print("TEST 2: Well-separated lines")
    print("=" * 60)

    # Turtle line: vertical head + diagonal transition + horizontal body
    turtle = make_connected_line([
        ('v', 100, 400, 300),          # Vertical: (100, 400) down to (100, 300) — 101 points
        ('d', 100, 300, 130, 270),     # Diagonal: connect to horizontal
        ('h', 270, 130, 500),          # Horizontal body
    ])

    # Partner line: same shape, shifted UP by 200 (well separated)
    partner = make_connected_line([
        ('v', 100, 200, 100),          # Vertical
        ('d', 100, 100, 130, 70),      # Diagonal
        ('h', 70, 130, 500),           # Horizontal body
    ])

    points = turtle + partner
    print(f"Created {len(points)} points")

    components = find_connected_components(points)
    print(f"Found {len(components)} components")
    for i, comp in enumerate(sorted(components, key=len, reverse=True)[:3]):
        bottom = max(comp, key=lambda p: p[1])
        print(f"  Component {i+1}: {len(comp)} points, bottom=({bottom[0]},{bottom[1]})")

    assert len(components) >= 2, f"Expected >= 2 components, got {len(components)}"

    config = Strategy2Config(
        FH=30.0, UH=30.0, SX=0.0, SY=0.0, PBL=200,
        sample_step=1.0, vertical_angle_tolerance=10.0, horizontal_angle_tolerance=10.0,
    )

    try:
        result = run_strategy2_on_points(points, config)
        print(f"\nStrategy 2 SUCCESS:")
        print(f"  TLSP: ({result.tlsp[0]}, {result.tlsp[1]})")
        print(f"  Turtle line path length: {len(result.turtle_line_path)}")
        print(f"  Front head: {len(result.front_head_run)} pts")
        print(f"  Upper head: {len(result.upper_head_run)} pts")
        print(f"  Mv: ({result.mv[0]}, {result.mv[1]})")
        print(f"  BSP: ({result.bsp[0]}, {result.bsp[1]})")
        print(f"  Bending points: {len(result.bending_points)}")
    except Strategy2Error as e:
        print(f"\nStrategy 2 FAILED: {e}")
        return False

    print("\nTEST 2 PASSED")
    return True


def test_noisy_data():
    """Test with noisier synthetic data."""
    print("\n" + "=" * 60)
    print("TEST 3: Noisy synthetic data (data generator)")
    print("=" * 60)

    params = SyntheticParams(noise=2.0, offset_y=-200)
    points = generate_turtle_and_partner(params)
    print(f"Generated {len(points)} noisy points")

    components = find_connected_components(points)
    print(f"Found {len(components)} components")
    for i, comp in enumerate(sorted(components, key=len, reverse=True)[:3]):
        bottom = max(comp, key=lambda p: p[1])
        print(f"  Component {i+1}: {len(comp)} points, bottom=({bottom[0]},{bottom[1]})")

    config = Strategy2Config(
        FH=30.0, UH=30.0, SX=0.0, SY=0.0, PBL=500,
        sample_step=1.0, vertical_angle_tolerance=10.0, horizontal_angle_tolerance=10.0,
    )

    try:
        result = run_strategy2_on_points(points, config)
        print(f"\nStrategy 2 SUCCESS:")
        print(f"  TLSP: ({result.tlsp[0]}, {result.tlsp[1]})")
        print(f"  Turtle line path length: {len(result.turtle_line_path)}")
        print(f"  Mv: ({result.mv[0]}, {result.mv[1]})")
        print(f"  Bending points: {len(result.bending_points)}")
    except Strategy2Error as e:
        print(f"\nStrategy 2 FAILED: {e}")
        return False

    print("\nTEST 3 PASSED")
    return True


def test_diagonal_connectivity():
    """Test that diagonal connections in paths are properly handled."""
    print("\n" + "=" * 60)
    print("TEST 4: Diagonal path connectivity")
    print("=" * 60)

    # Turtle line with proper 8-connectivity through diagonal
    turtle = make_connected_line([
        ('v', 100, 300, 200),          # Vertical: 101 pts
        ('d', 100, 200, 150, 150),     # Diagonal: 51 pts
        ('h', 150, 150, 500),          # Horizontal: 351 pts
    ])

    # Partner line well separated (y shifted up by 200)
    partner = make_connected_line([
        ('v', 100, 100, 0),
        ('d', 100, 0, 150, -50),
        ('h', -50, 150, 500),
    ])

    points = turtle + partner
    print(f"Created {len(points)} points (with diagonal)")

    components = find_connected_components(points)
    print(f"Found {len(components)} components")
    for i, comp in enumerate(sorted(components, key=len, reverse=True)[:3]):
        bottom = max(comp, key=lambda p: p[1])
        print(f"  Component {i+1}: {len(comp)} points, bottom=({bottom[0]},{bottom[1]})")

    config = Strategy2Config(
        FH=30.0, UH=30.0, SX=0.0, SY=0.0, PBL=100,
        sample_step=1.0, vertical_angle_tolerance=10.0, horizontal_angle_tolerance=10.0,
    )

    try:
        result = run_strategy2_on_points(points, config)
        print(f"\nStrategy 2 SUCCESS:")
        print(f"  TLSP: ({result.tlsp[0]}, {result.tlsp[1]})")
        print(f"  Turtle line path length: {len(result.turtle_line_path)}")
        print(f"  Front head: {len(result.front_head_run)} pts")
        print(f"  Upper head: {len(result.upper_head_run)} pts")
        print(f"  Mv: ({result.mv[0]}, {result.mv[1]})")
        print(f"  BSP: ({result.bsp[0]}, {result.bsp[1]})")
    except Strategy2Error as e:
        print(f"\nStrategy 2 FAILED: {e}")
        return False

    print("\nTEST 4 PASSED")
    return True


def test_full_pipeline_with_file():
    """Test the full pipeline using file I/O."""
    print("\n" + "=" * 60)
    print("TEST 5: Full pipeline with file I/O")
    print("=" * 60)

    turtle = make_connected_line([
        ('v', 200, 500, 400),
        ('d', 200, 400, 230, 380),
        ('h', 380, 230, 620),
    ])
    partner = make_connected_line([
        ('v', 200, 300, 200),
        ('d', 200, 200, 230, 180),
        ('h', 180, 230, 620),
    ])

    points = turtle + partner
    test_file = "/tmp/test_pipeline.txt"
    save_points_txt(test_file, points)
    print(f"Saved {len(points)} points to {test_file}")

    config = Strategy2Config(
        FH=30.0, UH=30.0, SX=10.0, SY=10.0, PBL=200,
        sample_step=1.0, vertical_angle_tolerance=10.0, horizontal_angle_tolerance=10.0,
    )

    try:
        result = run_strategy2_on_file(test_file, config)
        print(f"\nStrategy 2 SUCCESS:")
        print(f"  TLSP: ({result.tlsp[0]}, {result.tlsp[1]})")
        print(f"  Turtle line path length: {len(result.turtle_line_path)}")
        print(f"  Mv: ({result.mv[0]}, {result.mv[1]})")
        print(f"  Mv shifted: ({result.mv_shifted[0]}, {result.mv_shifted[1]})")
        print(f"  BSP: ({result.bsp[0]}, {result.bsp[1]})")
        print(f"  Bending points: {len(result.bending_points)}")

        diag = result.diagnostics
        assert diag.get("num_components", 0) >= 2
        assert diag.get("turtle_component_length", 0) > 100
        print(f"\n  Diagnostics OK:")
        print(f"    num_components: {diag.get('num_components')}")
        print(f"    line1: {diag.get('line1_length')} pts, bottom={diag.get('line1_bottom')}")
        print(f"    line2: {diag.get('line2_length')} pts, bottom={diag.get('line2_bottom')}")
    except Strategy2Error as e:
        print(f"\nStrategy 2 FAILED: {e}")
        return False

    print("\nTEST 5 PASSED")
    return True


def test_fragmented_with_noise():
    """Test with fragmented data and noise."""
    print("\n" + "=" * 60)
    print("TEST 6: Fragmented data with noise")
    print("=" * 60)

    import random
    random.seed(42)

    # Turtle line: connected line with vertical + diagonal + horizontal
    turtle = make_connected_line([
        ('v', 500, 300, 200),          # Vertical: 101 pts
        ('d', 500, 200, 530, 170),     # Diagonal transition: ~31 pts
        ('h', 170, 530, 900),          # Horizontal body: 371 pts
    ])

    # Partner line: well separated
    partner = make_connected_line([
        ('v', 500, 50, -50),
        ('d', 500, -50, 530, -80),
        ('h', -80, 530, 900),
    ])

    points = turtle + partner

    # Add isolated noise points
    for _ in range(50):
        x = random.randint(400, 950)
        y = random.randint(-150, 350)
        points.append((x, y))

    print(f"Created {len(points)} points (with noise)")

    components = find_connected_components(points)
    print(f"Found {len(components)} components")
    sizes = sorted([len(c) for c in components], reverse=True)
    print(f"Top 5 sizes: {sizes[:5]}")

    config = Strategy2Config(
        FH=30.0, UH=30.0, SX=0.0, SY=0.0, PBL=200,
        sample_step=1.0, vertical_angle_tolerance=15.0, horizontal_angle_tolerance=15.0,
    )

    try:
        result = run_strategy2_on_points(points, config)
        print(f"\nStrategy 2 SUCCESS:")
        print(f"  TLSP: ({result.tlsp[0]}, {result.tlsp[1]})")
        print(f"  Turtle line path length: {len(result.turtle_line_path)}")
        print(f"  Front head: {len(result.front_head_run)} pts")
        print(f"  Upper head: {len(result.upper_head_run)} pts")
        print(f"  Mv: ({result.mv[0]}, {result.mv[1]})")
        print(f"  BSP: ({result.bsp[0]}, {result.bsp[1]})")
        print(f"  Bending points: {len(result.bending_points)}")
    except Strategy2Error as e:
        print(f"\nStrategy 2 FAILED: {e}")
        return False

    print("\nTEST 6 PASSED")
    return True


if __name__ == "__main__":
    results = []
    results.append(("BFS path finding", test_bfs_path_finding()))
    results.append(("Well-separated lines", test_well_separated_lines()))
    results.append(("Noisy data (generator)", test_noisy_data()))
    results.append(("Diagonal connectivity", test_diagonal_connectivity()))
    results.append(("Full pipeline", test_full_pipeline_with_file()))
    results.append(("Fragmented with noise", test_fragmented_with_noise()))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_passed = True
    for name, passed in results:
        status = "PASSED" if passed else "FAILED"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\nAll tests PASSED!")
    else:
        print("\nSome tests FAILED!")
        sys.exit(1)
