from src.functions.grid_lines_func import grid_lines_func
import numpy as np


def test_grid_lines_func():
    """Test basic grid line intersection calculation."""

    # Test line completely inside grid
    x_grid = [0.0, 10.0]
    y_grid = [0.0, 10.0]
    x_line = [2.0, 8.0]
    y_line = [3.0, 7.0]

    f, x_int, y_int = grid_lines_func(x_grid, y_grid, x_line, y_line)

    # Line is completely inside, so fraction should be 1.0
    assert abs(f - 1.0) < 1e-10
    # Intersection points should be the line endpoints
    assert abs(x_int[0] - x_line[0]) < 1e-10
    assert abs(x_int[1] - x_line[1]) < 1e-10
    assert abs(y_int[0] - y_line[0]) < 1e-10
    assert abs(y_int[1] - y_line[1]) < 1e-10


def test_grid_lines_func_no_intersection():
    """Test cases with no intersection."""

    x_grid = [0.0, 10.0]
    y_grid = [0.0, 10.0]

    # Line completely to the left
    x_line = [-5.0, -2.0]
    y_line = [3.0, 7.0]
    f, x_int, y_int = grid_lines_func(x_grid, y_grid, x_line, y_line)
    assert f == 0.0

    # Line completely to the right
    x_line = [15.0, 20.0]
    y_line = [3.0, 7.0]
    f, x_int, y_int = grid_lines_func(x_grid, y_grid, x_line, y_line)
    assert f == 0.0

    # Line completely below
    x_line = [3.0, 7.0]
    y_line = [-5.0, -2.0]
    f, x_int, y_int = grid_lines_func(x_grid, y_grid, x_line, y_line)
    assert f == 0.0

    # Line completely above
    x_line = [3.0, 7.0]
    y_line = [15.0, 20.0]
    f, x_int, y_int = grid_lines_func(x_grid, y_grid, x_line, y_line)
    assert f == 0.0


def test_grid_lines_func_wrong_dimensions():
    """Test error handling for wrong input dimensions."""

    # Wrong x_grid dimensions
    x_grid = [0.0]  # Should be 2 elements
    y_grid = [0.0, 10.0]
    x_line = [2.0, 8.0]
    y_line = [3.0, 7.0]

    f, x_int, y_int = grid_lines_func(x_grid, y_grid, x_line, y_line)
    assert f == 0.0
    assert np.isnan(x_int[0]) and np.isnan(x_int[1])
    assert np.isnan(y_int[0]) and np.isnan(y_int[1])


def test_grid_lines_func_zero_length_line():
    """Test with zero-length line."""

    x_grid = [0.0, 10.0]
    y_grid = [0.0, 10.0]
    x_line = [5.0, 5.0]  # Same point
    y_line = [5.0, 5.0]  # Same point

    f, x_int, y_int = grid_lines_func(x_grid, y_grid, x_line, y_line)
    assert f == 0.0


def test_grid_lines_func_partial_intersection():
    """Test line partially intersecting grid."""

    x_grid = [0.0, 10.0]
    y_grid = [0.0, 10.0]

    # Line starting inside, ending outside
    x_line = [5.0, 15.0]
    y_line = [5.0, 5.0]  # Horizontal line

    f, x_int, y_int = grid_lines_func(x_grid, y_grid, x_line, y_line)

    # Should be partial intersection
    assert 0.0 < f < 1.0
    # One intersection point should be the starting point
    assert abs(x_int[0] - 5.0) < 1e-10
    assert abs(y_int[0] - 5.0) < 1e-10
    # Other intersection point should be at grid boundary
    assert abs(x_int[1] - 10.0) < 1e-10
    assert abs(y_int[1] - 5.0) < 1e-10


def test_grid_lines_func_horizontal_line():
    """Test horizontal line intersections."""

    x_grid = [0.0, 10.0]
    y_grid = [0.0, 10.0]

    # Horizontal line crossing grid
    x_line = [-5.0, 15.0]
    y_line = [5.0, 5.0]

    f, x_int, y_int = grid_lines_func(x_grid, y_grid, x_line, y_line)

    # Should intersect at x boundaries
    assert f > 0.0
    # Check intersection points are at grid boundaries
    intersections_x = sorted([x_int[0], x_int[1]])
    assert abs(intersections_x[0] - 0.0) < 1e-10
    assert abs(intersections_x[1] - 10.0) < 1e-10


def test_grid_lines_func_vertical_line():
    """Test vertical line intersections."""

    x_grid = [0.0, 10.0]
    y_grid = [0.0, 10.0]

    # Vertical line crossing grid
    x_line = [5.0, 5.0]
    y_line = [-5.0, 15.0]

    f, x_int, y_int = grid_lines_func(x_grid, y_grid, x_line, y_line)

    # Should intersect at y boundaries
    assert f > 0.0
    # Check intersection points are at grid boundaries
    intersections_y = sorted([y_int[0], y_int[1]])
    assert abs(intersections_y[0] - 0.0) < 1e-10
    assert abs(intersections_y[1] - 10.0) < 1e-10


def test_grid_lines_func_diagonal_intersection():
    """Test diagonal line with two intersections."""

    x_grid = [0.0, 10.0]
    y_grid = [0.0, 10.0]

    # Diagonal line crossing grid
    x_line = [-5.0, 15.0]
    y_line = [-5.0, 15.0]  # 45-degree line

    f, x_int, y_int = grid_lines_func(x_grid, y_grid, x_line, y_line)

    # Should have some intersection
    assert f > 0.0
    # Length inside grid should be sqrt(2) * 10
    expected_length_inside = np.sqrt(2) * 10
    total_length = np.sqrt((15 - (-5)) ** 2 + (15 - (-5)) ** 2)
    expected_f = expected_length_inside / total_length
    assert abs(f - expected_f) < 1e-10


def test_grid_lines_func_edge_cases():
    """Test various edge cases."""

    x_grid = [0.0, 10.0]
    y_grid = [0.0, 10.0]

    # Line on grid boundary
    x_line = [0.0, 10.0]
    y_line = [0.0, 0.0]  # Bottom edge

    f, x_int, y_int = grid_lines_func(x_grid, y_grid, x_line, y_line)
    assert f >= 0.0  # Should handle boundary cases

    # Line touching corner
    x_line = [-5.0, 0.0]
    y_line = [-5.0, 0.0]

    f, x_int, y_int = grid_lines_func(x_grid, y_grid, x_line, y_line)
    # May have minimal intersection at corner
    assert f >= 0.0


def test_grid_lines_func_one_point_inside():
    """Test case where one endpoint is inside grid."""

    x_grid = [0.0, 10.0]
    y_grid = [0.0, 10.0]

    # First point inside, second outside
    x_line = [3.0, 15.0]
    y_line = [4.0, 8.0]

    f, x_int, y_int = grid_lines_func(x_grid, y_grid, x_line, y_line)

    # Should have partial intersection
    assert 0.0 < f < 1.0

    # One intersection point should be the inside point
    inside_found = False
    if abs(x_int[0] - 3.0) < 1e-10 and abs(y_int[0] - 4.0) < 1e-10:
        inside_found = True
    elif abs(x_int[1] - 3.0) < 1e-10 and abs(y_int[1] - 4.0) < 1e-10:
        inside_found = True
    assert inside_found


def test_grid_lines_func_parallel_shifts():
    """Test parallel line shifts for edge cases."""

    x_grid = [0.0, 10.0]
    y_grid = [0.0, 10.0]

    # Line exactly on grid edge (should trigger shift)
    x_line = [0.0, 0.0]  # Vertical line on left edge
    y_line = [5.0, 15.0]

    f, x_int, y_int = grid_lines_func(x_grid, y_grid, x_line, y_line)

    # Should handle the parallel edge case
    assert f >= 0.0


def test_grid_lines_func_intersection_count():
    """Test proper intersection counting."""

    x_grid = [0.0, 10.0]
    y_grid = [0.0, 10.0]

    # Line that definitely crosses grid
    x_line = [-2.0, 12.0]
    y_line = [2.0, 8.0]

    f, x_int, y_int = grid_lines_func(x_grid, y_grid, x_line, y_line)

    # Should find exactly 2 intersections and non-zero fraction
    assert f > 0.0
    assert not (np.isnan(x_int[0]) or np.isnan(x_int[1]))
    assert not (np.isnan(y_int[0]) or np.isnan(y_int[1]))

    # Verify intersection points are on grid boundary
    for i in range(2):
        x_on_boundary = abs(x_int[i] - 0.0) < 1e-10 or abs(x_int[i] - 10.0) < 1e-10
        y_on_boundary = abs(y_int[i] - 0.0) < 1e-10 or abs(y_int[i] - 10.0) < 1e-10
        assert x_on_boundary or y_on_boundary


def test_grid_lines_func_fraction_calculation():
    """Test fraction calculation accuracy."""

    x_grid = [0.0, 5.0]
    y_grid = [0.0, 5.0]

    # Simple horizontal line crossing entire grid
    x_line = [-1.0, 6.0]
    y_line = [2.5, 2.5]

    f, x_int, y_int = grid_lines_func(x_grid, y_grid, x_line, y_line)

    # Total line length: 7.0, inside grid length: 5.0
    expected_f = 5.0 / 7.0
    assert abs(f - expected_f) < 1e-10
