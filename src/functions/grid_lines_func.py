import numpy as np
from typing import Tuple, List


def grid_lines_func(
    x_grid: List[float], y_grid: List[float], x_line: List[float], y_line: List[float]
) -> Tuple[float, List[float], List[float]]:
    """
    Calculate the fraction of road length that is in a grid.

    Returns the fraction of the road length that is in a grid
    along with intersection points.

    Args:
        x_grid: Grid x coordinates [x_min, x_max]
        y_grid: Grid y coordinates [y_min, y_max]
        x_line: Line x coordinates [x1, x2]
        y_line: Line y coordinates [y1, y2]

    Returns:
        tuple: (f, x_int, y_int) where:
            f: Fraction of line inside grid (0-1)
            x_int: X coordinates of intersection points [x1, x2] or [NaN, NaN]
            y_int: Y coordinates of intersection points [y1, y2] or [NaN, NaN]
    """
    if len(x_grid) != 2 or len(y_grid) != 2 or len(x_line) != 2 or len(y_line) != 2:
        print("Wrong dimensions in input data")
        return 0.0, [np.nan, np.nan], [np.nan, np.nan]

    length_line = np.sqrt((x_line[0] - x_line[1]) ** 2 + (y_line[0] - y_line[1]) ** 2)
    f = 0.0

    x_int = [np.nan, np.nan]
    y_int = [np.nan, np.nan]

    if length_line == 0.0:
        return f, x_int, y_int

    dx = max(x_grid) - min(x_grid)
    dy = max(y_grid) - min(y_grid)

    # Check first for lines that cannot have an intersection
    if x_line[0] < x_grid[0] and x_line[1] < x_grid[0]:
        return 0.0, x_int, y_int
    if x_line[0] >= x_grid[1] and x_line[1] >= x_grid[1]:
        return 0.0, x_int, y_int
    if y_line[0] < y_grid[0] and y_line[1] < y_grid[0]:
        return 0.0, x_int, y_int
    if y_line[0] >= y_grid[1] and y_line[1] >= y_grid[1]:
        return 0.0, x_int, y_int

    # Check for lines that are completely inside the grid
    if (
        (x_line[0] >= x_grid[0] and x_line[1] >= x_grid[0])
        and (x_line[0] < x_grid[1] and x_line[1] < x_grid[1])
        and (y_line[0] >= y_grid[0] and y_line[1] >= y_grid[0])
        and (y_line[0] < y_grid[1] and y_line[1] < y_grid[1])
    ):
        f = 1.0
        x_int = x_line.copy()
        y_int = y_line.copy()
        return f, x_int, y_int

    # Check for lines with one of the nodes within
    for node in range(2):
        anti_node = 1 - node

        if (x_line[node] >= x_grid[0] and x_line[node] < x_grid[1]) and (
            y_line[node] >= y_grid[0] and y_line[node] < y_grid[1]
        ):
            # This node is in the grid

            # Shift parallel and equal lines when they are on the grid edge
            x_line_work = x_line.copy()
            y_line_work = y_line.copy()

            if x_line[node] == x_line[anti_node] and x_line[node] == x_grid[0]:
                x_line_work[0] += dx * 1e-6
                x_line_work[1] += dx * 1e-6

            if y_line[node] == y_line[anti_node] and y_line[node] == y_grid[0]:
                y_line_work[0] += dy * 1e-6
                y_line_work[1] += dy * 1e-6

            # Can't intersect since it is parallel to the horizontal grid lines
            if y_line_work[node] != y_line_work[anti_node]:
                # Check intersection with the horizontal grid faces
                for node_y_grid in range(2):
                    x_temp = x_line_work[node] + (
                        y_grid[node_y_grid] - y_line_work[node]
                    ) * (x_line_work[anti_node] - x_line_work[node]) / (
                        y_line_work[anti_node] - y_line_work[node]
                    )
                    y_temp = y_grid[node_y_grid]

                    if (
                        y_temp >= min(y_line_work)
                        and y_temp <= max(y_line_work)
                        and y_temp != y_line_work[node]
                        and x_temp >= min(x_grid)
                        and x_temp <= max(x_grid)
                    ):
                        y_int[anti_node] = y_grid[node_y_grid]
                        x_int[anti_node] = x_temp
                        x_int[node] = x_line_work[node]
                        y_int[node] = y_line_work[node]

                        length_int = np.sqrt(
                            (x_int[node] - x_int[anti_node]) ** 2
                            + (y_int[node] - y_int[anti_node]) ** 2
                        )
                        f = length_int / length_line
                        return f, x_int, y_int

            # Can't intersect since it is parallel with the vertical grid lines
            if x_line_work[node] != x_line_work[anti_node]:
                # Check intersection with the vertical grid faces
                for node_x_grid in range(2):
                    y_temp = y_line_work[node] + (
                        x_grid[node_x_grid] - x_line_work[node]
                    ) * (y_line_work[anti_node] - y_line_work[node]) / (
                        x_line_work[anti_node] - x_line_work[node]
                    )
                    x_temp = x_grid[node_x_grid]

                    if (
                        x_temp >= min(x_line_work)
                        and x_temp <= max(x_line_work)
                        and x_temp != x_line_work[node]
                        and y_temp >= min(y_grid)
                        and y_temp <= max(y_grid)
                    ):
                        x_int[anti_node] = x_grid[node_x_grid]
                        y_int[anti_node] = y_temp
                        y_int[node] = y_line_work[node]
                        x_int[node] = x_line_work[node]

                        length_int = np.sqrt(
                            (x_int[node] - x_int[anti_node]) ** 2
                            + (y_int[node] - y_int[anti_node]) ** 2
                        )
                        f = length_int / length_line
                        return f, x_int, y_int

    # Only possibility left is that both nodes are outside the grid
    # Find 2 intersections then
    n_intersection = 0
    node = 0
    anti_node = 1

    if y_line[node] != y_line[anti_node]:  # Can't intersect since it is parallel
        for node_y_grid in range(2):
            # Check intersection with the horizontal grid faces
            x_temp = x_line[node] + (y_grid[node_y_grid] - y_line[node]) * (
                x_line[anti_node] - x_line[node]
            ) / (y_line[anti_node] - y_line[node])
            y_temp = y_grid[node_y_grid]

            if (
                y_temp >= min(y_line)
                and y_temp <= max(y_line)
                and x_temp >= min(x_grid)
                and x_temp <= max(x_grid)
                and n_intersection < 2
            ):
                y_int[n_intersection] = y_temp
                x_int[n_intersection] = x_temp
                n_intersection += 1

    if x_line[node] != x_line[anti_node]:  # Can't intersect since it is parallel
        for node_x_grid in range(2):
            y_temp = y_line[node] + (x_grid[node_x_grid] - x_line[node]) * (
                y_line[anti_node] - y_line[node]
            ) / (x_line[anti_node] - x_line[node])
            x_temp = x_grid[node_x_grid]

            # Use y_temp < max(y_grid) in case it is in one of the corners
            if (
                x_temp >= min(x_line)
                and x_temp <= max(x_line)
                and y_temp >= min(y_grid)
                and y_temp < max(y_grid)
                and n_intersection < 2
            ):
                x_int[n_intersection] = x_temp
                y_int[n_intersection] = y_temp
                n_intersection += 1

    if n_intersection == 2:
        length_int = np.sqrt((x_int[0] - x_int[1]) ** 2 + (y_int[0] - y_int[1]) ** 2)
        f = length_int / length_line
    elif n_intersection == 1:
        # One node is on the edge and f=0
        pass

    return f, x_int, y_int
