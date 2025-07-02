import numpy as np


def running_mean_temperature_func(
    T_a: np.ndarray,
    num_hours: float,
    min_time: int,
    max_time: int,
    dt: float,
    use_normal: bool = False,
) -> np.ndarray:
    """
    Calculate running mean temperature over a specified time window.

    Averaging time is typically 1.5 - 3 days.

    Args:
        T_a: Air temperature array
        num_hours: Number of hours for the running mean window
        min_time: Minimum time index
        max_time: Maximum time index
        dt: Time step in hours
        use_normal: If True, use simple backward average; if False, use exponential approach

    Returns:
        np.ndarray: Running mean temperature for the time range [min_time:max_time]
    """
    T_running = T_a[min_time : max_time + 1].copy()

    for ti in range(min_time, max_time + 1):
        if use_normal:
            # Simple backward averaging
            min_running_index = max(0, ti - int(num_hours))
            T_running[ti - min_time] = np.mean(T_a[min_running_index : ti + 1])
        else:
            # Alternative formulation to preserve initial value at min_time index
            # Uses exponential approach
            if ti > min_time:
                prev_idx = max(0, ti - 1) - min_time
                curr_idx = ti - min_time
                T_running[curr_idx] = (
                    T_running[prev_idx] * (1.0 - dt / num_hours)
                    + T_a[ti] * dt / num_hours
                )

    return T_running
