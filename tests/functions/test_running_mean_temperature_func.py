import numpy as np
from src.functions.running_mean_temperature_func import running_mean_temperature_func


def test_running_mean_temperature_func():
    """Test running mean temperature calculation."""

    # Create test temperature data
    T_a = np.array([15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0])
    num_hours = 48.0  # 2-day running mean
    min_time = 0
    max_time = 7
    dt = 1.0  # 1 hour time step

    # Test exponential approach (default)
    result = running_mean_temperature_func(
        T_a, num_hours, min_time, max_time, dt, use_normal=False
    )

    # Result should have same length as input range
    assert len(result) == (max_time - min_time + 1)

    # All values should be reasonable temperatures
    assert np.all(result >= 10.0)
    assert np.all(result <= 25.0)

    # First value should be the original first value
    assert abs(result[0] - T_a[min_time]) < 1e-10


def test_running_mean_temperature_func_constant_input():
    """Test with constant temperature input."""

    # Constant temperature
    T_a = np.array([20.0, 20.0, 20.0, 20.0, 20.0, 20.0])
    num_hours = 24.0
    min_time = 0
    max_time = 5
    dt = 1.0

    # Test exponential approach
    result = running_mean_temperature_func(
        T_a, num_hours, min_time, max_time, dt, use_normal=False
    )

    # All values should be 20.0 (constant input)
    assert np.allclose(result, 20.0)


def test_running_mean_temperature_func_normal_approach():
    """Test normal averaging approach."""

    T_a = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
    num_hours = 8.0
    min_time = 0
    max_time = 5
    dt = 1.0

    result_normal = running_mean_temperature_func(
        T_a, num_hours, min_time, max_time, dt, use_normal=True
    )

    # Should have correct length
    assert len(result_normal) == 6

    # For normal approach, each value should be mean of available data up to that point
    for i in range(len(result_normal)):
        start_idx = max(0, i - int(num_hours))
        expected_mean = np.mean(T_a[start_idx : i + 1])
        assert abs(result_normal[i] - expected_mean) < 1e-10


def test_running_mean_temperature_func_window_size():
    """Test effects of different running mean window sizes."""

    # Create varying temperature data
    T_a = np.array([15.0, 20.0, 10.0, 25.0, 15.0, 20.0])
    min_time = 0
    max_time = 5
    dt = 1.0

    # Short window
    result_short = running_mean_temperature_func(
        T_a, 2.0, min_time, max_time, dt, use_normal=False
    )

    # Long window
    result_long = running_mean_temperature_func(
        T_a, 10.0, min_time, max_time, dt, use_normal=False
    )

    # Both should have same length
    assert len(result_short) == len(result_long) == 6

    # Short window should be more variable
    var_short = np.var(result_short)
    var_long = np.var(result_long)
    assert var_short >= var_long
