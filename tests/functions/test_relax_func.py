import numpy as np
from src.functions.relax_func import relax_func


def test_relax_func():
    """Test relaxation function for energy correction modification."""

    # Test basic case
    dt = 1.0
    hour_of_forecast = 1

    result = relax_func(dt, hour_of_forecast)

    # Should be between 0 and 1
    assert 0 <= result <= 1

    # First hour should give 1.0 (maximum relaxation)
    assert result == 1.0

    # Test at different forecast hours
    dt = 1.0
    results = []
    for hour in range(1, 6):  # Hours 1-5
        result = relax_func(dt, hour)
        results.append(result)
        assert 0 <= result <= 1

    # Should decrease with forecast hour
    assert results[0] >= results[1] >= results[2]

    # Test with different time step
    dt = 0.5  # 30-minute time step
    result_30min = relax_func(dt, 2)

    # Should be between 0 and 1
    assert 0 <= result_30min <= 1

    # Test beyond relaxation period
    dt = 1.0
    hour_beyond = 10  # Well beyond 3-hour relaxation period
    result_beyond = relax_func(dt, hour_beyond)

    # Should be 0.0 after relaxation period
    assert result_beyond == 0.0


def test_relax_func_linear_decrease():
    """Test that relaxation decreases linearly over 3 hours."""

    dt = 1.0  # 1-hour time step

    # Test first 3 hours (should cover relaxation period)
    hour1 = relax_func(dt, 1)
    hour2 = relax_func(dt, 2)
    hour3 = relax_func(dt, 3)
    hour4 = relax_func(dt, 4)

    # Should start at 1.0
    assert hour1 == 1.0

    # Should be 0.0 after 3 hours
    assert hour4 == 0.0

    # Should decrease linearly
    assert hour1 > hour2 > hour3

    # Check approximate linearity
    # With 3 steps over 3 hours: [1.0, 0.5, 0.0] (linspace from 1 to 0 with 3 points)
    expected_hour2 = 0.5  # Middle value in linspace(1, 0, 3)
    expected_hour3 = 0.0  # Last value in linspace(1, 0, 3)

    assert abs(hour2 - expected_hour2) < 0.1
    assert abs(hour3 - expected_hour3) < 0.1


def test_relax_func_different_timesteps():
    """Test relaxation with different time steps."""

    # 30-minute time step (6 steps = 3 hours)
    dt = 0.5
    results_30min = []
    for hour in range(1, 8):
        result = relax_func(dt, hour)
        results_30min.append(result)

    # First value should be 1.0
    assert results_30min[0] == 1.0

    # Should reach 0.0 after 6 steps (3 hours)
    assert results_30min[5] == 0.0
    assert results_30min[6] == 0.0

    # 2-hour time step (2 steps = 4 hours > 3 hours)
    dt = 2.0
    result_2h_step1 = relax_func(dt, 1)
    result_2h_step2 = relax_func(dt, 2)

    assert result_2h_step1 == 1.0
    assert result_2h_step2 == 0.0  # Should be 0 after exceeding 3 hours
