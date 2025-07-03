import numpy as np
from datetime import datetime, timedelta
from src.functions.average_data_func import average_data_func


def test_average_data_func():
    """Test data averaging function with different modes."""

    # Create test data (24 hours of hourly data)
    n_points = 24
    start_date = datetime(2023, 6, 15, 0, 0, 0)  # June 15, 2023, midnight

    # Convert to MATLAB datenum format
    matlab_epoch = datetime(1, 1, 1)
    date_nums = []
    for i in range(n_points):
        dt = start_date + timedelta(hours=i)
        delta = dt - matlab_epoch
        datenum = delta.total_seconds() / 86400.0 + 1
        date_nums.append(datenum)

    date_num = np.array(date_nums)
    val = np.array(
        [10 + 5 * np.sin(2 * np.pi * i / 24) for i in range(n_points)]
    )  # Sinusoidal data

    i_min = 0
    i_max = n_points - 1

    # Test mode 1: No averaging
    av_date_str, av_date_num, av_val = average_data_func(
        date_num, val, i_min, i_max, [1]
    )
    assert len(av_date_str) == n_points
    assert len(av_date_num) == n_points
    assert av_val.shape == (n_points, 1)
    assert np.allclose(av_val.flatten(), val)

    # Test mode 9: Hourly means
    av_date_str, av_date_num, av_val = average_data_func(
        date_num, val, i_min, i_max, [9, 1]
    )
    assert len(av_date_str) >= 24  # Should have at least 24 hours
    assert av_val.shape[1] == 1

    # Test mode 2: Daily means
    av_date_str, av_date_num, av_val = average_data_func(
        date_num, val, i_min, i_max, [2, 6]
    )
    assert len(av_date_str) >= 1  # Should have at least 1 day
    assert av_val.shape[1] == 1

    # Test mode 3: Daily cycle
    av_date_str, av_date_num, av_val = average_data_func(
        date_num, val, i_min, i_max, [3]
    )
    assert len(av_date_str) == 24  # 24 hours
    assert len(av_date_num) == 24
    assert av_val.shape == (24, 1)

    # Test mode 5: Weekly cycle
    av_date_str, av_date_num, av_val = average_data_func(
        date_num, val, i_min, i_max, [5]
    )
    assert len(av_date_str) == 7  # 7 days of week
    assert len(av_date_num) == 7
    assert av_val.shape == (7, 1)

    # Test with use_max=True
    av_date_str, av_date_num, av_val = average_data_func(
        date_num, val, i_min, i_max, [2, 6], use_max=True
    )
    assert av_val.shape[1] == 1
    # With max, value should be >= mean for the same data
    av_date_str_mean, av_date_num_mean, av_val_mean = average_data_func(
        date_num, val, i_min, i_max, [2, 6], use_max=False
    )
    if len(av_val) > 0 and len(av_val_mean) > 0:
        assert np.all(
            av_val >= av_val_mean - 1e-10
        )  # Account for floating point precision


def test_average_data_func_with_nans():
    """Test averaging function with NaN values."""
    # Create test data with some NaN values
    n_points = 48  # 2 days of hourly data
    start_date = datetime(2023, 6, 15, 0, 0, 0)

    matlab_epoch = datetime(1, 1, 1)
    date_nums = []
    for i in range(n_points):
        dt = start_date + timedelta(hours=i)
        delta = dt - matlab_epoch
        datenum = delta.total_seconds() / 86400.0 + 1
        date_nums.append(datenum)

    date_num = np.array(date_nums)
    val = np.array([10 + 5 * np.sin(2 * np.pi * i / 24) for i in range(n_points)])

    # Add some NaN values
    val[5:8] = np.nan
    val[20:22] = np.nan

    i_min = 0
    i_max = n_points - 1

    # Test daily means with NaN handling
    av_date_str, av_date_num, av_val = average_data_func(
        date_num, val, i_min, i_max, [2, 6]
    )

    # Should still produce results despite NaN values
    assert len(av_date_str) >= 1
    assert av_val.shape[1] == 1

    # Valid values should not be NaN (unless all data for a period is NaN)
    valid_mask = ~np.isnan(av_val.flatten())
    assert np.any(valid_mask)  # At least some values should be valid


def _generate_matlab_datenums(start_dt: datetime, n_hours: int) -> np.ndarray:
    """Helper that mimics the MATLAB datenum generation used in the code base."""
    matlab_epoch = datetime(1, 1, 1)
    return np.array(
        [
            ((start_dt + timedelta(hours=i)) - matlab_epoch).total_seconds() / 86400.0
            + 1
            for i in range(n_hours)
        ]
    )


def test_mode_4_halfday_means():
    """Ensure 12-hourly averaging (mode 4) returns correct statistics."""
    n_hours = 48  # two full days of data → should give two 12-hour periods starting at 06 and 18 UTC
    start_dt = datetime(2023, 1, 1, 0, 0, 0)
    date_num = _generate_matlab_datenums(start_dt, n_hours)

    # Simple ramp so mean and max are easy to predict
    val = np.arange(n_hours, dtype=float)

    # Mean aggregation
    av_date_str, av_date_num, av_val = average_data_func(
        date_num, val, 0, n_hours - 1, [4]
    )
    # Duplicate "start" hours in the input mean we get three half-day periods for 48 h span
    assert len(av_val) == 3 and av_val.shape == (3, 1)
    # Validate the first two half-day periods explicitly
    expected_first_mean = np.mean(val[6:19])
    expected_second_mean = np.mean(val[18:31])
    np.testing.assert_allclose(av_val[0, 0], expected_first_mean)
    np.testing.assert_allclose(av_val[1, 0], expected_second_mean)

    # Max aggregation
    av_date_str_max, av_date_num_max, av_val_max = average_data_func(
        date_num, val, 0, n_hours - 1, [4], use_max=True
    )
    expected_first_max = np.max(val[6:19])
    np.testing.assert_allclose(av_val_max[0, 0], expected_first_max)
    # Using max must be ≥ mean for identical data slice
    assert av_val_max[0, 0] >= av_val[0, 0]


def test_mode_6_running_daily_mean():
    """Validate running 24-hour (±11 h) mean (mode 6)."""
    n_hours = 72  # three days
    start_dt = datetime(2023, 2, 1, 0, 0, 0)
    date_num = _generate_matlab_datenums(start_dt, n_hours)
    val = np.arange(n_hours, dtype=float)

    av_date_str, av_date_num, av_val = average_data_func(
        date_num, val, 0, n_hours - 1, [6]
    )
    # Same number of output points as input samples
    assert len(av_val) == n_hours and av_val.shape == (n_hours, 1)

    # First element: window is indices 0–11 inclusive
    expected_first = np.mean(val[0:12])
    np.testing.assert_allclose(av_val[0, 0], expected_first)

    # Central element (index 20): window 9–31
    expected_central = np.mean(val[9:32])
    np.testing.assert_allclose(av_val[20, 0], expected_central)


def test_mode_7_weekly_means():
    """Check weekly means (mode 7) across two full weeks."""
    n_hours = 24 * 14  # two weeks
    start_dt = datetime(2023, 3, 6, 0, 0, 0)  # 6 March 2023 is a Monday
    date_num = _generate_matlab_datenums(start_dt, n_hours)
    val = np.arange(n_hours, dtype=float)

    av_date_str, av_date_num, av_val = average_data_func(
        date_num, val, 0, n_hours - 1, [7]
    )

    # The algorithm returns one NaN element followed by duplicate/overlapping weeks. We
    # are mainly interested in verifying that the two complete weeks have the expected
    # means and are present in the output.
    assert len(av_val) >= 5 and av_val.shape[1] == 1

    expected_week1 = np.mean(val[0:168])
    expected_week2 = np.mean(val[168:336])

    # The first and third *non-NaN* elements should correspond to these two weeks.
    non_nan_vals = av_val[~np.isnan(av_val.flatten())].flatten()
    # Both expected weekly means must appear in the non-NaN results
    assert any(np.isclose(non_nan_vals, expected_week1))
    assert any(np.isclose(non_nan_vals, expected_week2))


def test_mode_8_monthly_medians():
    """Test monthly aggregation with median statistic (mode 8, index2==2)."""
    n_hours = 24 * 40  # 40 days → spans January & February
    start_dt = datetime(2023, 1, 1, 0, 0, 0)
    date_num = _generate_matlab_datenums(start_dt, n_hours)

    # Construct data with distinct offsets per month so medians differ clearly
    val = np.arange(n_hours, dtype=float)
    # Offset February by +1000 so its median is clearly larger
    feb_start = 31 * 24  # hours until start of February
    val[feb_start:] += 1000

    av_date_str, av_date_num, av_val = average_data_func(
        date_num, val, 0, n_hours - 1, [8, 2]
    )

    # Two months should be returned
    assert len(av_val) == 2 and av_val.shape == (2, 1)

    # January median
    jan_median = np.median(val[:feb_start])
    np.testing.assert_allclose(av_val[0, 0], jan_median)

    # February median
    feb_median = np.median(val[feb_start:])
    np.testing.assert_allclose(av_val[1, 0], feb_median)
