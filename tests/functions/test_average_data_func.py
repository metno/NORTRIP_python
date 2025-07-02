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
