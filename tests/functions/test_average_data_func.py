import numpy as np
from datetime import datetime, timedelta
from src.functions.average_data_func import average_data_func


def _generate_matlab_datenums(start_dt: datetime, n_hours: int) -> np.ndarray:
    """Helper that mimics the MATLAB datenum generation used in the code base."""
    datetime_objects = [start_dt + timedelta(hours=i) for i in range(n_hours)]
    return np.array(
        [
            dt.toordinal() + 366 + dt.hour / 24.0 + dt.minute / (24.0 * 60.0)
            for dt in datetime_objects
        ]
    )


def test_average_data_func_mode_1_no_averaging():
    """Test mode 1: No averaging - returns original data."""
    # Create test data: 3 days of hourly data
    start_dt = datetime(2023, 1, 1, 0, 0)  # Sunday
    n_hours = 72  # 3 days
    date_num = _generate_matlab_datenums(start_dt, n_hours)

    # Create test values with some variation
    val = np.array([10 + i % 24 for i in range(n_hours)], dtype=float)
    # Add some NaN values
    val[5] = np.nan
    val[25] = np.nan
    val[50] = np.nan

    i_min = 0
    i_max = n_hours - 1
    index_in = [1]

    av_date_str, av_date_num, av_val = average_data_func(
        date_num, val, i_min, i_max, index_in
    )

    # Should return original data
    assert len(av_date_str) == n_hours
    assert len(av_date_num) == n_hours
    assert av_val.shape == (n_hours, 1)

    # Check that values match (accounting for NaN)
    np.testing.assert_array_equal(av_val.flatten(), val)
    np.testing.assert_array_equal(av_date_num, date_num)


def test_average_data_func_mode_2_daily_means():
    """Test mode 2: Daily means."""
    # Create test data: 3 days of hourly data
    start_dt = datetime(2023, 1, 1, 0, 0)  # Sunday
    n_hours = 72  # 3 days
    date_num = _generate_matlab_datenums(start_dt, n_hours)

    # Create test values with some variation
    val = np.array([10 + i % 24 for i in range(n_hours)], dtype=float)
    # Add some NaN values
    val[5] = np.nan
    val[25] = np.nan
    val[50] = np.nan

    i_min = 0
    i_max = n_hours - 1
    index_in = [2, 6]  # Daily means with minimum 6 valid values

    av_date_str, av_date_num, av_val = average_data_func(
        date_num, val, i_min, i_max, index_in
    )

    # Should have 3 daily means (3 days of data)
    assert len(av_date_str) == 3
    assert len(av_date_num) == 3
    assert av_val.shape == (3, 1)

    # The function uses strict bounds (r > i_min & r < i_max), so excludes first and last indices
    # Day 1: indices 1-22 (excluding 0 and 23), values 11-33, excluding NaN at index 5
    # Expected: mean of [11,12,13,14,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33]
    # fmt: off
    day1_values = [11,12,13,14,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33]
    # fmt: on
    expected_daily_avg = np.mean(day1_values)
    np.testing.assert_almost_equal(av_val[0, 0], expected_daily_avg, decimal=1)


def test_average_data_func_mode_3_daily_cycles():
    """Test mode 3: Daily cycles - hourly averages across all days."""
    # Create test data: 3 days of hourly data
    start_dt = datetime(2023, 1, 1, 0, 0)  # Sunday
    n_hours = 72  # 3 days
    date_num = _generate_matlab_datenums(start_dt, n_hours)

    # Create test values with some variation
    val = np.array([10 + i % 24 for i in range(n_hours)], dtype=float)
    # Add some NaN values
    val[5] = np.nan
    val[25] = np.nan
    val[50] = np.nan

    i_min = 0
    i_max = n_hours - 1
    index_in = [3]

    av_date_str, av_date_num, av_val = average_data_func(
        date_num, val, i_min, i_max, index_in
    )

    # Should have 24 hours
    assert len(av_date_str) == 24
    assert len(av_date_num) == 24
    assert av_val.shape == (24, 1)

    # Check hourly labels
    expected_labels = [f"{i:02d}" for i in range(24)]
    assert av_date_str == expected_labels

    # Check that hour 0 average is correct
    # Hour 0 occurs at indices 0, 24, 48 with values 10, 10, 10 (same pattern repeats)
    expected_hour0_avg = 10.0
    np.testing.assert_almost_equal(av_val[0, 0], expected_hour0_avg, decimal=1)

    # Check that hour 5 average is correct
    # Hour 5 occurs at indices 5, 29, 53 with values NaN, 15, 15
    # So we average 15 and 15, excluding the NaN
    expected_hour5_avg = 15.0
    np.testing.assert_almost_equal(av_val[5, 0], expected_hour5_avg, decimal=1)


def test_average_data_func_mode_4_12_hourly_means():
    """Test mode 4: 12 hourly means starting at 6 and 18."""
    # Create test data: 3 days of hourly data
    start_dt = datetime(2023, 1, 1, 0, 0)  # Sunday
    n_hours = 72  # 3 days
    date_num = _generate_matlab_datenums(start_dt, n_hours)

    # Create test values with some variation
    val = np.array([10 + i % 24 for i in range(n_hours)], dtype=float)
    # Add some NaN values
    val[5] = np.nan
    val[25] = np.nan
    val[50] = np.nan

    i_min = 0
    i_max = n_hours - 1
    index_in = [4, 6, 18]  # 12-hour periods starting at 6 and 18

    av_date_str, av_date_num, av_val = average_data_func(
        date_num, val, i_min, i_max, index_in
    )

    # Should have some 12-hour periods
    assert len(av_date_str) > 0
    assert len(av_date_num) > 0
    assert av_val.shape[0] == len(av_date_str)

    # Check that we get reasonable number of periods for 3 days
    assert len(av_date_str) <= 6  # Max 6 periods for 3 days


def test_average_data_func_mode_5_weekly_cycles():
    """Test mode 5: Weekly cycles - weekday averages."""
    # Create test data: 3 days of hourly data
    start_dt = datetime(2023, 1, 1, 0, 0)  # Sunday
    n_hours = 72  # 3 days
    date_num = _generate_matlab_datenums(start_dt, n_hours)

    # Create test values with some variation
    val = np.array([10 + i % 24 for i in range(n_hours)], dtype=float)
    # Add some NaN values
    val[5] = np.nan
    val[25] = np.nan
    val[50] = np.nan

    i_min = 0
    i_max = n_hours - 1
    index_in = [5]

    av_date_str, av_date_num, av_val = average_data_func(
        date_num, val, i_min, i_max, index_in
    )

    # Should have 7 weekdays
    assert len(av_date_str) == 7
    assert len(av_date_num) == 7
    assert av_val.shape == (7, 1)

    # Check weekday names
    expected_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    assert av_date_str == expected_names

    # Check that Sunday (index 6) has data (starts on Sunday)
    assert not np.isnan(av_val[6, 0])


def test_average_data_func_mode_6_daily_running_means():
    """Test mode 6: Daily running means with 23-hour window."""
    # Create test data: 3 days of hourly data
    start_dt = datetime(2023, 1, 1, 0, 0)  # Sunday
    n_hours = 72  # 3 days
    date_num = _generate_matlab_datenums(start_dt, n_hours)

    # Create test values with some variation
    val = np.array([10 + i % 24 for i in range(n_hours)], dtype=float)
    # Add some NaN values
    val[5] = np.nan
    val[25] = np.nan
    val[50] = np.nan

    i_min = 0
    i_max = n_hours - 1
    index_in = [6]

    av_date_str, av_date_num, av_val = average_data_func(
        date_num, val, i_min, i_max, index_in
    )

    # Should return same length as input
    assert len(av_date_str) == n_hours
    assert len(av_date_num) == n_hours
    assert av_val.shape == (n_hours, 1)

    # Check that dates match input
    np.testing.assert_array_equal(av_date_num, date_num)

    # Check that we get running averages (not original values)
    assert not np.array_equal(av_val.flatten(), val)


def test_average_data_func_mode_7_weekly_means():
    """Test mode 7: Weekly means starting on Monday."""
    # Create test data: 3 days of hourly data
    start_dt = datetime(2023, 1, 1, 0, 0)  # Sunday
    n_hours = 72  # 3 days
    date_num = _generate_matlab_datenums(start_dt, n_hours)

    # Create test values with some variation
    val = np.array([10 + i % 24 for i in range(n_hours)], dtype=float)
    # Add some NaN values
    val[5] = np.nan
    val[25] = np.nan
    val[50] = np.nan

    i_min = 0
    i_max = n_hours - 1
    index_in = [7]

    av_date_str, av_date_num, av_val = average_data_func(
        date_num, val, i_min, i_max, index_in
    )

    # Should have at least one weekly mean
    assert len(av_date_str) >= 1
    assert len(av_date_num) >= 1
    assert av_val.shape[0] == len(av_date_str)

    # For 3 days starting Sunday, should have partial week
    assert len(av_date_str) <= 2


def test_average_data_func_mode_8_monthly_means():
    """Test mode 8: Monthly means."""
    # Create test data: 3 days of hourly data
    start_dt = datetime(2023, 1, 1, 0, 0)  # Sunday
    n_hours = 72  # 3 days
    date_num = _generate_matlab_datenums(start_dt, n_hours)

    # Create test values with some variation
    val = np.array([10 + i % 24 for i in range(n_hours)], dtype=float)
    # Add some NaN values
    val[5] = np.nan
    val[25] = np.nan
    val[50] = np.nan

    i_min = 0
    i_max = n_hours - 1
    index_in = [8, 1]  # Monthly means with mean (not median)

    av_date_str, av_date_num, av_val = average_data_func(
        date_num, val, i_min, i_max, index_in
    )

    # Should have one monthly mean (all data in January 2023)
    assert len(av_date_str) == 1
    assert len(av_date_num) == 1
    assert av_val.shape == (1, 1)

    # Check date format
    assert "Jan 2023" in av_date_str[0]

    # Monthly means requires len(valid) > n_av // 4 where n_av = 24 * 30 = 720
    # So we need > 180 valid values, but we only have 72 hours total
    # Therefore the result should be NaN
    assert np.isnan(av_val[0, 0])


def test_average_data_func_mode_9_hourly_means():
    """Test mode 9: Hourly means."""
    # Create test data: 3 days of hourly data
    start_dt = datetime(2023, 1, 1, 0, 0)  # Sunday
    n_hours = 72  # 3 days
    date_num = _generate_matlab_datenums(start_dt, n_hours)

    # Create test values with some variation
    val = np.array([10 + i % 24 for i in range(n_hours)], dtype=float)
    # Add some NaN values
    val[5] = np.nan
    val[25] = np.nan
    val[50] = np.nan

    i_min = 0
    i_max = n_hours - 1
    index_in = [9, 1]  # Hourly means with minimum 1 valid value

    av_date_str, av_date_num, av_val = average_data_func(
        date_num, val, i_min, i_max, index_in
    )

    # Should have hourly data
    assert len(av_date_str) > 0
    assert len(av_date_num) > 0
    assert av_val.shape[0] == len(av_date_str)

    # Should have reasonable number of hours for 3 days
    assert len(av_date_str) >= 72  # At least original hours
    assert len(av_date_str) <= 75  # At most a few extra hours


def test_average_data_func_use_max_parameter():
    """Test that use_max parameter works correctly."""
    # Create test data: 3 days of hourly data
    start_dt = datetime(2023, 1, 1, 0, 0)  # Sunday
    n_hours = 72  # 3 days
    date_num = _generate_matlab_datenums(start_dt, n_hours)

    # Create test values with some variation
    val = np.array([10 + i % 24 for i in range(n_hours)], dtype=float)
    # Add some NaN values
    val[5] = np.nan
    val[25] = np.nan
    val[50] = np.nan

    i_min = 0
    i_max = n_hours - 1
    index_in = [2, 6]  # Daily means

    # Test with mean (default)
    _, _, av_val_mean = average_data_func(date_num, val, i_min, i_max, index_in)

    # Test with max
    _, _, av_val_max = average_data_func(
        date_num, val, i_min, i_max, index_in, use_max=True
    )

    # Max values should be greater than or equal to mean values
    assert np.all(av_val_max >= av_val_mean)


def test_average_data_func_nan_handling():
    """Test that NaN values are handled correctly."""
    # Create test data: 3 days of hourly data
    start_dt = datetime(2023, 1, 1, 0, 0)  # Sunday
    n_hours = 72  # 3 days
    date_num = _generate_matlab_datenums(start_dt, n_hours)

    # Create test values with some variation
    val = np.array([10 + i % 24 for i in range(n_hours)], dtype=float)
    # Add more NaNs
    val[10:20] = np.nan

    i_min = 0
    i_max = n_hours - 1
    index_in = [2, 6]  # Daily means

    av_date_str, av_date_num, av_val = average_data_func(
        date_num, val, i_min, i_max, index_in
    )

    # Should still return results
    assert len(av_date_str) > 0
    assert len(av_date_num) > 0
    assert av_val.shape[0] == len(av_date_str)

    # Some values might be NaN due to insufficient valid data
    # This is expected behavior
