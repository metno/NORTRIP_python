from src.functions.r_0_wind_func import r_0_wind_func


def test_r_0_wind_func():
    """Test wind blown dust wind speed dependency function."""

    # Test basic case with wind speed above threshold
    FF = 12.0  # Wind speed
    tau_wind = 3600.0  # Time scale in seconds (1 hour)
    FF_thresh = 8.0  # Threshold wind speed

    result = r_0_wind_func(FF, tau_wind, FF_thresh)

    # Calculate expected manually
    h_FF = (FF / FF_thresh - 1) ** 3
    expected = (1 / tau_wind) * h_FF
    assert abs(result - expected) < 1e-10

    # Result should be positive
    assert result > 0

    # Test with wind speed below threshold
    FF_low = 6.0  # Below threshold
    result_low = r_0_wind_func(FF_low, tau_wind, FF_thresh)
    assert result_low == 0.0

    # Test with wind speed equal to threshold
    result_equal = r_0_wind_func(FF_thresh, tau_wind, FF_thresh)
    assert result_equal == 0.0


def test_r_0_wind_func_threshold_behavior():
    """Test behavior around threshold wind speed."""

    tau_wind = 1800.0  # 30 minutes
    FF_thresh = 10.0

    # Test values just above and below threshold
    FF_just_below = 9.99
    FF_just_above = 10.01

    result_below = r_0_wind_func(FF_just_below, tau_wind, FF_thresh)
    result_above = r_0_wind_func(FF_just_above, tau_wind, FF_thresh)

    assert result_below == 0.0
    assert result_above > 0.0

    # Test with higher wind speeds
    wind_speeds = [12.0, 15.0, 20.0, 25.0]
    results = []

    for FF in wind_speeds:
        result = r_0_wind_func(FF, tau_wind, FF_thresh)
        results.append(result)
        assert result > 0

    # Higher wind speeds should give higher results (cubic relationship)
    for i in range(1, len(results)):
        assert results[i] > results[i - 1]


def test_r_0_wind_func_time_scale_effects():
    """Test effects of different time scales."""

    FF = 15.0
    FF_thresh = 8.0

    # Test different time scales
    tau_short = 600.0  # 10 minutes
    tau_long = 7200.0  # 2 hours

    result_short = r_0_wind_func(FF, tau_short, FF_thresh)
    result_long = r_0_wind_func(FF, tau_long, FF_thresh)

    # Shorter time scale should give higher result (inverse relationship)
    assert result_short > result_long

    # Both should be positive
    assert result_short > 0
    assert result_long > 0


def test_r_0_wind_func_cubic_relationship():
    """Test the cubic relationship with wind speed excess."""

    tau_wind = 3600.0
    FF_thresh = 10.0

    # Test specific wind speeds to verify cubic relationship
    FF_1 = 20.0  # 2x threshold
    FF_2 = 30.0  # 3x threshold

    result_1 = r_0_wind_func(FF_1, tau_wind, FF_thresh)
    result_2 = r_0_wind_func(FF_2, tau_wind, FF_thresh)

    # Calculate expected ratios
    # For FF_1: (20/10 - 1)^3 = 1^3 = 1
    # For FF_2: (30/10 - 1)^3 = 2^3 = 8
    # So result_2 should be 8 times result_1

    ratio = result_2 / result_1
    expected_ratio = 8.0
    assert abs(ratio - expected_ratio) < 1e-10


def test_r_0_wind_func_zero_cases():
    """Test cases that should return zero."""

    tau_wind = 3600.0
    FF_thresh = 12.0

    # Wind speed below threshold
    assert r_0_wind_func(5.0, tau_wind, FF_thresh) == 0.0
    assert r_0_wind_func(11.9, tau_wind, FF_thresh) == 0.0
    assert r_0_wind_func(0.0, tau_wind, FF_thresh) == 0.0

    # Wind speed equal to threshold
    assert r_0_wind_func(FF_thresh, tau_wind, FF_thresh) == 0.0


def test_r_0_wind_func_extreme_values():
    """Test with extreme parameter values."""

    # Very high wind speed
    FF_high = 50.0
    tau_wind = 3600.0
    FF_thresh = 10.0

    result_high = r_0_wind_func(FF_high, tau_wind, FF_thresh)
    assert result_high > 0

    # Very short time scale
    tau_very_short = 1.0  # 1 second
    result_short_tau = r_0_wind_func(15.0, tau_very_short, FF_thresh)
    assert result_short_tau > 0

    # Very long time scale
    tau_very_long = 86400.0  # 1 day
    result_long_tau = r_0_wind_func(15.0, tau_very_long, FF_thresh)
    assert result_long_tau > 0

    # Short time scale should give much higher result than long time scale
    assert result_short_tau > result_long_tau
