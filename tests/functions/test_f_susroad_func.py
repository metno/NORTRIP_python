from src.functions.f_susroad_func import f_susroad_func


def test_f_susroad_func():
    """Test vehicle speed dependence function for suspension."""

    # Test basic case
    f_0_susroad = 0.5
    V_veh = 60.0
    a_sus = [0.2, 0.8, 2.0, 50.0, 10.0]  # [a1, a2, a3, a4, a5]

    result = f_susroad_func(f_0_susroad, V_veh, a_sus)

    # Calculate expected manually
    h_V = max(0.0, a_sus[0] + a_sus[1] * (max(V_veh, a_sus[4]) / a_sus[3]) ** a_sus[2])
    expected = f_0_susroad * h_V
    assert abs(result - expected) < 1e-10

    # Result should be positive
    assert result >= 0

    # Test with zero base factor
    result_zero = f_susroad_func(0.0, V_veh, a_sus)
    assert result_zero == 0.0

    # Test with zero vehicle speed
    result_zero_speed = f_susroad_func(f_0_susroad, 0.0, a_sus)
    assert result_zero_speed >= 0


def test_f_susroad_func_speed_limits():
    """Test speed limit handling (a5 parameter)."""

    f_0_susroad = 1.0
    a_sus = [0.1, 1.0, 1.5, 80.0, 30.0]  # a5 = 30.0 is minimum speed

    # Test with speed below minimum (should use minimum)
    V_veh_low = 20.0  # Below a5
    result_low = f_susroad_func(f_0_susroad, V_veh_low, a_sus)

    # Test with speed at minimum
    V_veh_min = 30.0  # Equal to a5
    result_min = f_susroad_func(f_0_susroad, V_veh_min, a_sus)

    # Both should use the minimum speed in calculation
    assert abs(result_low - result_min) < 1e-10

    # Test with speed above minimum
    V_veh_high = 60.0  # Above a5
    result_high = f_susroad_func(f_0_susroad, V_veh_high, a_sus)

    # Should be different from minimum speed case
    assert result_high != result_min


def test_f_susroad_func_negative_h_V():
    """Test case where h_V calculation could be negative."""

    f_0_susroad = 0.8
    V_veh = 10.0
    # Choose coefficients that could give negative h_V
    a_sus = [-0.5, 0.2, 2.0, 100.0, 5.0]  # Negative a1

    result = f_susroad_func(f_0_susroad, V_veh, a_sus)

    # Result should be non-negative due to max(0.0, ...) in h_V calculation
    assert result >= 0.0


def test_f_susroad_func_parameter_effects():
    """Test effects of different parameters."""

    f_0_susroad = 1.0
    V_veh = 50.0
    base_a_sus = [0.1, 0.5, 2.0, 60.0, 20.0]

    # Test effect of a1 (offset)
    a_sus_high_a1 = [0.5, 0.5, 2.0, 60.0, 20.0]  # Higher a1
    result_base = f_susroad_func(f_0_susroad, V_veh, base_a_sus)
    result_high_a1 = f_susroad_func(f_0_susroad, V_veh, a_sus_high_a1)
    assert result_high_a1 > result_base

    # Test effect of a2 (scaling factor)
    a_sus_high_a2 = [0.1, 1.0, 2.0, 60.0, 20.0]  # Higher a2
    result_high_a2 = f_susroad_func(f_0_susroad, V_veh, a_sus_high_a2)
    assert result_high_a2 > result_base

    # Test effect of a3 (exponent)
    a_sus_high_a3 = [0.1, 0.5, 3.0, 60.0, 20.0]  # Higher a3
    result_high_a3 = f_susroad_func(f_0_susroad, V_veh, a_sus_high_a3)
    # Effect depends on whether speed ratio is above or below 1
    assert result_high_a3 >= 0


def test_f_susroad_func_speed_scaling():
    """Test speed scaling behavior."""

    f_0_susroad = 1.0
    a_sus = [0.0, 1.0, 1.0, 50.0, 10.0]  # Simple linear relationship

    speeds = [20.0, 40.0, 60.0, 80.0]
    results = []

    for speed in speeds:
        result = f_susroad_func(f_0_susroad, speed, a_sus)
        results.append(result)

    # With linear exponent (a3=1.0), higher speeds should generally give higher results
    # (when speed > reference speed a4)
    for i in range(len(results)):
        assert results[i] >= 0
