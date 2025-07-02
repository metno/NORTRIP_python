from src.functions.f_spray_func import f_spray_func


def test_f_spray_func():
    """Test spray factor calculation."""

    # Test basic case with water spray active
    R_0_spray = 0.1
    V_veh = 80.0
    V_ref_spray = 70.0
    V_thresh_spray = 30.0
    a_spray = 2.0
    water_spray_flag = True

    result = f_spray_func(
        R_0_spray, V_veh, V_ref_spray, V_thresh_spray, a_spray, water_spray_flag
    )

    # Expected: R_0_spray * ((V_veh - V_thresh_spray) / (V_ref_spray - V_thresh_spray))^a_spray
    expected = (
        R_0_spray
        * ((V_veh - V_thresh_spray) / (V_ref_spray - V_thresh_spray)) ** a_spray
    )
    assert abs(result - expected) < 1e-10

    # Test with water spray disabled
    result_no_spray = f_spray_func(
        R_0_spray, V_veh, V_ref_spray, V_thresh_spray, a_spray, False
    )
    assert result_no_spray == 0.0

    # Test with V_ref_spray <= V_thresh_spray (should return 0)
    result_invalid_ref = f_spray_func(
        R_0_spray, V_veh, 25.0, V_thresh_spray, a_spray, water_spray_flag
    )
    assert result_invalid_ref == 0.0

    # Test with vehicle speed below threshold
    V_veh_low = 20.0  # Below V_thresh_spray
    result_low_speed = f_spray_func(
        R_0_spray, V_veh_low, V_ref_spray, V_thresh_spray, a_spray, water_spray_flag
    )
    assert result_low_speed == 0.0

    # Test with vehicle speed equal to threshold
    result_at_thresh = f_spray_func(
        R_0_spray,
        V_thresh_spray,
        V_ref_spray,
        V_thresh_spray,
        a_spray,
        water_spray_flag,
    )
    assert result_at_thresh == 0.0


def test_f_spray_func_edge_cases():
    """Test edge cases for spray function."""

    R_0_spray = 0.05
    V_ref_spray = 60.0
    V_thresh_spray = 20.0
    a_spray = 1.5
    water_spray_flag = True

    # Test with zero base spray factor
    result = f_spray_func(
        0.0, 50.0, V_ref_spray, V_thresh_spray, a_spray, water_spray_flag
    )
    assert result == 0.0

    # Test with very high vehicle speed
    result_high = f_spray_func(
        R_0_spray, 150.0, V_ref_spray, V_thresh_spray, a_spray, water_spray_flag
    )
    assert result_high > 0

    # Test with different exponents
    result_exp_1 = f_spray_func(
        R_0_spray, 50.0, V_ref_spray, V_thresh_spray, 1.0, water_spray_flag
    )
    result_exp_2 = f_spray_func(
        R_0_spray, 50.0, V_ref_spray, V_thresh_spray, 2.0, water_spray_flag
    )

    # Higher exponent should give different result for same conditions
    assert result_exp_1 != result_exp_2


def test_f_spray_func_speed_relationship():
    """Test relationship between vehicle speed and spray factor."""

    R_0_spray = 0.2
    V_ref_spray = 80.0
    V_thresh_spray = 40.0
    a_spray = 2.0
    water_spray_flag = True

    # Test that higher speeds give higher spray factors
    speeds = [45.0, 55.0, 65.0, 75.0]
    results = []

    for speed in speeds:
        result = f_spray_func(
            R_0_spray, speed, V_ref_spray, V_thresh_spray, a_spray, water_spray_flag
        )
        results.append(result)

    # Results should increase with speed
    for i in range(1, len(results)):
        assert results[i] > results[i - 1]
