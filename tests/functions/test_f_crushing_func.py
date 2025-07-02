from src.functions.f_crushing_func import f_crushing_func


def test_f_crushing_func():
    """Test crushing function."""

    # Test basic case
    f_crushing_0 = 0.3
    V_veh = 50.0
    s_road = 0.0  # No snow/water
    V_ref = 60.0
    s_roadwear_thresh = 0.1

    result = f_crushing_func(f_crushing_0, V_veh, s_road, V_ref, s_roadwear_thresh)

    # Expected: f_crushing_0 * (V_veh/V_ref) * 1.0
    expected = f_crushing_0 * (V_veh / V_ref) * 1.0
    assert abs(result - expected) < 1e-10

    # Test with snow on road (should reduce to zero)
    s_road_snow = 0.2  # Above threshold
    result_snow = f_crushing_func(
        f_crushing_0, V_veh, s_road_snow, V_ref, s_roadwear_thresh
    )

    assert result_snow == 0.0

    # Test with zero reference velocity
    result_zero_ref = f_crushing_func(
        f_crushing_0, V_veh, s_road, 0.0, s_roadwear_thresh
    )

    # Should use f_V = 1.0 when V_ref = 0
    expected_zero_ref = f_crushing_0 * 1.0 * 1.0
    assert abs(result_zero_ref - expected_zero_ref) < 1e-10

    # Test velocity scaling
    V_veh_double = V_veh * 2
    result_double = f_crushing_func(
        f_crushing_0, V_veh_double, s_road, V_ref, s_roadwear_thresh
    )
    result_normal = f_crushing_func(
        f_crushing_0, V_veh, s_road, V_ref, s_roadwear_thresh
    )

    # Double velocity should give double result
    assert abs(result_double - 2 * result_normal) < 1e-10


def test_f_crushing_func_edge_cases():
    """Test edge cases."""

    f_crushing_0 = 0.5
    V_veh = 40.0
    V_ref = 50.0
    s_roadwear_thresh = 0.05

    # Zero crushing factor
    result = f_crushing_func(0.0, V_veh, 0.0, V_ref, s_roadwear_thresh)
    assert result == 0.0

    # Zero vehicle speed
    result = f_crushing_func(f_crushing_0, 0.0, 0.0, V_ref, s_roadwear_thresh)
    assert result == 0.0

    # s_road exactly at threshold
    result = f_crushing_func(
        f_crushing_0, V_veh, s_roadwear_thresh, V_ref, s_roadwear_thresh
    )
    assert result > 0  # Should still allow crushing
