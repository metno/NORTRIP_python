from src.functions.f_abrasion_func import f_abrasion_func


def test_f_abrasion_func():
    """Test sandpaper/abrasion function."""

    # Test basic case
    f_sandpaper_0 = 0.5
    h_pave = 1.2
    V_veh = 50.0
    s_road = 0.0  # No snow/water
    V_ref = 60.0
    s_roadwear_thresh = 0.1

    result = f_abrasion_func(
        f_sandpaper_0, h_pave, V_veh, s_road, V_ref, s_roadwear_thresh
    )

    # Expected: f_sandpaper_0 * h_pave * (V_veh/V_ref) * 1.0
    expected = f_sandpaper_0 * h_pave * (V_veh / V_ref) * 1.0
    assert abs(result - expected) < 1e-10

    # Test with snow on road (should reduce to zero)
    s_road_snow = 0.2  # Above threshold
    result_snow = f_abrasion_func(
        f_sandpaper_0, h_pave, V_veh, s_road_snow, V_ref, s_roadwear_thresh
    )

    assert result_snow == 0.0

    # Test with different vehicle speed
    V_veh_low = 30.0
    result_low_speed = f_abrasion_func(
        f_sandpaper_0, h_pave, V_veh_low, s_road, V_ref, s_roadwear_thresh
    )
    result_high_speed = f_abrasion_func(
        f_sandpaper_0, h_pave, V_veh, s_road, V_ref, s_roadwear_thresh
    )

    # Higher speed should give higher abrasion
    assert result_high_speed > result_low_speed

    # Test edge case: s_road exactly at threshold
    s_road_thresh = s_roadwear_thresh
    result_thresh = f_abrasion_func(
        f_sandpaper_0, h_pave, V_veh, s_road_thresh, V_ref, s_roadwear_thresh
    )

    assert result_thresh > 0  # Should still allow wear


def test_f_abrasion_func_zero_cases():
    """Test cases that should return zero."""

    f_sandpaper_0 = 0.5
    h_pave = 1.0
    V_veh = 40.0
    V_ref = 50.0
    s_roadwear_thresh = 0.05

    # Case 1: Zero base factor
    result = f_abrasion_func(0.0, h_pave, V_veh, 0.0, V_ref, s_roadwear_thresh)
    assert result == 0.0

    # Case 2: Zero pavement factor
    result = f_abrasion_func(f_sandpaper_0, 0.0, V_veh, 0.0, V_ref, s_roadwear_thresh)
    assert result == 0.0

    # Case 3: Zero vehicle speed
    result = f_abrasion_func(f_sandpaper_0, h_pave, 0.0, 0.0, V_ref, s_roadwear_thresh)
    assert result == 0.0
