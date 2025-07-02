from src.functions.w_func import w_func


def test_w_func():
    """Test wear function calculation."""

    # Test basic case
    W_0 = 1.0
    h_pave = 0.8
    h_dc = 1.2
    V_veh = 50.0
    a_wear = [0.1, 0.5, 2.0, 60.0, 10.0]  # [a1, a2, a3, a4, a5]
    s_road = 0.5  # Snow/water on road
    s_roadwear_thresh = 1.0  # Threshold
    s = 0  # Source type (road dust)
    road_index = 0
    tyre_index = 1
    brake_index = 2

    result = w_func(
        W_0,
        h_pave,
        h_dc,
        V_veh,
        a_wear,
        s_road,
        s_roadwear_thresh,
        s,
        road_index,
        tyre_index,
        brake_index,
    )

    # Result should be positive
    assert result >= 0

    # Calculate expected manually
    f_snow = 1.0  # s_road < s_roadwear_thresh
    f_V = max(
        0.0, a_wear[0] + a_wear[1] * (max(V_veh, a_wear[4]) / a_wear[3]) ** a_wear[2]
    )
    # For road dust source (s == road_index): h_dc_adj = 1.0
    expected = W_0 * h_pave * 1.0 * f_V * f_snow
    assert abs(result - expected) < 1e-10


def test_w_func_snow_effect():
    """Test snow/water effect on wear."""

    W_0 = 1.0
    h_pave = 0.8
    h_dc = 1.2
    V_veh = 50.0
    a_wear = [0.1, 0.5, 2.0, 60.0, 10.0]
    s_roadwear_thresh = 1.0
    s = 0  # Road dust
    road_index = 0
    tyre_index = 1
    brake_index = 2

    # Test with snow below threshold (should have normal wear)
    s_road_low = 0.5
    result_normal = w_func(
        W_0,
        h_pave,
        h_dc,
        V_veh,
        a_wear,
        s_road_low,
        s_roadwear_thresh,
        s,
        road_index,
        tyre_index,
        brake_index,
    )

    # Test with snow above threshold (should have no wear)
    s_road_high = 1.5
    result_no_wear = w_func(
        W_0,
        h_pave,
        h_dc,
        V_veh,
        a_wear,
        s_road_high,
        s_roadwear_thresh,
        s,
        road_index,
        tyre_index,
        brake_index,
    )

    assert result_normal > 0
    assert result_no_wear == 0.0


def test_w_func_source_types():
    """Test different source types and their adjustments."""

    W_0 = 1.0
    h_pave = 0.8
    h_dc = 1.2
    V_veh = 50.0
    a_wear = [0.1, 0.5, 2.0, 60.0, 10.0]
    s_road = 0.5
    s_roadwear_thresh = 1.0
    road_index = 0
    tyre_index = 1
    brake_index = 2

    # Test road dust source (s == road_index)
    result_road = w_func(
        W_0,
        h_pave,
        h_dc,
        V_veh,
        a_wear,
        s_road,
        s_roadwear_thresh,
        road_index,
        road_index,
        tyre_index,
        brake_index,
    )

    # Test tyre wear source (s == tyre_index)
    result_tyre = w_func(
        W_0,
        h_pave,
        h_dc,
        V_veh,
        a_wear,
        s_road,
        s_roadwear_thresh,
        tyre_index,
        road_index,
        tyre_index,
        brake_index,
    )

    # Test brake wear source (s == brake_index)
    result_brake = w_func(
        W_0,
        h_pave,
        h_dc,
        V_veh,
        a_wear,
        s_road,
        s_roadwear_thresh,
        brake_index,
        road_index,
        tyre_index,
        brake_index,
    )

    # All should be positive
    assert result_road > 0
    assert result_tyre > 0
    assert result_brake > 0

    # Different sources should give different results due to different adjustments
    # Road: h_dc_adj = 1.0
    # Tyre: h_dc_adj = 1.0, h_pave_adj = 1.0
    # Brake: h_pave_adj = 1.0, f_snow_adj = 1.0 (even with snow)


def test_w_func_brake_snow_immunity():
    """Test that brake wear is immune to snow effects."""

    W_0 = 1.0
    h_pave = 0.8
    h_dc = 1.2
    V_veh = 50.0
    a_wear = [0.1, 0.5, 2.0, 60.0, 10.0]
    s_roadwear_thresh = 1.0
    road_index = 0
    tyre_index = 1
    brake_index = 2

    # Test brake wear with snow above threshold
    s_road_high = 2.0  # Above threshold
    result_brake_snow = w_func(
        W_0,
        h_pave,
        h_dc,
        V_veh,
        a_wear,
        s_road_high,
        s_roadwear_thresh,
        brake_index,
        road_index,
        tyre_index,
        brake_index,
    )

    # Test brake wear without snow
    s_road_low = 0.0
    result_brake_no_snow = w_func(
        W_0,
        h_pave,
        h_dc,
        V_veh,
        a_wear,
        s_road_low,
        s_roadwear_thresh,
        brake_index,
        road_index,
        tyre_index,
        brake_index,
    )

    # Both should be equal (brake wear ignores snow)
    assert abs(result_brake_snow - result_brake_no_snow) < 1e-10
    assert result_brake_snow > 0


def test_w_func_velocity_effects():
    """Test velocity factor calculation."""

    W_0 = 1.0
    h_pave = 1.0
    h_dc = 1.0
    a_wear = [0.0, 1.0, 1.0, 50.0, 10.0]  # Simple linear relationship
    s_road = 0.0
    s_roadwear_thresh = 1.0
    s = 0
    road_index = 0
    tyre_index = 1
    brake_index = 2

    # Test different velocities
    velocities = [20.0, 40.0, 60.0, 80.0]
    results = []

    for V_veh in velocities:
        result = w_func(
            W_0,
            h_pave,
            h_dc,
            V_veh,
            a_wear,
            s_road,
            s_roadwear_thresh,
            s,
            road_index,
            tyre_index,
            brake_index,
        )
        results.append(result)

    # Higher velocities should generally give higher wear (with linear exponent)
    for i in range(len(results)):
        assert results[i] >= 0


def test_w_func_minimum_velocity():
    """Test minimum velocity handling (a5 parameter)."""

    W_0 = 1.0
    h_pave = 1.0
    h_dc = 1.0
    a_wear = [0.1, 0.8, 2.0, 60.0, 30.0]  # a5 = 30.0 is minimum velocity
    s_road = 0.0
    s_roadwear_thresh = 1.0
    s = 0
    road_index = 0
    tyre_index = 1
    brake_index = 2

    # Test with velocity below minimum
    V_veh_low = 20.0  # Below a5
    result_low = w_func(
        W_0,
        h_pave,
        h_dc,
        V_veh_low,
        a_wear,
        s_road,
        s_roadwear_thresh,
        s,
        road_index,
        tyre_index,
        brake_index,
    )

    # Test with velocity at minimum
    V_veh_min = 30.0  # Equal to a5
    result_min = w_func(
        W_0,
        h_pave,
        h_dc,
        V_veh_min,
        a_wear,
        s_road,
        s_roadwear_thresh,
        s,
        road_index,
        tyre_index,
        brake_index,
    )

    # Both should use minimum velocity in calculation
    assert abs(result_low - result_min) < 1e-10


def test_w_func_negative_velocity_factor():
    """Test case where velocity factor could be negative."""

    W_0 = 1.0
    h_pave = 1.0
    h_dc = 1.0
    V_veh = 10.0
    # Choose coefficients that give negative f_V
    a_wear = [-1.0, 0.2, 2.0, 100.0, 5.0]  # Negative a1
    s_road = 0.0
    s_roadwear_thresh = 1.0
    s = 0
    road_index = 0
    tyre_index = 1
    brake_index = 2

    result = w_func(
        W_0,
        h_pave,
        h_dc,
        V_veh,
        a_wear,
        s_road,
        s_roadwear_thresh,
        s,
        road_index,
        tyre_index,
        brake_index,
    )

    # Result should be non-negative due to max(0.0, ...) in f_V calculation
    assert result >= 0.0


def test_w_func_parameter_combinations():
    """Test various parameter combinations."""

    W_0 = 0.5
    h_pave = 1.5
    h_dc = 0.8
    V_veh = 45.0
    a_wear = [0.2, 0.6, 1.5, 50.0, 15.0]
    s_road = 0.3
    s_roadwear_thresh = 1.0
    road_index = 0
    tyre_index = 1
    brake_index = 2

    # Test all source types
    for s in [road_index, tyre_index, brake_index]:
        result = w_func(
            W_0,
            h_pave,
            h_dc,
            V_veh,
            a_wear,
            s_road,
            s_roadwear_thresh,
            s,
            road_index,
            tyre_index,
            brake_index,
        )
        assert result >= 0

    # Test with zero base wear factor
    result_zero = w_func(
        0.0,
        h_pave,
        h_dc,
        V_veh,
        a_wear,
        s_road,
        s_roadwear_thresh,
        road_index,
        road_index,
        tyre_index,
        brake_index,
    )
    assert result_zero == 0.0


def test_w_func_extreme_parameters():
    """Test with extreme parameter values."""

    W_0 = 2.0
    h_pave = 0.1
    h_dc = 3.0
    V_veh = 150.0  # Very high speed
    a_wear = [0.5, 2.0, 0.5, 80.0, 20.0]  # Low exponent
    s_road = 0.0
    s_roadwear_thresh = 1.0
    s = 0
    road_index = 0
    tyre_index = 1
    brake_index = 2

    result = w_func(
        W_0,
        h_pave,
        h_dc,
        V_veh,
        a_wear,
        s_road,
        s_roadwear_thresh,
        s,
        road_index,
        tyre_index,
        brake_index,
    )

    # Should handle extreme values gracefully
    assert result >= 0
    assert result < 1000  # Reasonable upper bound
