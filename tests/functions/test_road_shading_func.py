from functions import road_shading_func


def test_road_shading_func():
    """Test the road shading function with various scenarios."""
    # Test case 1: Basic functionality
    azimuth = 180.0
    zenith = 45.0
    ang_road = 0.0
    b_road = 10.0
    b_canyon = 20.0
    h_canyon = [5.0, 5.0]

    shadow_fraction = road_shading_func(
        azimuth, zenith, ang_road, b_road, b_canyon, h_canyon
    )

    assert isinstance(shadow_fraction, float)
    assert 0.0 <= shadow_fraction <= 1.0

    # Test case 2: No shadow when angle difference is 0
    shadow_no_angle = road_shading_func(90.0, 45.0, 90.0, b_road, b_canyon, h_canyon)
    assert shadow_no_angle == 0.0

    # Test case 3: Full shadow when zenith >= 90 (sun below horizon)
    shadow_low_sun = road_shading_func(
        azimuth, 90.0, ang_road, b_road, b_canyon, h_canyon
    )
    assert shadow_low_sun == 1.0

    shadow_below_horizon = road_shading_func(
        azimuth, 95.0, ang_road, b_road, b_canyon, h_canyon
    )
    assert shadow_below_horizon == 1.0

    # Test case 4: Road angle normalization (>180 degrees)
    shadow_normalized = road_shading_func(
        azimuth, zenith, 270.0, b_road, b_canyon, h_canyon
    )
    shadow_expected = road_shading_func(
        azimuth, zenith, 90.0, b_road, b_canyon, h_canyon
    )
    assert abs(shadow_normalized - shadow_expected) < 1e-10

    # Test case 5: Different canyon wall selection
    h_canyon_asymmetric = [8.0, 4.0]

    # This should use north wall (h_canyon[0] = 8.0)
    shadow_north = road_shading_func(
        45.0, 60.0, 90.0, b_road, b_canyon, h_canyon_asymmetric
    )

    # This should use south wall (h_canyon[1] = 4.0)
    shadow_south = road_shading_func(
        135.0, 60.0, 90.0, b_road, b_canyon, h_canyon_asymmetric
    )

    # Shadow should be different due to different wall heights
    assert shadow_north != shadow_south
