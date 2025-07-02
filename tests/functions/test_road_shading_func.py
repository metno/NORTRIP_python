import numpy as np
from src.functions.road_shading_func import road_shading_func


def test_road_shading_func():
    """Test road shading calculation in street canyon."""

    # Test basic case with sun overhead (no shading)
    azimuth = 180.0  # South
    zenith = 0.0  # Directly overhead
    ang_road = 90.0  # East-West road
    b_road = 10.0  # 10m road width
    b_canyon = 20.0  # 20m canyon width
    h_canyon = [5.0, 5.0]  # 5m building heights on both sides

    result = road_shading_func(azimuth, zenith, ang_road, b_road, b_canyon, h_canyon)

    # With sun overhead, shadow fraction should be 0
    assert result == 0.0

    # Test case with sun at zenith = 90° (sun at horizon)
    result_horizon = road_shading_func(
        azimuth, 90.0, ang_road, b_road, b_canyon, h_canyon
    )

    # With sun at horizon, shadow fraction should be 1 (complete shadow)
    assert result_horizon == 1.0

    # Test case with sun at 45° zenith
    result_45 = road_shading_func(azimuth, 45.0, ang_road, b_road, b_canyon, h_canyon)

    # Should be between 0 and 1
    assert 0 <= result_45 <= 1


def test_road_shading_func_angle_normalization():
    """Test road angle normalization."""

    azimuth = 180.0
    zenith = 30.0
    b_road = 8.0
    b_canyon = 16.0
    h_canyon = [4.0, 4.0]

    # Test angles > 180° (should be normalized)
    ang_road_normal = 90.0
    ang_road_high = 270.0  # Should be normalized to 90°

    result_normal = road_shading_func(
        azimuth, zenith, ang_road_normal, b_road, b_canyon, h_canyon
    )
    result_high = road_shading_func(
        azimuth, zenith, ang_road_high, b_road, b_canyon, h_canyon
    )

    # Results should be the same due to normalization
    assert abs(result_normal - result_high) < 1e-10


def test_road_shading_func_sun_direction():
    """Test shading with different sun directions."""

    zenith = 60.0  # Sun at 60° from vertical
    ang_road = 0.0  # North-South road
    b_road = 12.0
    b_canyon = 24.0
    h_canyon = [6.0, 6.0]

    # Sun from east
    result_east = road_shading_func(90.0, zenith, ang_road, b_road, b_canyon, h_canyon)

    # Sun from west
    result_west = road_shading_func(270.0, zenith, ang_road, b_road, b_canyon, h_canyon)

    # Both should be positive (some shading)
    assert result_east >= 0
    assert result_west >= 0
    assert result_east <= 1
    assert result_west <= 1

    # Due to symmetry, they should be similar for symmetric canyon
    assert abs(result_east - result_west) < 0.1


def test_road_shading_func_canyon_geometry():
    """Test effects of canyon geometry on shading."""

    azimuth = 135.0  # Southeast
    zenith = 45.0
    ang_road = 45.0  # Northeast-Southwest road
    b_road = 10.0

    # Narrow canyon
    b_canyon_narrow = 15.0
    h_canyon_narrow = [8.0, 8.0]
    result_narrow = road_shading_func(
        azimuth, zenith, ang_road, b_road, b_canyon_narrow, h_canyon_narrow
    )

    # Wide canyon
    b_canyon_wide = 30.0
    h_canyon_wide = [8.0, 8.0]
    result_wide = road_shading_func(
        azimuth, zenith, ang_road, b_road, b_canyon_wide, h_canyon_wide
    )

    # Narrow canyon should have more shading
    assert result_narrow >= result_wide

    # Both should be valid fractions
    assert 0 <= result_narrow <= 1
    assert 0 <= result_wide <= 1


def test_road_shading_func_building_heights():
    """Test effects of different building heights."""

    azimuth = 180.0
    zenith = 60.0
    ang_road = 90.0
    b_road = 10.0
    b_canyon = 20.0

    # Low buildings
    h_canyon_low = [3.0, 3.0]
    result_low = road_shading_func(
        azimuth, zenith, ang_road, b_road, b_canyon, h_canyon_low
    )

    # Tall buildings
    h_canyon_tall = [12.0, 12.0]
    result_tall = road_shading_func(
        azimuth, zenith, ang_road, b_road, b_canyon, h_canyon_tall
    )

    # Taller buildings should create more shadow
    assert result_tall >= result_low

    # Both should be valid fractions
    assert 0 <= result_low <= 1
    assert 0 <= result_tall <= 1


def test_road_shading_func_asymmetric_canyon():
    """Test shading with asymmetric canyon (different building heights)."""

    azimuth = 180.0  # Sun from south
    zenith = 45.0
    ang_road = 90.0  # East-West road
    b_road = 8.0
    b_canyon = 16.0

    # Asymmetric canyon
    h_canyon_asym = [5.0, 10.0]  # North side lower, south side higher
    result_asym = road_shading_func(
        azimuth, zenith, ang_road, b_road, b_canyon, h_canyon_asym
    )

    # Should be a valid fraction
    assert 0 <= result_asym <= 1

    # Compare with symmetric canyon of same average height
    h_canyon_sym = [7.5, 7.5]
    result_sym = road_shading_func(
        azimuth, zenith, ang_road, b_road, b_canyon, h_canyon_sym
    )

    # Both should be valid
    assert 0 <= result_sym <= 1


def test_road_shading_func_extreme_cases():
    """Test extreme cases and edge conditions."""

    b_road = 10.0
    b_canyon = 20.0
    h_canyon = [6.0, 6.0]
    ang_road = 45.0

    # Test with zenith >= 90° (sun at or below horizon)
    result_90 = road_shading_func(180.0, 90.0, ang_road, b_road, b_canyon, h_canyon)
    assert result_90 == 1.0  # Complete shadow

    result_95 = road_shading_func(180.0, 95.0, ang_road, b_road, b_canyon, h_canyon)
    assert result_95 == 1.0  # Complete shadow

    # Test with zero building heights
    h_canyon_zero = [0.0, 0.0]
    result_zero = road_shading_func(
        180.0, 45.0, ang_road, b_road, b_canyon, h_canyon_zero
    )
    assert result_zero == 0.0  # No shadow from zero-height buildings

    # Test with very narrow road
    b_road_narrow = 1.0
    result_narrow_road = road_shading_func(
        180.0, 45.0, ang_road, b_road_narrow, b_canyon, h_canyon
    )
    assert 0 <= result_narrow_road <= 1

    # Test with road width equal to canyon width (no sidewalks)
    result_full_width = road_shading_func(
        180.0, 45.0, ang_road, b_canyon, b_canyon, h_canyon
    )
    assert 0 <= result_full_width <= 1


def test_road_shading_func_angle_differences():
    """Test behavior with different angle differences between sun and road."""

    zenith = 60.0
    b_road = 10.0
    b_canyon = 20.0
    h_canyon = [8.0, 8.0]
    ang_road = 0.0  # North-South road

    # Test different sun azimuths
    azimuths = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]
    results = []

    for azimuth in azimuths:
        result = road_shading_func(
            azimuth, zenith, ang_road, b_road, b_canyon, h_canyon
        )
        results.append(result)

        # All results should be valid fractions
        assert 0 <= result <= 1

    # Test that perpendicular angles (90° and 270°) give similar results for symmetric canyon
    idx_90 = azimuths.index(90.0)
    idx_270 = azimuths.index(270.0)
    assert abs(results[idx_90] - results[idx_270]) < 0.1


def test_road_shading_func_mathematical_consistency():
    """Test mathematical consistency of the shading calculation."""

    azimuth = 200.0
    zenith = 30.0
    ang_road = 75.0
    b_road = 12.0
    b_canyon = 24.0
    h_canyon = [6.0, 9.0]

    result = road_shading_func(azimuth, zenith, ang_road, b_road, b_canyon, h_canyon)

    # Result should be bounded
    assert 0 <= result <= 1

    # Test that very small zenith angle gives small shadow
    result_small_zenith = road_shading_func(
        azimuth, 5.0, ang_road, b_road, b_canyon, h_canyon
    )
    assert (
        result_small_zenith <= result
    )  # Smaller zenith should give less or equal shadow

    # Test that larger zenith gives more shadow
    result_large_zenith = road_shading_func(
        azimuth, 75.0, ang_road, b_road, b_canyon, h_canyon
    )
    assert (
        result_large_zenith >= result
    )  # Larger zenith should give more or equal shadow


def test_road_shading_func_parameter_ranges():
    """Test function behavior across realistic parameter ranges."""

    # Typical urban canyon parameters
    test_cases = [
        # azimuth, zenith, ang_road, b_road, b_canyon, h_canyon
        (180.0, 30.0, 0.0, 8.0, 16.0, [5.0, 5.0]),
        (90.0, 45.0, 90.0, 12.0, 24.0, [10.0, 8.0]),
        (270.0, 60.0, 45.0, 6.0, 20.0, [15.0, 12.0]),
        (0.0, 75.0, 135.0, 10.0, 18.0, [6.0, 9.0]),
    ]

    for azimuth, zenith, ang_road, b_road, b_canyon, h_canyon in test_cases:
        result = road_shading_func(
            azimuth, zenith, ang_road, b_road, b_canyon, h_canyon
        )

        # All results should be valid fractions
        assert 0 <= result <= 1

        # Result should be finite
        assert np.isfinite(result)
