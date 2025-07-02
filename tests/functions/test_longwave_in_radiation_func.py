from src.functions.longwave_in_radiation_func import longwave_in_radiation_func


def test_longwave_in_radiation_func():
    """Test incoming longwave radiation calculation."""

    # Test with typical atmospheric conditions
    TC = 15.0  # 15°C air temperature
    RH = 70.0  # 70% relative humidity
    n_c = 0.3  # 30% cloud cover
    P = 101325.0  # Standard atmospheric pressure (Pa)

    result = longwave_in_radiation_func(TC, RH, n_c, P)

    # Longwave radiation should be positive
    assert result > 0

    # Should be in reasonable range for atmospheric longwave radiation (W/m²)
    assert 200 < result < 500

    # Test with clear sky (no clouds)
    result_clear = longwave_in_radiation_func(TC, RH, 0.0, P)

    # Test with overcast sky (full cloud cover)
    result_overcast = longwave_in_radiation_func(TC, RH, 1.0, P)

    # Overcast should have higher longwave radiation than clear sky
    assert result_overcast > result_clear
    assert result_clear > 0

    # Test with different temperatures
    result_cold = longwave_in_radiation_func(-10.0, RH, n_c, P)
    result_warm = longwave_in_radiation_func(30.0, RH, n_c, P)

    # Warmer air should emit more longwave radiation
    assert result_warm > result_cold
    assert result_cold > 0


def test_longwave_in_radiation_func_humidity_effects():
    """Test humidity effects on longwave radiation."""

    TC = 20.0
    n_c = 0.2
    P = 101325.0

    # Low humidity
    result_dry = longwave_in_radiation_func(TC, 30.0, n_c, P)

    # High humidity
    result_humid = longwave_in_radiation_func(TC, 90.0, n_c, P)

    # Higher humidity should result in higher longwave radiation
    # due to water vapor being a greenhouse gas
    assert result_humid > result_dry
    assert result_dry > 0


def test_longwave_in_radiation_func_pressure_effects():
    """Test pressure effects on longwave radiation."""

    TC = 15.0
    RH = 60.0
    n_c = 0.4

    # Sea level pressure
    result_sealevel = longwave_in_radiation_func(TC, RH, n_c, 101325.0)

    # High altitude pressure (lower)
    result_altitude = longwave_in_radiation_func(TC, RH, n_c, 80000.0)

    # Both should be positive
    assert result_sealevel > 0
    assert result_altitude > 0

    # The difference should be related to pressure effects on vapor pressure


def test_longwave_in_radiation_func_extreme_conditions():
    """Test function with extreme conditions."""

    P = 101325.0
    n_c = 0.5

    # Very cold conditions
    result_very_cold = longwave_in_radiation_func(-40.0, 80.0, n_c, P)
    assert result_very_cold > 0
    assert result_very_cold < 350  # Should be relatively low

    # Very warm conditions
    result_very_warm = longwave_in_radiation_func(40.0, 60.0, n_c, P)
    assert result_very_warm > 0
    assert result_very_warm > 350  # Should be relatively high

    # Very dry conditions
    result_very_dry = longwave_in_radiation_func(20.0, 5.0, n_c, P)
    assert result_very_dry > 0

    # Very humid conditions
    result_very_humid = longwave_in_radiation_func(20.0, 100.0, n_c, P)
    assert result_very_humid > 0

    # Dry should be less than humid
    assert result_very_humid > result_very_dry


def test_longwave_in_radiation_func_cloud_sensitivity():
    """Test sensitivity to cloud cover changes."""

    TC = 10.0
    RH = 75.0
    P = 101325.0

    cloud_covers = [0.0, 0.25, 0.5, 0.75, 1.0]
    results = []

    for n_c in cloud_covers:
        result = longwave_in_radiation_func(TC, RH, n_c, P)
        results.append(result)
        assert result > 0

    # Longwave radiation should generally increase with cloud cover
    for i in range(1, len(results)):
        assert results[i] >= results[i - 1]


def test_longwave_in_radiation_func_physical_limits():
    """Test that results are within physical limits."""

    # Test various combinations
    test_conditions = [
        (-20.0, 40.0, 0.1, 101325.0),
        (0.0, 80.0, 0.5, 101325.0),
        (25.0, 95.0, 0.9, 101325.0),
        (35.0, 30.0, 0.0, 90000.0),
    ]

    for TC, RH, n_c, P in test_conditions:
        result = longwave_in_radiation_func(TC, RH, n_c, P)

        # Should be positive
        assert result > 0

        # Should be less than Stefan-Boltzmann emission from air temperature
        # (since emissivity < 1)
        T0C = 273.15
        sigma = 5.67e-8
        blackbody_emission = sigma * (T0C + TC) ** 4
        assert result < blackbody_emission

        # Should be reasonable atmospheric range
        assert 140 < result < 600
