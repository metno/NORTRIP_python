import numpy as np
from src.functions.q_sat_func import q_sat_func


def test_q_sat_func():
    """Test saturation vapor pressure and specific humidity calculations."""

    # Test at standard conditions (20°C, 1 atm)
    TC = 20.0
    P = 101325.0  # Pa

    esat, qsat, d_qsat_dT = q_sat_func(TC, P)

    # Saturation vapor pressure should be positive
    assert esat > 0

    # At 20°C, saturation vapor pressure should be around 2.34 kPa = 2340 Pa
    assert 2200 < esat < 2500

    # Saturation specific humidity should be positive and reasonable
    assert qsat > 0
    assert qsat < 0.02  # Should be less than 2% (reasonable upper bound)

    # Temperature derivative should be positive (increasing with temperature)
    assert d_qsat_dT > 0

    # Test at freezing point
    TC_freeze = 0.0
    esat_freeze, qsat_freeze, d_qsat_dT_freeze = q_sat_func(TC_freeze, P)

    assert esat_freeze > 0
    assert qsat_freeze > 0
    assert d_qsat_dT_freeze > 0

    # At freezing, values should be lower than at 20°C
    assert esat_freeze < esat
    assert qsat_freeze < qsat

    # Test at negative temperature
    TC_cold = -10.0
    esat_cold, qsat_cold, d_qsat_dT_cold = q_sat_func(TC_cold, P)

    assert esat_cold > 0
    assert qsat_cold > 0
    assert d_qsat_dT_cold > 0

    # Cold conditions should have lower values
    assert esat_cold < esat_freeze < esat
    assert qsat_cold < qsat_freeze < qsat


def test_q_sat_func_pressure_effects():
    """Test pressure effects on saturation properties."""

    TC = 15.0

    # Standard pressure
    P_std = 101325.0
    esat_std, qsat_std, d_qsat_dT_std = q_sat_func(TC, P_std)

    # High pressure (sea level)
    P_high = 102000.0
    esat_high, qsat_high, d_qsat_dT_high = q_sat_func(TC, P_high)

    # Low pressure (high altitude)
    P_low = 80000.0
    esat_low, qsat_low, d_qsat_dT_low = q_sat_func(TC, P_low)

    # Saturation vapor pressure should be independent of air pressure
    assert abs(esat_std - esat_high) < 1.0
    assert abs(esat_std - esat_low) < 1.0

    # Specific humidity should depend on pressure
    assert qsat_low > qsat_std > qsat_high  # Lower pressure = higher specific humidity

    # All should be positive
    assert all(
        val > 0
        for val in [esat_std, qsat_std, esat_high, qsat_high, esat_low, qsat_low]
    )


def test_q_sat_func_temperature_dependence():
    """Test temperature dependence of saturation properties."""

    P = 101325.0
    temperatures = [-20.0, -10.0, 0.0, 10.0, 20.0, 30.0, 40.0]

    esat_values = []
    qsat_values = []
    d_qsat_dT_values = []

    for TC in temperatures:
        esat, qsat, d_qsat_dT = q_sat_func(TC, P)
        esat_values.append(esat)
        qsat_values.append(qsat)
        d_qsat_dT_values.append(d_qsat_dT)

        # All values should be positive
        assert esat > 0
        assert qsat > 0
        assert d_qsat_dT > 0

    # Values should increase with temperature
    for i in range(1, len(temperatures)):
        assert esat_values[i] > esat_values[i - 1]
        assert qsat_values[i] > qsat_values[i - 1]
        # d_qsat_dT should generally increase but may have some variation


def test_q_sat_func_magnus_formula():
    """Test that the function correctly implements Magnus formula."""

    TC = 25.0
    P = 101325.0

    esat, qsat, d_qsat_dT = q_sat_func(TC, P)

    # Calculate expected values using Magnus formula
    a = 6.1121
    b = 17.67
    c = 243.5

    # Expected saturation vapor pressure in hPa
    esat_expected_hPa = a * np.exp(b * TC / (c + TC))
    esat_expected_Pa = esat_expected_hPa * 100.0  # Convert to Pa

    assert abs(esat - esat_expected_Pa) < 1.0

    # Expected specific humidity
    P_hPa = P / 100.0
    qsat_expected = 0.622 * esat_expected_hPa / (P_hPa - 0.378 * esat_expected_hPa)

    assert abs(qsat - qsat_expected) < 1e-6


def test_q_sat_func_physical_consistency():
    """Test physical consistency of the calculations."""

    # Test Clausius-Clapeyron relation approximately
    TC1 = 20.0
    TC2 = 21.0
    P = 101325.0

    esat1, qsat1, d_qsat_dT1 = q_sat_func(TC1, P)
    esat2, qsat2, d_qsat_dT2 = q_sat_func(TC2, P)

    # Numerical derivative
    desat_dT_numerical = (esat2 - esat1) / (TC2 - TC1)
    dqsat_dT_numerical = (qsat2 - qsat1) / (TC2 - TC1)

    # The analytical derivative should be close to numerical derivative
    # (within reasonable tolerance for numerical differences)
    # Note: We compare the qsat derivatives since esat derivative is calculated differently
    relative_error = abs(d_qsat_dT1 - dqsat_dT_numerical) / d_qsat_dT1
    assert relative_error < 0.1  # Within 10%


def test_q_sat_func_extreme_conditions():
    """Test function behavior at extreme conditions."""

    P = 101325.0

    # Very cold temperature
    TC_very_cold = -40.0
    esat_cold, qsat_cold, d_qsat_dT_cold = q_sat_func(TC_very_cold, P)

    assert esat_cold > 0
    assert qsat_cold > 0
    assert qsat_cold < 0.001  # Should be very small
    assert d_qsat_dT_cold > 0

    # Very warm temperature
    TC_very_warm = 50.0
    esat_warm, qsat_warm, d_qsat_dT_warm = q_sat_func(TC_very_warm, P)

    assert esat_warm > 0
    assert qsat_warm > 0
    assert qsat_warm < 0.1  # Still should be reasonable
    assert d_qsat_dT_warm > 0

    # Warm should be much larger than cold
    assert esat_warm > 10 * esat_cold
    assert qsat_warm > 10 * qsat_cold


def test_q_sat_func_unit_consistency():
    """Test that units are consistent throughout the calculation."""

    TC = 15.0
    P_Pa = 101325.0

    esat, qsat, d_qsat_dT = q_sat_func(TC, P_Pa)

    # esat should be in Pa (Pascal)
    # Typical range at 15°C should be around 1700 Pa
    assert 1500 < esat < 2000

    # qsat should be dimensionless (kg/kg)
    # Typical range at 15°C should be around 0.01
    assert 0.005 < qsat < 0.015

    # d_qsat_dT should be in kg/kg/K
    # Should be positive and reasonable magnitude
    assert 0.0001 < d_qsat_dT < 0.001
