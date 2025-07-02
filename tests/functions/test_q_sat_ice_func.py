import numpy as np
from src.functions.q_sat_ice_func import q_sat_ice_func


def test_q_sat_ice_func():
    """Test saturation vapor pressure over ice calculations."""

    # Test at freezing point
    TC = 0.0
    P = 101325.0  # Pa

    esat, qsat, d_qsat_dT = q_sat_ice_func(TC, P)

    # All values should be positive
    assert esat > 0
    assert qsat > 0
    assert d_qsat_dT > 0

    # At 0°C, saturation vapor pressure over ice should be around 611 Pa
    assert 600 < esat < 650

    # Specific humidity should be reasonable
    assert 0.001 < qsat < 0.01

    # Test at sub-freezing temperatures
    TC_cold = -10.0
    esat_cold, qsat_cold, d_qsat_dT_cold = q_sat_ice_func(TC_cold, P)

    assert esat_cold > 0
    assert qsat_cold > 0
    assert d_qsat_dT_cold > 0

    # Colder temperature should have lower saturation vapor pressure
    assert esat_cold < esat
    assert qsat_cold < qsat


def test_q_sat_ice_func_pressure_effects():
    """Test pressure effects on saturation over ice."""

    TC = -5.0

    # Standard pressure
    P_std = 101325.0
    esat_std, qsat_std, d_qsat_dT_std = q_sat_ice_func(TC, P_std)

    # High pressure
    P_high = 110000.0
    esat_high, qsat_high, d_qsat_dT_high = q_sat_ice_func(TC, P_high)

    # Low pressure (high altitude)
    P_low = 70000.0
    esat_low, qsat_low, d_qsat_dT_low = q_sat_ice_func(TC, P_low)

    # Saturation vapor pressure over ice should be independent of air pressure
    assert abs(esat_std - esat_high) < 1.0
    assert abs(esat_std - esat_low) < 1.0

    # Specific humidity should depend on pressure
    assert qsat_low > qsat_std > qsat_high

    # All should be positive
    assert all(
        val > 0
        for val in [esat_std, qsat_std, esat_high, qsat_high, esat_low, qsat_low]
    )


def test_q_sat_ice_func_magnus_formula():
    """Test that the function correctly implements Magnus formula for ice."""

    TC = -15.0
    P = 101325.0

    esat, qsat, d_qsat_dT = q_sat_ice_func(TC, P)

    # Calculate expected values using Magnus formula for ice
    a = 6.1121
    b = 22.46
    c = 272.62

    # Expected saturation vapor pressure in hPa
    esat_expected_hPa = a * np.exp(b * TC / (c + TC))
    esat_expected_Pa = esat_expected_hPa * 100.0  # Convert to Pa

    assert abs(esat - esat_expected_Pa) < 1.0

    # Expected specific humidity
    qsat_expected = 0.622 * esat_expected_Pa / (P - 0.378 * esat_expected_Pa)

    assert abs(qsat - qsat_expected) < 1e-6


def test_q_sat_ice_func_temperature_dependence():
    """Test temperature dependence of saturation over ice."""

    P = 101325.0
    temperatures = [-40.0, -30.0, -20.0, -10.0, -5.0, 0.0]

    esat_values = []
    qsat_values = []

    for TC in temperatures:
        esat, qsat, d_qsat_dT = q_sat_ice_func(TC, P)
        esat_values.append(esat)
        qsat_values.append(qsat)

        # All values should be positive
        assert esat > 0
        assert qsat > 0
        assert d_qsat_dT > 0

    # Values should increase with temperature
    for i in range(1, len(temperatures)):
        assert esat_values[i] > esat_values[i - 1]
        assert qsat_values[i] > qsat_values[i - 1]


def test_q_sat_ice_func_extreme_cold():
    """Test function behavior at extremely cold temperatures."""

    P = 101325.0

    # Very cold temperature
    TC_extreme = -50.0
    esat_extreme, qsat_extreme, d_qsat_dT_extreme = q_sat_ice_func(TC_extreme, P)

    assert esat_extreme > 0
    assert qsat_extreme > 0
    assert qsat_extreme < 0.0001  # Should be very small
    assert d_qsat_dT_extreme > 0

    # Compare with less extreme cold
    TC_moderate = -20.0
    esat_moderate, qsat_moderate, d_qsat_dT_moderate = q_sat_ice_func(TC_moderate, P)

    # Extreme cold should have much smaller values
    assert esat_extreme < 0.1 * esat_moderate
    assert qsat_extreme < 0.1 * qsat_moderate
