import numpy as np
from src.functions import q_sat_ice_func


def test_q_sat_ice_func_basic():
    """Test basic functionality of saturation vapor pressure over ice."""
    # Test case: -10°C, standard pressure
    TC = -10.0
    P = 101325.0  # Pa

    esat, qsat, d_qsat_dT = q_sat_ice_func(TC, P)

    # Check that returned values are reasonable
    assert esat > 0  # Saturation vapor pressure should be positive
    assert qsat > 0  # Specific humidity should be positive
    assert qsat < 0.1  # Should be less than 0.1 kg/kg
    assert d_qsat_dT > 0  # Temperature derivative should be positive


def test_q_sat_ice_func_comparison_with_water():
    """Test that ice saturation is lower than water saturation."""
    from src.functions import q_sat_func

    TC = -5.0
    P = 101325.0

    # Ice saturation
    esat_ice, qsat_ice, _ = q_sat_ice_func(TC, P)

    # Water saturation
    esat_water, qsat_water, _ = q_sat_func(TC, P)

    # Ice saturation should be lower than water saturation
    assert esat_ice < esat_water
    assert qsat_ice < qsat_water


def test_q_sat_ice_func_temperature_dependence():
    """Test temperature dependence."""
    P = 101325.0

    # Test at different temperatures
    TC1 = -20.0
    TC2 = -5.0

    esat1, qsat1, _ = q_sat_ice_func(TC1, P)
    esat2, qsat2, _ = q_sat_ice_func(TC2, P)

    # Higher temperature should give higher saturation values
    assert esat2 > esat1
    assert qsat2 > qsat1
