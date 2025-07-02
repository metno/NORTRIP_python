from functions import longwave_in_radiation_func
import numpy as np


def test_longwave_in_radiation_func_basic():
    """Test basic functionality with typical values."""
    TC = 20.0  # 20°C
    RH = 50.0  # 50% relative humidity
    n_c = 0.5  # 50% cloud cover
    P = 101325.0  # Standard pressure in Pa

    result = longwave_in_radiation_func(TC, RH, n_c, P)

    # Check that result is a float and within reasonable range for longwave radiation
    assert (
        200.0 <= result <= 500.0
    )  # Typical range for incoming longwave radiation W/m²


def test_longwave_in_radiation_func_clear_sky():
    """Test clear sky conditions (n_c = 0)."""
    TC = 15.0
    RH = 60.0
    n_c = 0.0  # Clear sky
    P = 101325.0

    result = longwave_in_radiation_func(TC, RH, n_c, P)

    # Under clear sky, result should be positive and reasonable
    assert result > 0

    # Test that results are reasonable for clear sky conditions
    assert 150.0 <= result <= 250.0  # Reasonable range for clear sky longwave


def test_longwave_in_radiation_func_overcast():
    """Test overcast conditions (n_c = 1)."""
    TC = 15.0
    RH = 60.0
    n_c = 1.0  # Completely overcast
    P = 101325.0

    result = longwave_in_radiation_func(TC, RH, n_c, P)

    # Under overcast conditions, result should be higher due to cloud emissivity
    assert result > 0
    assert 350.0 <= result <= 450.0  # Reasonable range for overcast conditions
