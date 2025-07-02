import numpy as np
from src.functions import dewpoint_from_rh_func


def test_dewpoint_from_rh_func_basic():
    """Test basic functionality of dewpoint from RH calculation."""
    # Test case: 20°C air temperature, 50% RH
    TC = 20.0
    RH = 50.0

    dewpoint = dewpoint_from_rh_func(TC, RH)

    # Dewpoint should be less than air temperature
    assert dewpoint < TC

    # For this case, dewpoint should be around 9-10°C
    assert 8 <= dewpoint <= 11


def test_dewpoint_from_rh_func_edge_cases():
    """Test edge cases."""
    # When RH is 100%, dewpoint should equal air temperature
    TC = 15.0
    RH = 100.0

    dewpoint = dewpoint_from_rh_func(TC, RH)

    # Should be very close to air temperature
    assert abs(dewpoint - TC) < 0.1


def test_dewpoint_from_rh_func_numpy_arrays():
    """Test with numpy arrays."""
    TC = np.array([20.0, 15.0, 10.0])
    RH = np.array([50.0, 70.0, 90.0])

    dewpoint = dewpoint_from_rh_func(TC, RH)

    # Should return array of same shape
    assert dewpoint.shape == TC.shape
    # All dewpoints should be less than or equal to air temperature
    assert np.all(dewpoint <= TC)
