import numpy as np
from src.functions import rh_from_dewpoint_func


def test_rh_from_dewpoint_func_basic():
    """Test basic functionality of RH from dewpoint calculation."""
    # Test case: 20°C air temperature, 10°C dewpoint
    TC = 20.0
    TC_dewpoint = 10.0

    rh = rh_from_dewpoint_func(TC, TC_dewpoint)

    # RH should be between 0 and 100%
    assert 0 <= rh <= 100

    # For this case, RH should be around 52%
    assert 50 <= rh <= 60


def test_rh_from_dewpoint_func_edge_cases():
    """Test edge cases."""
    # When dewpoint equals air temperature, RH should be 100%
    TC = 15.0
    TC_dewpoint = 15.0

    rh = rh_from_dewpoint_func(TC, TC_dewpoint)

    # Should be very close to 100%
    assert 99 <= rh <= 101


def test_rh_from_dewpoint_func_numpy_arrays():
    """Test with numpy arrays."""
    TC = np.array([20.0, 15.0, 10.0])
    TC_dewpoint = np.array([10.0, 5.0, 0.0])

    rh = rh_from_dewpoint_func(TC, TC_dewpoint)

    # Should return array of same shape
    assert rh.shape == TC.shape
    # All values should be valid RH values
    assert np.all((rh >= 0) & (rh <= 100))
