import numpy as np
from src.functions.dewpoint_from_rh_func import dewpoint_from_rh_func


def test_dewpoint_from_rh_func():
    """Test dewpoint calculation from temperature and relative humidity."""

    # Test at 100% RH - dewpoint should equal air temperature
    TC = 20.0
    RH = 100.0
    result = dewpoint_from_rh_func(TC, RH)
    assert abs(result - TC) < 0.001

    # Test at 50% RH and 20°C
    TC = 20.0
    RH = 50.0
    result = dewpoint_from_rh_func(TC, RH)
    # Dewpoint should be lower than air temperature
    assert result < TC
    # Approximate expected value for 20°C, 50% RH is around 9.3°C
    assert abs(result - 9.3) < 1.0

    # Test at 0°C and 80% RH
    TC = 0.0
    RH = 80.0
    result = dewpoint_from_rh_func(TC, RH)
    assert result < TC
    # Should be around -3°C
    assert abs(result - (-3.0)) < 1.0

    # Test with negative temperature
    TC = -10.0
    RH = 70.0
    result = dewpoint_from_rh_func(TC, RH)
    assert result < TC

    # Test with array inputs
    TC_array = np.array([0.0, 10.0, 20.0, 30.0])
    RH_array = np.array([50.0, 60.0, 70.0, 80.0])
    result_array = dewpoint_from_rh_func(TC_array, RH_array)

    # Ensure result is array
    result_array = np.asarray(result_array)
    assert result_array.shape == (4,)
    # All dewpoints should be less than air temperatures (except at 100% RH)
    assert np.all(result_array <= TC_array)

    # Test edge case: very low RH
    TC = 25.0
    RH = 5.0
    result = dewpoint_from_rh_func(TC, RH)
    assert result < TC
    # Should be significantly lower
    assert (TC - result) > 15.0

    # Test edge case: very high RH (but not 100%)
    TC = 15.0
    RH = 99.0
    result = dewpoint_from_rh_func(TC, RH)
    assert result < TC
    # Should be very close to air temperature
    assert (TC - result) < 1.0


def test_dewpoint_from_rh_func_physical_validity():
    """Test that dewpoint calculations follow physical constraints."""

    # Test that increasing RH at constant temperature increases dewpoint
    TC = 20.0
    RH_values = [30.0, 50.0, 70.0, 90.0]
    dewpoints = [dewpoint_from_rh_func(TC, rh) for rh in RH_values]

    # Dewpoints should increase with increasing RH
    for i in range(1, len(dewpoints)):
        assert dewpoints[i] > dewpoints[i - 1]

    # Test that all dewpoints are physically reasonable
    for dp in dewpoints:
        assert dp <= TC  # Dewpoint cannot exceed air temperature
        assert dp > -50.0  # Reasonable lower bound

    # Test with extreme values
    TC = 40.0
    RH = 95.0
    result = dewpoint_from_rh_func(TC, RH)
    assert result <= TC
    assert result > 30.0  # Should be high dewpoint

    # Test consistency: if we have dewpoint, we should be able to reverse engineer RH
    TC = 25.0
    original_RH = 65.0
    calculated_dewpoint = dewpoint_from_rh_func(TC, original_RH)

    # Using Magnus formula constants
    a = 6.1121
    b = 17.67
    c = 243.5

    # Calculate what RH would be for this dewpoint
    esat = a * np.exp(b * TC / (c + TC))
    eair = a * np.exp(b * calculated_dewpoint / (c + calculated_dewpoint))
    reverse_RH = 100 * eair / esat

    assert abs(reverse_RH - original_RH) < 0.01
