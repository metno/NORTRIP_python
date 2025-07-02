import numpy as np
from src.functions.rh_from_dewpoint_func import rh_from_dewpoint_func


def test_rh_from_dewpoint_func():
    """Test relative humidity calculation from temperature and dewpoint."""

    # Test when dewpoint equals air temperature (100% RH)
    TC = 20.0
    TC_dewpoint = 20.0
    result = rh_from_dewpoint_func(TC, TC_dewpoint)
    assert abs(result - 100.0) < 0.001

    # Test typical conditions
    TC = 20.0
    TC_dewpoint = 10.0  # 10°C dewpoint
    result = rh_from_dewpoint_func(TC, TC_dewpoint)

    # RH should be positive and less than 100%
    assert 0 < result < 100
    # For 20°C air temp and 10°C dewpoint, RH should be around 50%
    assert 45 < result < 60

    # Test with negative air temperature
    TC = -5.0
    TC_dewpoint = -10.0
    result = rh_from_dewpoint_func(TC, TC_dewpoint)
    assert 0 < result < 100

    # Test with very dry conditions (large difference)
    TC = 30.0
    TC_dewpoint = 0.0
    result = rh_from_dewpoint_func(TC, TC_dewpoint)
    assert 0 < result < 30  # Should be quite low


def test_rh_from_dewpoint_func_physical_constraints():
    """Test that the function follows physical constraints."""

    # Dewpoint cannot exceed air temperature in nature
    TC = 15.0
    TC_dewpoint = 15.0  # Equal case
    result = rh_from_dewpoint_func(TC, TC_dewpoint)
    assert abs(result - 100.0) < 0.001

    # Test with various dewpoint depressions
    dewpoint_depressions = [0, 2, 5, 10, 15]  # Difference between air temp and dewpoint
    base_temp = 20.0

    rh_values = []
    for depression in dewpoint_depressions:
        TC_dewpoint = base_temp - depression
        rh = rh_from_dewpoint_func(base_temp, TC_dewpoint)
        rh_values.append(rh)

        # RH should be between 0 and 100%
        assert 0 <= rh <= 100

    # RH should decrease as dewpoint depression increases
    for i in range(1, len(rh_values)):
        assert rh_values[i] < rh_values[i - 1]


def test_rh_from_dewpoint_func_array_inputs():
    """Test function with array inputs."""

    # Test with arrays
    TC_array = np.array([10.0, 15.0, 20.0, 25.0])
    TC_dewpoint_array = np.array([5.0, 10.0, 15.0, 20.0])

    result_array = rh_from_dewpoint_func(TC_array, TC_dewpoint_array)
    result_array = np.asarray(result_array)

    assert result_array.shape == (4,)

    # All results should be between 0 and 100%
    assert np.all(result_array >= 0)
    assert np.all(result_array <= 100)

    # Test individual calculations match array calculation
    for i in range(len(TC_array)):
        individual_result = rh_from_dewpoint_func(TC_array[i], TC_dewpoint_array[i])
        assert abs(result_array[i] - individual_result) < 0.001


def test_rh_from_dewpoint_func_magnus_formula():
    """Test that function correctly implements Magnus formula."""

    TC = 25.0
    TC_dewpoint = 15.0

    result = rh_from_dewpoint_func(TC, TC_dewpoint)

    # Calculate expected using Magnus formula directly
    a = 6.1121
    b = 17.67
    c = 243.5

    esat = a * np.exp(b * TC / (c + TC))
    eair = a * np.exp(b * TC_dewpoint / (c + TC_dewpoint))
    expected_rh = 100 * eair / esat

    assert abs(result - expected_rh) < 0.001


def test_rh_from_dewpoint_func_extreme_conditions():
    """Test function with extreme conditions."""

    # Very cold conditions
    TC = -30.0
    TC_dewpoint = -35.0
    result = rh_from_dewpoint_func(TC, TC_dewpoint)
    assert 0 < result < 100

    # Very warm conditions
    TC = 40.0
    TC_dewpoint = 30.0
    result = rh_from_dewpoint_func(TC, TC_dewpoint)
    assert 0 < result < 100

    # Large dewpoint depression
    TC = 35.0
    TC_dewpoint = 5.0
    result = rh_from_dewpoint_func(TC, TC_dewpoint)
    assert 0 < result < 25  # Should be very low humidity

    # Small dewpoint depression
    TC = 22.0
    TC_dewpoint = 21.5
    result = rh_from_dewpoint_func(TC, TC_dewpoint)
    assert 90 < result < 100  # Should be very high humidity


def test_rh_from_dewpoint_func_consistency():
    """Test consistency with dewpoint_from_rh_func (inverse function)."""

    # This test assumes we have access to dewpoint_from_rh_func
    # We'll test internal consistency by checking round-trip calculations

    # Start with known conditions
    TC = 18.0
    original_rh = 75.0

    # We can manually calculate what the dewpoint should be
    a = 6.1121
    b = 17.67
    c = 243.5

    # Calculate dewpoint from the original RH
    esat = a * np.exp(b * TC / (c + TC))
    eair = original_rh / 100.0 * esat
    calculated_dewpoint = c * np.log(eair / a) / (b - np.log(eair / a))

    # Now use our function to calculate RH from this dewpoint
    calculated_rh = rh_from_dewpoint_func(TC, calculated_dewpoint)

    # Should get back the original RH
    assert abs(calculated_rh - original_rh) < 0.01


def test_rh_from_dewpoint_func_temperature_effects():
    """Test temperature effects on RH calculation."""

    # Fixed dewpoint, varying air temperature
    TC_dewpoint = 10.0
    air_temperatures = [10.0, 15.0, 20.0, 25.0, 30.0]

    rh_values = []
    for TC in air_temperatures:
        rh = rh_from_dewpoint_func(TC, TC_dewpoint)
        rh_values.append(rh)
        assert 0 <= rh <= 100

    # RH should decrease as air temperature increases (for fixed dewpoint)
    for i in range(1, len(rh_values)):
        assert rh_values[i] < rh_values[i - 1]

    # First value should be 100% (when air temp = dewpoint)
    assert abs(rh_values[0] - 100.0) < 0.001
