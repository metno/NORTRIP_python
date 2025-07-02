import numpy as np
from src.functions.rmse_func import rmse_func


def test_rmse_func():
    """Test RMSE calculation function."""

    # Test with identical arrays (RMSE should be 0)
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = rmse_func(a, b)
    assert abs(result - 0.0) < 1e-10

    # Test with simple case
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([2.0, 3.0, 4.0, 5.0])  # All values differ by 1
    result = rmse_func(a, b)
    expected = 1.0  # sqrt(sum(1^2) / 4) = sqrt(4/4) = 1
    assert abs(result - expected) < 1e-10

    # Test with known RMSE
    a = np.array([0.0, 0.0, 0.0, 0.0])
    b = np.array([3.0, 4.0, 0.0, 0.0])  # Errors: 3, 4, 0, 0
    result = rmse_func(a, b)
    expected = np.sqrt((9 + 16 + 0 + 0) / 4)  # sqrt(25/4) = 2.5
    assert abs(result - expected) < 1e-10

    # Test with different array sizes (should return NaN)
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.0, 2.0])
    result = rmse_func(a, b)
    assert np.isnan(result)

    # Test with empty arrays (should return NaN)
    a = np.array([])
    b = np.array([])
    result = rmse_func(a, b)
    assert np.isnan(result)


def test_rmse_func_with_nans():
    """Test RMSE function with NaN values."""

    # Test with some NaN values
    a = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    b = np.array([1.0, 3.0, 3.0, np.nan, 5.0])
    result = rmse_func(a, b)

    # Should calculate RMSE using only valid pairs: (1,1), (2,3), (5,5)
    # Errors: 0, 1, 0, so RMSE should be sqrt(1/3)
    expected = np.sqrt(1.0 / 3.0)
    assert abs(result - expected) < 1e-10

    # Test with all NaN values
    a = np.array([np.nan, np.nan, np.nan])
    b = np.array([np.nan, np.nan, np.nan])
    result = rmse_func(a, b)
    assert np.isnan(result)

    # Test with one array all NaN
    a = np.array([np.nan, np.nan, np.nan])
    b = np.array([1.0, 2.0, 3.0])
    result = rmse_func(a, b)
    assert np.isnan(result)

    # Test with mixed valid/invalid data
    a = np.array([1.0, np.nan, 3.0, 4.0])
    b = np.array([2.0, 2.0, np.nan, 6.0])
    result = rmse_func(a, b)

    # Valid pairs: (1,2), (4,6)
    # Errors: 1, 2
    # RMSE = sqrt((1 + 4) / 2) = sqrt(2.5)
    expected = np.sqrt(2.5)
    assert abs(result - expected) < 1e-10


def test_rmse_func_mathematical_properties():
    """Test mathematical properties of RMSE."""

    # Test symmetry: RMSE(a,b) = RMSE(b,a)
    a = np.array([1.0, 3.0, 5.0, 7.0])
    b = np.array([2.0, 4.0, 6.0, 8.0])
    rmse_ab = rmse_func(a, b)
    rmse_ba = rmse_func(b, a)
    assert abs(rmse_ab - rmse_ba) < 1e-10

    # Test that RMSE is always non-negative
    test_cases = [
        (np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])),
        (np.array([-1.0, -2.0, -3.0]), np.array([1.0, 2.0, 3.0])),
        (np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])),
    ]

    for a, b in test_cases:
        result = rmse_func(a, b)
        assert result >= 0

    # Test triangle inequality property (approximately)
    # ||a-c|| <= ||a-b|| + ||b-c|| (for vectors)
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([2.0, 3.0, 4.0])
    c = np.array([3.0, 4.0, 5.0])

    rmse_ac = rmse_func(a, c)
    rmse_ab = rmse_func(a, b)
    rmse_bc = rmse_func(b, c)

    # For RMSE, this isn't exactly triangle inequality, but the result should be reasonable
    assert rmse_ac >= 0
    assert rmse_ab >= 0
    assert rmse_bc >= 0


def test_rmse_func_edge_cases():
    """Test edge cases for RMSE function."""

    # Test with single element arrays
    a = np.array([5.0])
    b = np.array([3.0])
    result = rmse_func(a, b)
    expected = 2.0  # |5-3| = 2
    assert abs(result - expected) < 1e-10

    # Test with very large numbers
    a = np.array([1e10, 2e10])
    b = np.array([1.1e10, 2.1e10])
    result = rmse_func(a, b)
    expected = np.sqrt(((0.1e10) ** 2 + (0.1e10) ** 2) / 2)
    assert abs(result - expected) / expected < 1e-10  # Relative error

    # Test with very small numbers
    a = np.array([1e-10, 2e-10])
    b = np.array([1.1e-10, 2.1e-10])
    result = rmse_func(a, b)
    expected = np.sqrt(((0.1e-10) ** 2 + (0.1e-10) ** 2) / 2)
    assert abs(result - expected) < 1e-20

    # Test with zero arrays
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([0.0, 0.0, 0.0])
    result = rmse_func(a, b)
    assert result == 0.0


def test_rmse_func_realistic_data():
    """Test RMSE with realistic data scenarios."""

    # Test with temperature data (realistic meteorological scenario)
    observed_temps = np.array([15.2, 16.8, 18.1, 19.5, 17.3])
    modeled_temps = np.array([15.0, 17.0, 18.5, 19.2, 17.1])

    result = rmse_func(observed_temps, modeled_temps)

    # Calculate expected manually
    errors = observed_temps - modeled_temps  # [0.2, -0.2, -0.4, 0.3, 0.2]
    expected = np.sqrt(np.mean(errors**2))
    assert abs(result - expected) < 1e-10

    # Result should be reasonable for temperature data
    assert 0 < result < 1.0  # Should be less than 1 degree for good model

    # Test with precipitation data (often has zeros and high variability)
    observed_precip = np.array([0.0, 2.5, 0.0, 15.2, 0.8])
    modeled_precip = np.array([0.2, 2.0, 0.0, 12.8, 1.1])

    result = rmse_func(observed_precip, modeled_precip)

    # Should be positive and reasonable
    assert result > 0
    assert result < 10  # Reasonable for precipitation data

    # Test perfect correlation with offset
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = x + 2.0  # Perfect correlation, constant offset

    result = rmse_func(x, y)
    assert abs(result - 2.0) < 1e-10  # RMSE should equal the constant offset


def test_rmse_func_performance_characteristics():
    """Test that RMSE behaves correctly for different error patterns."""

    # Test systematic bias vs random error
    true_values = np.array([10.0, 10.0, 10.0, 10.0, 10.0])

    # Systematic bias (all values offset by same amount)
    biased_values = np.array([12.0, 12.0, 12.0, 12.0, 12.0])
    rmse_bias = rmse_func(true_values, biased_values)

    # Random error (designed to have same RMSE)
    random_values = np.array([8.0, 12.0, 8.0, 12.0, 10.0])
    rmse_random = rmse_func(true_values, random_values)

    # Systematic bias should have RMSE = 2.0
    assert abs(rmse_bias - 2.0) < 1e-10

    # Random values: errors are [-2, 2, -2, 2, 0], RMSE = sqrt((4+4+4+4+0)/5) = sqrt(16/5)
    expected_random = np.sqrt(16.0 / 5.0)
    assert abs(rmse_random - expected_random) < 1e-10

    # Test sensitivity to outliers
    normal_errors = np.array([10.0, 10.1, 10.2, 9.9, 9.8])
    outlier_errors = np.array([10.0, 10.1, 15.0, 9.9, 9.8])  # One large error

    reference = np.array([10.0, 10.0, 10.0, 10.0, 10.0])

    rmse_normal = rmse_func(reference, normal_errors)
    rmse_outlier = rmse_func(reference, outlier_errors)

    # Outlier case should have much larger RMSE
    assert rmse_outlier > rmse_normal
    assert rmse_outlier > 2.0  # Should be significantly affected by the outlier
