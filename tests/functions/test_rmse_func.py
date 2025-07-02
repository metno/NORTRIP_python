import numpy as np
from src.functions import rmse_func


def test_rmse_func_basic():
    """Test basic RMSE calculation."""
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    b = np.array([1.1, 2.1, 2.9, 4.1, 4.9])

    rmse = rmse_func(a, b)

    # Should be a small positive value
    assert rmse > 0
    assert rmse < 1.0  # Given the small differences, RMSE should be < 1


def test_rmse_func_identical_arrays():
    """Test with identical arrays."""
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    b = a.copy()

    rmse = rmse_func(a, b)

    # Should be zero for identical arrays
    assert rmse == 0.0


def test_rmse_func_with_nan():
    """Test with NaN values."""
    a = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    b = np.array([1.1, 2.1, 3.0, 4.1, np.nan])

    rmse = rmse_func(a, b)

    # Should calculate RMSE using only valid pairs
    assert not np.isnan(rmse)
    assert rmse > 0


def test_rmse_func_all_nan():
    """Test with all NaN values."""
    a = np.array([np.nan, np.nan, np.nan])
    b = np.array([np.nan, np.nan, np.nan])

    rmse = rmse_func(a, b)

    # Should return NaN when no valid data pairs exist
    assert np.isnan(rmse)


def test_rmse_func_different_lengths():
    """Test with arrays of different lengths."""
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.0, 2.0])

    rmse = rmse_func(a, b)

    # Should return NaN for incompatible arrays
    assert np.isnan(rmse)
