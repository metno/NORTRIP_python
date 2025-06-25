import numpy as np
from pd_util.forward_fill_missing import forward_fill_missing


def test_forward_fill_missing_basic():
    """Test basic forward filling functionality."""
    data = np.array([1.0, -99.0, 3.0, -99.0, 5.0])
    nodata = -99.0

    result_data, missing_indices = forward_fill_missing(data, nodata)

    # Check that missing values are forward filled
    expected = np.array([1.0, 1.0, 3.0, 3.0, 5.0])
    np.testing.assert_array_equal(result_data, expected)

    # Check missing indices
    assert missing_indices == [1, 3]


def test_forward_fill_missing_with_nan():
    """Test forward filling with NaN values."""
    data = np.array([2.0, np.nan, 4.0, np.nan, 6.0])
    nodata = -99.0

    result_data, missing_indices = forward_fill_missing(data, nodata)

    # Check that NaN values are forward filled
    expected = np.array([2.0, 2.0, 4.0, 4.0, 6.0])
    np.testing.assert_array_equal(result_data, expected)

    # Check missing indices
    assert missing_indices == [1, 3]


def test_forward_fill_missing_first_value_missing():
    """Test forward filling when first value is missing (should remain missing)."""
    data = np.array([-99.0, 2.0, -99.0, 4.0])
    nodata = -99.0

    result_data, missing_indices = forward_fill_missing(data, nodata)

    # First value should remain missing, second missing value should be filled
    expected = np.array([-99.0, 2.0, 2.0, 4.0])
    np.testing.assert_array_equal(result_data, expected)

    # Check missing indices
    assert missing_indices == [0, 2]


def test_forward_fill_missing_no_missing():
    """Test forward filling when no values are missing."""
    data = np.array([1.0, 2.0, 3.0, 4.0])
    nodata = -99.0

    result_data, missing_indices = forward_fill_missing(data, nodata)

    # Data should remain unchanged
    np.testing.assert_array_equal(result_data, data)

    # No missing indices
    assert missing_indices == []


def test_forward_fill_missing_all_missing():
    """Test forward filling when all values are missing."""
    data = np.array([-99.0, -99.0, -99.0])
    nodata = -99.0

    result_data, missing_indices = forward_fill_missing(data, nodata)

    # All values should remain missing
    expected = np.array([-99.0, -99.0, -99.0])
    np.testing.assert_array_equal(result_data, expected)

    # All indices should be missing
    assert missing_indices == [0, 1, 2]


def test_forward_fill_missing_mixed_nodata_and_nan():
    """Test forward filling with both nodata values and NaN."""
    data = np.array([1.0, -99.0, np.nan, 4.0, -99.0])
    nodata = -99.0

    result_data, missing_indices = forward_fill_missing(data, nodata)

    # Both nodata and NaN should be forward filled
    expected = np.array([1.0, 1.0, 1.0, 4.0, 4.0])
    np.testing.assert_array_equal(result_data, expected)

    # Check missing indices
    assert missing_indices == [1, 2, 4]
