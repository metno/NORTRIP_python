import numpy as np
from src.functions import check_data_func


def test_check_data_func_no_missing_data():
    """Test with no missing data."""
    val = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    available = 1
    nodata = -99.0

    result_val, result_available, missing_flag = check_data_func(val, available, nodata)

    # Should return unchanged data
    assert np.array_equal(result_val, val)
    assert result_available == 1
    assert len(missing_flag) == 0


def test_check_data_func_with_missing_data():
    """Test with some missing data."""
    val = np.array([1.0, -99.0, 3.0, -99.0, 5.0])
    available = 1
    nodata = -99.0

    result_val, result_available, missing_flag = check_data_func(val, available, nodata)

    # Missing values should be forward filled
    expected = np.array([1.0, 1.0, 3.0, 3.0, 5.0])
    assert np.array_equal(result_val, expected)
    assert result_available == 1
    assert len(missing_flag) > 0


def test_check_data_func_all_missing():
    """Test with all data missing."""
    val = np.array([-99.0, -99.0, -99.0])
    available = 1
    nodata = -99.0

    result_val, result_available, missing_flag = check_data_func(val, available, nodata)

    # Should mark as unavailable
    assert result_available == 0


def test_check_data_func_with_nan():
    """Test with NaN values."""
    val = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
    available = 1
    nodata = -99.0

    result_val, result_available, missing_flag = check_data_func(val, available, nodata)

    # NaN values should be forward filled
    expected = np.array([1.0, 1.0, 3.0, 3.0, 5.0])
    assert np.array_equal(result_val, expected)
    assert result_available == 1
