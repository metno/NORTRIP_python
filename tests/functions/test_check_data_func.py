import numpy as np
from src.functions.check_data_func import check_data_func


def test_check_data_func():
    """Test data checking and forward fill functionality."""

    # Test with valid data
    val = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    available = 1
    nodata = -999.0

    result_val, result_available, missing_flag = check_data_func(val, available, nodata)

    assert result_available == 1
    assert np.array_equal(result_val, val)
    assert len(missing_flag) == 0

    # Test with some missing data (nodata values)
    val = np.array([1.0, -999.0, 3.0, -999.0, 5.0])
    result_val, result_available, missing_flag = check_data_func(val, available, nodata)

    assert result_available == 1
    assert result_val[0] == 1.0
    assert result_val[1] == 1.0  # Forward filled
    assert result_val[2] == 3.0
    assert result_val[3] == 3.0  # Forward filled
    assert result_val[4] == 5.0
    assert 1 in missing_flag
    assert 3 in missing_flag

    # Test with NaN values
    val = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
    result_val, result_available, missing_flag = check_data_func(val, available, nodata)

    assert result_available == 1
    assert result_val[0] == 1.0
    assert result_val[1] == 1.0  # Forward filled
    assert result_val[2] == 3.0
    assert result_val[3] == 3.0  # Forward filled
    assert result_val[4] == 5.0
    assert 1 in missing_flag
    assert 3 in missing_flag

    # Test with missing data at the beginning
    val = np.array([-999.0, -999.0, 3.0, 4.0, 5.0])
    result_val, result_available, missing_flag = check_data_func(val, available, nodata)

    assert result_available == 1
    assert result_val[0] == 3.0  # Backward filled from first valid value
    assert result_val[1] == 3.0  # Backward filled from first valid value
    assert result_val[2] == 3.0
    assert result_val[3] == 4.0
    assert result_val[4] == 5.0
    assert 0 in missing_flag

    # Test with all data missing
    val = np.array([-999.0, -999.0, -999.0])
    result_val, result_available, missing_flag = check_data_func(val, available, nodata)

    assert result_available == 0

    # Test with all NaN data
    val = np.array([np.nan, np.nan, np.nan])
    result_val, result_available, missing_flag = check_data_func(val, available, nodata)

    assert result_available == 0

    # Test with data already unavailable
    val = np.array([1.0, 2.0, 3.0])
    available = 0
    result_val, result_available, missing_flag = check_data_func(val, available, nodata)

    assert result_available == 0
    assert len(missing_flag) == 0

    # Test single element array with missing data
    val = np.array([-999.0])
    available = 1
    result_val, result_available, missing_flag = check_data_func(val, available, nodata)

    assert result_available == 0

    # Test single element array with valid data
    val = np.array([5.0])
    result_val, result_available, missing_flag = check_data_func(val, available, nodata)

    assert result_available == 1
    assert result_val[0] == 5.0
    assert len(missing_flag) == 0


def test_check_data_func_edge_cases():
    """Test edge cases for check_data_func."""

    # Test with mixed NaN and nodata
    val = np.array([1.0, np.nan, -999.0, 4.0])
    available = 1
    nodata = -999.0

    result_val, result_available, missing_flag = check_data_func(val, available, nodata)

    assert result_available == 1
    assert result_val[0] == 1.0
    assert result_val[1] == 1.0  # Forward filled
    assert result_val[2] == 1.0  # Forward filled
    assert result_val[3] == 4.0
    assert 1 in missing_flag
    assert 2 in missing_flag

    # Test array that starts with valid data then all missing
    val = np.array([1.0, 2.0, np.nan, np.nan, np.nan])
    result_val, result_available, missing_flag = check_data_func(val, available, nodata)

    assert result_available == 1
    assert result_val[0] == 1.0
    assert result_val[1] == 2.0
    assert result_val[2] == 2.0  # Forward filled
    assert result_val[3] == 2.0  # Forward filled
    assert result_val[4] == 2.0  # Forward filled
    assert 2 in missing_flag
    assert 3 in missing_flag
    assert 4 in missing_flag
