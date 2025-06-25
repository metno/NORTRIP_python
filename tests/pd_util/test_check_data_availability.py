import numpy as np
from pd_util.check_data_availability import check_data_availability


def test_check_data_availability_all_valid():
    """Test with all valid data."""
    data = np.array([1.0, 2.0, 3.0, 4.0])
    available = 1
    nodata = -99.0

    result_data, result_available, missing_indices = check_data_availability(
        data, available, nodata
    )

    assert np.array_equal(result_data, data)
    assert result_available == 1
    assert missing_indices == []


def test_check_data_availability_some_missing():
    """Test with some missing data."""
    data = np.array([1.0, -99.0, 3.0, np.nan])
    available = 1
    nodata = -99.0

    result_data, result_available, missing_indices = check_data_availability(
        data, available, nodata
    )

    assert np.array_equal(result_data, data, equal_nan=True)
    assert result_available == 1
    assert missing_indices == [1, 3]


def test_check_data_availability_all_missing():
    """Test with all data missing."""
    data = np.array([-99.0, np.nan, -99.0])
    available = 1
    nodata = -99.0

    result_data, result_available, missing_indices = check_data_availability(
        data, available, nodata
    )

    assert np.array_equal(result_data, data, equal_nan=True)
    assert result_available == 0
    assert missing_indices == [0, 1, 2]


def test_check_data_availability_not_available():
    """Test when data is not initially available."""
    data = np.array([1.0, 2.0, 3.0])
    available = 0
    nodata = -99.0

    result_data, result_available, missing_indices = check_data_availability(
        data, available, nodata
    )

    assert np.array_equal(result_data, data)
    assert result_available == 0
    assert missing_indices == []


def test_check_data_availability_only_nodata():
    """Test with only nodata values."""
    data = np.array([-99.0, -99.0, -99.0])
    available = 1
    nodata = -99.0

    result_data, result_available, missing_indices = check_data_availability(
        data, available, nodata
    )

    assert np.array_equal(result_data, data)
    assert result_available == 0
    assert missing_indices == [0, 1, 2]


def test_check_data_availability_only_nan():
    """Test with only NaN values."""
    data = np.array([np.nan, np.nan])
    available = 1
    nodata = -99.0

    result_data, result_available, missing_indices = check_data_availability(
        data, available, nodata
    )

    assert np.array_equal(result_data, data, equal_nan=True)
    assert result_available == 0
    assert missing_indices == [0, 1]


def test_check_data_availability_mixed_missing():
    """Test with mixed nodata and NaN values."""
    data = np.array([1.0, -99.0, np.nan, 4.0, -99.0])
    available = 1
    nodata = -99.0

    result_data, result_available, missing_indices = check_data_availability(
        data, available, nodata
    )

    assert np.array_equal(result_data, data, equal_nan=True)
    assert result_available == 1
    assert missing_indices == [1, 2, 4]
