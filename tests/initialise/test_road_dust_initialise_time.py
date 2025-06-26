import numpy as np
import datetime
import pytest
from initialise.road_dust_initialise_time import (
    road_dust_initialise_time,
    _parse_date_string,
    _find_matching_time_index,
    time_config,
)
from input_classes import input_metadata
import constants


def create_test_date_data():
    """Create test date data array."""
    n_date = 24  # 24 hours of data
    date_data = np.full((constants.num_date_index, n_date, constants.n_roads), -99.0)

    # Fill with hourly data for 2023-01-01
    for i in range(n_date):
        date_data[constants.year_index, i, 0] = 2023
        date_data[constants.month_index, i, 0] = 1
        date_data[constants.day_index, i, 0] = 1
        date_data[constants.hour_index, i, 0] = i
        date_data[constants.minute_index, i, 0] = 0
        date_data[constants.datenum_index, i, 0] = (
            738521.0 + i / 24.0
        )  # 2023-01-01 + hour

    return date_data


def create_test_metadata():
    """Create test metadata object."""
    metadata = input_metadata()
    metadata.start_date_str = ""
    metadata.end_date_str = ""
    metadata.start_date_save_str = ""
    metadata.end_date_save_str = ""
    metadata.n_save_subdate = 1
    metadata.start_subdate_save_str = []
    metadata.end_subdate_save_str = []
    return metadata


# Helper function tests (2 tests)
def test_parse_date_string():
    """Test date string parsing with NORTRIP format."""
    result = _parse_date_string("01.01.2023 12:30")
    expected = datetime.datetime(2023, 1, 1, 12, 30)
    assert result == expected


def test_find_matching_time_index():
    """Test finding time index in date data."""
    date_data = create_test_date_data()
    target_datetime = datetime.datetime(2023, 1, 1, 5, 0)
    result = _find_matching_time_index(date_data, target_datetime, 24)
    assert result == 5


# Config test (1 test)
def test_time_config_defaults():
    """Test time_config dataclass has correct defaults."""
    config = time_config()
    assert config.min_time == 0
    assert config.max_time == 0
    assert config.dt == 1.0
    assert config.time_bad == 0


# Main function tests (5 tests)
def test_road_dust_initialise_time_basic():
    """Test basic time initialization without date restrictions."""
    date_data = create_test_date_data()
    metadata = create_test_metadata()

    config = road_dust_initialise_time(date_data, 24, metadata)

    # Check 0-based indexing
    assert config.min_time == 0
    assert config.max_time == 24  # Exclusive end
    assert config.max_time_inputdata == 24
    assert config.time_bad == 0
    assert config.dt == pytest.approx(
        1.0, abs=1e-6
    )  # 1 hour time step (allow floating point precision)


def test_road_dust_initialise_time_with_dates():
    """Test time initialization with start and end dates."""
    date_data = create_test_date_data()
    metadata = create_test_metadata()
    metadata.start_date_str = "01.01.2023 02:00"
    metadata.end_date_str = "01.01.2023 05:00"

    config = road_dust_initialise_time(date_data, 24, metadata)

    # Check 0-based indexing: start at index 2, end at index 6 (exclusive)
    assert config.min_time == 2
    assert config.max_time == 6  # Index 5 + 1 for exclusive range
    assert config.time_bad == 0


def test_road_dust_initialise_time_with_save_dates():
    """Test time initialization with save date configuration."""
    date_data = create_test_date_data()
    metadata = create_test_metadata()
    metadata.start_date_save_str = "01.01.2023 03:00"
    metadata.end_date_save_str = "01.01.2023 06:00"

    config = road_dust_initialise_time(date_data, 24, metadata)

    # Check save time configuration
    assert config.min_time_save == 3
    assert config.max_time_save == 7  # Index 6 + 1 for exclusive range


def test_road_dust_initialise_time_date_not_found():
    """Test error handling when date is not found in data."""
    date_data = create_test_date_data()
    metadata = create_test_metadata()
    metadata.start_date_str = "01.01.2024 10:00"  # Date not in data

    config = road_dust_initialise_time(date_data, 24, metadata)

    assert config.time_bad == 1


def test_road_dust_initialise_time_fortran_flag():
    """Test that fortran flag overrides date restrictions."""
    date_data = create_test_date_data()
    metadata = create_test_metadata()
    metadata.start_date_str = "01.01.2023 05:00"
    metadata.end_date_str = "01.01.2023 10:00"

    config = road_dust_initialise_time(date_data, 24, metadata, use_fortran_flag=True)

    # Should run all data regardless of date strings
    assert config.min_time == 0
    assert config.max_time == 24
    assert config.time_bad == 0
