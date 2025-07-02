import numpy as np
import datetime
from initialise.road_dust_initialise_time import (
    road_dust_initialise_time,
    _parse_date_string,
    _find_time_index,
)
from input_classes import converted_data, input_metadata
import constants


def test_parse_date_string():
    """Test date string parsing functionality."""
    # Test standard format
    year, month, day, hour, minute, second = _parse_date_string("2023-03-15 14:30:00")
    assert year == 2023
    assert month == 3
    assert day == 15
    assert hour == 14
    assert minute == 30
    assert second == 0

    # Test date only format
    year, month, day, hour, minute, second = _parse_date_string("2023-03-15")
    assert year == 2023
    assert month == 3
    assert day == 15
    assert hour == 0
    assert minute == 0
    assert second == 0

    # Test alternative format
    year, month, day, hour, minute, second = _parse_date_string("2023.03.15 14:30:00")
    assert year == 2023
    assert month == 3
    assert day == 15
    assert hour == 14
    assert minute == 30
    assert second == 0

    # Test European format (day.month.year)
    year, month, day, hour, minute, second = _parse_date_string("01.10.2010 01:00")
    assert year == 2010
    assert month == 10
    assert day == 1
    assert hour == 1
    assert minute == 0
    assert second == 0

    # Test empty string
    year, month, day, hour, minute, second = _parse_date_string("")
    assert year == 0
    assert month == 0
    assert day == 0
    assert hour == 0
    assert minute == 0
    assert second == 0

    # Test invalid format
    year, month, day, hour, minute, second = _parse_date_string("invalid date")
    assert year == 0
    assert month == 0
    assert day == 0
    assert hour == 0
    assert minute == 0
    assert second == 0


def test_find_time_index():
    """Test finding time index in date_data array."""
    # Create sample date data
    n_date = 5
    n_roads = 1
    date_data = np.zeros((constants.num_date_index, n_date, n_roads))

    # Fill with sample dates
    years = [2023, 2023, 2023, 2023, 2023]
    months = [3, 3, 3, 3, 3]
    days = [15, 15, 15, 15, 15]
    hours = [10, 11, 12, 13, 14]

    date_data[constants.year_index, :, 0] = years
    date_data[constants.month_index, :, 0] = months
    date_data[constants.day_index, :, 0] = days
    date_data[constants.hour_index, :, 0] = hours

    # Test finding existing time
    index = _find_time_index(2023, 3, 15, 12, date_data)
    assert index == 2

    # Test finding non-existing time
    index = _find_time_index(2023, 3, 16, 12, date_data)
    assert index == -1


def test_road_dust_initialise_time_basic():
    """Test basic functionality of road_dust_initialise_time."""
    # Create sample date data
    n_date = 10
    n_roads = 1
    date_data = np.zeros((constants.num_date_index, n_date, n_roads))

    # Fill with sample dates (hourly data)
    base_date = datetime.datetime(2023, 3, 15, 10, 0, 0)
    for i in range(n_date):
        current_date = base_date + datetime.timedelta(hours=i)
        date_data[constants.year_index, i, 0] = current_date.year
        date_data[constants.month_index, i, 0] = current_date.month
        date_data[constants.day_index, i, 0] = current_date.day
        date_data[constants.hour_index, i, 0] = current_date.hour
        date_data[constants.minute_index, i, 0] = current_date.minute
        # Simulate MATLAB datenum (days since year 1 + fractional day)
        date_data[constants.datenum_index, i, 0] = (
            current_date.toordinal() + 366 + current_date.hour / 24.0
        )

    # Create converted_data object
    data = converted_data()
    data.date_data = date_data
    data.n_date = n_date

    # Create metadata
    metadata = input_metadata()

    # Test without any date restrictions
    config = road_dust_initialise_time(data, metadata)

    assert config.min_time == 0
    assert config.max_time == n_date - 1
    assert config.max_time_inputdata == n_date - 1
    assert (
        abs(config.dt - 1.0) < 1e-6
    )  # 1 hour difference (with floating point tolerance)
    assert config.time_bad == 0
    assert config.min_time_save == 0
    assert config.max_time_save == n_date - 1


def test_road_dust_initialise_time_with_date_restrictions():
    """Test road_dust_initialise_time with start and end date restrictions."""
    # Create sample date data
    n_date = 24  # 24 hours of data
    n_roads = 1
    date_data = np.zeros((constants.num_date_index, n_date, n_roads))

    # Fill with sample dates (hourly data from 2023-03-15 00:00 to 2023-03-15 23:00)
    base_date = datetime.datetime(2023, 3, 15, 0, 0, 0)
    for i in range(n_date):
        current_date = base_date + datetime.timedelta(hours=i)
        date_data[constants.year_index, i, 0] = current_date.year
        date_data[constants.month_index, i, 0] = current_date.month
        date_data[constants.day_index, i, 0] = current_date.day
        date_data[constants.hour_index, i, 0] = current_date.hour
        date_data[constants.minute_index, i, 0] = current_date.minute
        date_data[constants.datenum_index, i, 0] = (
            current_date.toordinal() + 366 + current_date.hour / 24.0
        )

    # Create converted_data object
    data = converted_data()
    data.date_data = date_data
    data.n_date = n_date

    # Create metadata with date restrictions
    metadata = input_metadata()
    metadata.start_date_str = "2023-03-15 08:00:00"
    metadata.end_date_str = "2023-03-15 16:00:00"
    metadata.start_date_save_str = "2023-03-15 10:00:00"
    metadata.end_date_save_str = "2023-03-15 14:00:00"

    config = road_dust_initialise_time(data, metadata)

    assert config.min_time == 8
    assert config.max_time == 16
    assert config.min_time_save == 10
    assert config.max_time_save == 14
    assert config.time_bad == 0


def test_road_dust_initialise_time_invalid_dates():
    """Test road_dust_initialise_time with invalid dates."""
    # Create sample date data
    n_date = 5
    n_roads = 1
    date_data = np.zeros((constants.num_date_index, n_date, n_roads))

    # Fill with sample dates
    base_date = datetime.datetime(2023, 3, 15, 10, 0, 0)
    for i in range(n_date):
        current_date = base_date + datetime.timedelta(hours=i)
        date_data[constants.year_index, i, 0] = current_date.year
        date_data[constants.month_index, i, 0] = current_date.month
        date_data[constants.day_index, i, 0] = current_date.day
        date_data[constants.hour_index, i, 0] = current_date.hour
        date_data[constants.minute_index, i, 0] = current_date.minute
        date_data[constants.datenum_index, i, 0] = (
            current_date.toordinal() + 366 + current_date.hour / 24.0
        )

    # Create converted_data object
    data = converted_data()
    data.date_data = date_data
    data.n_date = n_date

    # Create metadata with invalid start date
    metadata = input_metadata()
    metadata.start_date_str = "2023-03-16 10:00:00"  # Date not in data

    config = road_dust_initialise_time(data, metadata)

    assert config.time_bad == 1


def test_road_dust_initialise_time_fortran_flag():
    """Test road_dust_initialise_time with fortran flag."""
    # Create sample date data
    n_date = 10
    n_roads = 1
    date_data = np.zeros((constants.num_date_index, n_date, n_roads))

    # Fill with sample dates
    base_date = datetime.datetime(2023, 3, 15, 10, 0, 0)
    for i in range(n_date):
        current_date = base_date + datetime.timedelta(hours=i)
        date_data[constants.year_index, i, 0] = current_date.year
        date_data[constants.month_index, i, 0] = current_date.month
        date_data[constants.day_index, i, 0] = current_date.day
        date_data[constants.hour_index, i, 0] = current_date.hour
        date_data[constants.minute_index, i, 0] = current_date.minute
        date_data[constants.datenum_index, i, 0] = (
            current_date.toordinal() + 366 + current_date.hour / 24.0
        )

    # Create converted_data object
    data = converted_data()
    data.date_data = date_data
    data.n_date = n_date

    # Create metadata with date restrictions
    metadata = input_metadata()
    metadata.start_date_str = "2023-03-15 12:00:00"
    metadata.end_date_str = "2023-03-15 15:00:00"

    # Test with fortran flag (should override date restrictions)
    config = road_dust_initialise_time(data, metadata, use_fortran_flag=True)

    assert config.min_time == 0
    assert config.max_time == n_date - 1
    assert config.time_bad == 0


def test_road_dust_initialise_time_subdate():
    """Test road_dust_initialise_time with subdates."""
    # Create sample date data
    n_date = 24  # 24 hours of data
    n_roads = 1
    date_data = np.zeros((constants.num_date_index, n_date, n_roads))

    # Fill with sample dates (hourly data from 2023-03-15 00:00 to 2023-03-15 23:00)
    base_date = datetime.datetime(2023, 3, 15, 0, 0, 0)
    for i in range(n_date):
        current_date = base_date + datetime.timedelta(hours=i)
        date_data[constants.year_index, i, 0] = current_date.year
        date_data[constants.month_index, i, 0] = current_date.month
        date_data[constants.day_index, i, 0] = current_date.day
        date_data[constants.hour_index, i, 0] = current_date.hour
        date_data[constants.minute_index, i, 0] = current_date.minute
        date_data[constants.datenum_index, i, 0] = (
            current_date.toordinal() + 366 + current_date.hour / 24.0
        )

    # Create converted_data object
    data = converted_data()
    data.date_data = date_data
    data.n_date = n_date

    # Create metadata with subdates
    metadata = input_metadata()
    metadata.n_save_subdate = 2
    metadata.start_subdate_save_str = ["2023-03-15 06:00:00", "2023-03-15 18:00:00"]
    metadata.end_subdate_save_str = ["2023-03-15 12:00:00", "2023-03-15 22:00:00"]

    config = road_dust_initialise_time(data, metadata)

    assert len(config.min_subtime_save) == 2
    assert len(config.max_subtime_save) == 2
    assert config.min_subtime_save[0] == 6
    assert config.max_subtime_save[0] == 12
    assert config.min_subtime_save[1] == 18
    assert config.max_subtime_save[1] == 22
    assert config.time_bad == 0


def test_road_dust_initialise_time_single_timestep():
    """Test road_dust_initialise_time with only one timestep."""
    # Create sample date data with single timestep
    n_date = 1
    n_roads = 1
    date_data = np.zeros((constants.num_date_index, n_date, n_roads))

    # Fill with single date
    date_data[constants.year_index, 0, 0] = 2023
    date_data[constants.month_index, 0, 0] = 3
    date_data[constants.day_index, 0, 0] = 15
    date_data[constants.hour_index, 0, 0] = 10
    date_data[constants.minute_index, 0, 0] = 0
    date_data[constants.datenum_index, 0, 0] = 738611.416667  # Example datenum

    # Create converted_data object
    data = converted_data()
    data.date_data = date_data
    data.n_date = n_date

    # Create metadata
    metadata = input_metadata()

    config = road_dust_initialise_time(data, metadata)

    assert config.min_time == 0
    assert config.max_time == 0
    assert config.dt == 1.0
    assert config.time_bad == 0
