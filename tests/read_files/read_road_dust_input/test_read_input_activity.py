import pandas as pd
import numpy as np
from read_files.read_road_dust_input.read_input_activity import read_input_activity
import constants


def test_read_input_activity_basic():
    """Test basic activity data reading."""
    # fmt: off
    test_data = [
        ["Year", "Month", "Day", "Hour", "Minute", "M_sanding", "M_salting(na)", "M_salting(mg)", "Wetting", "Ploughing_road", "Cleaning_road", "Fugitive"],
        [2023, 1, 15, 8, 0, 10.5, 20.0, 5.0, 2.0, 1.0, 0.5, 1.5],
        [2023, 1, 16, 10, 30, 0.0, 15.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [2023, 1, 17, 12, 45, 5.0, 0.0, 8.0, 1.0, 1.0, 0.0, 2.0],
    ]
    # fmt: on

    activity_df = pd.DataFrame(test_data)
    result = read_input_activity(activity_df)

    assert result.n_act == 3
    assert len(result.year) == 3

    # Check date/time data
    np.testing.assert_array_equal(result.year, [2023, 2023, 2023])
    np.testing.assert_array_equal(result.month, [1, 1, 1])
    np.testing.assert_array_equal(result.day, [15, 16, 17])
    np.testing.assert_array_equal(result.hour, [8, 10, 12])
    np.testing.assert_array_equal(result.minute, [0, 30, 45])

    # Check activity data
    np.testing.assert_array_equal(result.M_sanding, [10.5, 0.0, 5.0])
    np.testing.assert_array_equal(result.M_salting[0, :], [20.0, 15.0, 0.0])  # na salt
    np.testing.assert_array_equal(result.M_salting[1, :], [5.0, 0.0, 8.0])  # mg salt
    np.testing.assert_array_equal(result.g_road_wetting, [2.0, 0.0, 1.0])
    np.testing.assert_array_equal(result.t_ploughing, [1.0, 0.0, 1.0])
    np.testing.assert_array_equal(result.t_cleaning, [0.5, 1.0, 0.0])
    np.testing.assert_array_equal(result.M_fugitive, [1.5, 0.0, 2.0])

    # Check salt type information
    np.testing.assert_array_equal(result.salt_type, [constants.na, constants.mg])
    assert result.second_salt_type == constants.mg
    assert result.second_salt_available == 1
    assert result.salt2_str == "mg"
    assert result.g_road_wetting_available == 1


def test_read_input_activity_no_minute_column():
    """Test reading activity data without minute column."""
    # fmt: off
    test_data = [
        ["Year", "Month", "Day", "Hour", "M_sanding", "Ploughing_road", "Cleaning_road"],
        [2023, 1, 15, 8, 10.5, 1.0, 0.5],
        [2023, 1, 16, 10, 0.0, 0.0, 1.0],
    ]
    # fmt: on

    activity_df = pd.DataFrame(test_data)
    result = read_input_activity(activity_df)

    assert result.n_act == 2
    np.testing.assert_array_equal(result.minute, [0, 0])  # Should default to zeros


def test_read_input_activity_salt_priority_cma():
    """Test salt type priority: should use cma when mg not available."""
    # fmt: off
    test_data = [
        ["Year", "Month", "Day", "Hour", "M_salting(na)", "M_salting(cma)", "Ploughing_road", "Cleaning_road"],
        [2023, 1, 15, 8, 20.0, 12.0, 1.0, 0.5],
    ]
    # fmt: on

    activity_df = pd.DataFrame(test_data)
    result = read_input_activity(activity_df)

    assert result.second_salt_type == constants.cma
    assert result.second_salt_available == 1
    assert result.salt2_str == "cma"
    np.testing.assert_array_equal(result.salt_type, [constants.na, constants.cma])
    np.testing.assert_array_equal(result.M_salting[1, :], [12.0])


def test_read_input_activity_salt_priority_ca():
    """Test salt type priority: should use ca when mg and cma not available."""
    # fmt: off
    test_data = [
        ["Year", "Month", "Day", "Hour", "M_salting(na)", "M_salting(ca)", "Ploughing_road", "Cleaning_road"],
        [2023, 1, 15, 8, 20.0, 8.0, 1.0, 0.5],
    ]
    # fmt: on

    activity_df = pd.DataFrame(test_data)
    result = read_input_activity(activity_df)

    assert result.second_salt_type == constants.ca
    assert result.second_salt_available == 1
    assert result.salt2_str == "ca"
    np.testing.assert_array_equal(result.salt_type, [constants.na, constants.ca])
    np.testing.assert_array_equal(result.M_salting[1, :], [8.0])


def test_read_input_activity_salt_priority_mg_over_others():
    """Test salt type priority: mg should take precedence over cma and ca."""
    # fmt: off
    test_data = [
        ["Year", "Month", "Day", "Hour", "M_salting(na)", "M_salting(mg)", "M_salting(cma)", "M_salting(ca)", "Ploughing_road", "Cleaning_road"],
        [2023, 1, 15, 8, 20.0, 5.0, 12.0, 8.0, 1.0, 0.5],
    ]
    # fmt: on

    activity_df = pd.DataFrame(test_data)
    result = read_input_activity(activity_df)

    assert result.second_salt_type == constants.mg
    assert result.second_salt_available == 1
    assert result.salt2_str == "mg"
    np.testing.assert_array_equal(
        result.M_salting[1, :], [5.0]
    )  # Should use mg, not cma or ca


def test_read_input_activity_no_secondary_salt():
    """Test when no secondary salt data is available."""
    # fmt: off
    test_data = [
        ["Year", "Month", "Day", "Hour", "M_salting(na)", "Ploughing_road", "Cleaning_road"],
        [2023, 1, 15, 8, 20.0, 1.0, 0.5],
    ]
    # fmt: on

    activity_df = pd.DataFrame(test_data)
    result = read_input_activity(activity_df)

    assert result.second_salt_type == constants.mg  # Default
    assert result.second_salt_available == 0
    assert result.salt2_str == "mg"
    np.testing.assert_array_equal(result.M_salting[1, :], [0.0])  # Should be zero


def test_read_input_activity_missing_optional_columns():
    """Test reading with missing optional columns."""
    # fmt: off
    test_data = [
        ["Year", "Month", "Day", "Hour", "Ploughing_road", "Cleaning_road"],
        [2023, 1, 15, 8, 1.0, 0.5],
    ]
    # fmt: on

    activity_df = pd.DataFrame(test_data)
    result = read_input_activity(activity_df)

    assert result.n_act == 1
    # Check that optional fields default to zero
    np.testing.assert_array_equal(result.M_sanding, [0.0])
    np.testing.assert_array_equal(result.M_salting[0, :], [0.0])  # na salt
    np.testing.assert_array_equal(result.M_salting[1, :], [0.0])  # secondary salt
    np.testing.assert_array_equal(result.g_road_wetting, [0.0])
    np.testing.assert_array_equal(result.M_fugitive, [0.0])
    assert result.g_road_wetting_available == 0


def test_read_input_activity_data_redistribution():
    """Test data redistribution to match traffic timeline."""
    # fmt: off
    test_activity_data = [
        ["Year", "Month", "Day", "Hour", "Minute", "M_sanding", "M_salting(na)", "Ploughing_road", "Cleaning_road"],
        [2023, 1, 15, 8, 0, 10.0, 20.0, 1.0, 0.5],
        [2023, 1, 16, 10, 0, 5.0, 15.0, 0.0, 1.0],
    ]
    # fmt: on

    activity_df = pd.DataFrame(test_activity_data)

    # Traffic timeline with 4 records (including the activity times)
    traffic_year = np.array([2023, 2023, 2023, 2023])
    traffic_month = np.array([1, 1, 1, 1])
    traffic_day = np.array([15, 15, 16, 16])
    traffic_hour = np.array([8, 9, 10, 11])
    traffic_minute = np.array([0, 0, 0, 0])

    result = read_input_activity(
        activity_df,
        traffic_year,
        traffic_month,
        traffic_day,
        traffic_hour,
        traffic_minute,
    )

    # Should have 4 records to match traffic timeline
    assert len(result.M_sanding) == 4
    assert len(result.M_salting[0, :]) == 4

    # Check data redistribution
    np.testing.assert_array_equal(result.M_sanding, [10.0, 0.0, 5.0, 0.0])
    np.testing.assert_array_equal(result.M_salting[0, :], [20.0, 0.0, 15.0, 0.0])
    np.testing.assert_array_equal(result.t_ploughing, [1.0, 0.0, 0.0, 0.0])
    np.testing.assert_array_equal(result.t_cleaning, [0.5, 0.0, 1.0, 0.0])


def test_read_input_activity_data_accumulation():
    """Test accumulation of multiple activity events at same time."""
    # fmt: off
    test_activity_data = [
        ["Year", "Month", "Day", "Hour", "Minute", "M_sanding", "M_salting(na)", "Ploughing_road"],
        [2023, 1, 15, 8, 0, 10.0, 20.0, 1.0],
        [2023, 1, 15, 8, 0, 5.0, 15.0, 0.5],  # Same time as first
    ]
    # fmt: on

    activity_df = pd.DataFrame(test_activity_data)

    # Traffic timeline with single matching time
    traffic_year = np.array([2023])
    traffic_month = np.array([1])
    traffic_day = np.array([15])
    traffic_hour = np.array([8])
    traffic_minute = np.array([0])

    result = read_input_activity(
        activity_df,
        traffic_year,
        traffic_month,
        traffic_day,
        traffic_hour,
        traffic_minute,
    )

    # Should accumulate values at the same time
    assert len(result.M_sanding) == 1
    np.testing.assert_array_equal(result.M_sanding, [15.0])  # 10.0 + 5.0
    np.testing.assert_array_equal(result.M_salting[0, :], [35.0])  # 20.0 + 15.0
    np.testing.assert_array_equal(result.t_ploughing, [1.5])  # 1.0 + 0.5


def test_read_input_activity_empty_dataframe():
    """Test handling of empty activity DataFrame."""
    empty_df = pd.DataFrame()

    # Provide traffic timeline
    traffic_year = np.array([2023, 2023])
    traffic_month = np.array([1, 1])
    traffic_day = np.array([15, 16])
    traffic_hour = np.array([8, 9])
    traffic_minute = np.array([0, 0])

    result = read_input_activity(
        empty_df, traffic_year, traffic_month, traffic_day, traffic_hour, traffic_minute
    )

    # Should initialize with zeros for traffic timeline length
    assert len(result.M_sanding) == 2
    np.testing.assert_array_equal(result.M_sanding, [0.0, 0.0])
    np.testing.assert_array_equal(result.M_salting[0, :], [0.0, 0.0])
    np.testing.assert_array_equal(result.M_salting[1, :], [0.0, 0.0])
    assert result.salt_type[0] == constants.na
    assert result.salt_type[1] == constants.mg
    assert result.salt2_str == "mg"


def test_read_input_activity_no_matching_traffic_times():
    """Test when activity times don't match any traffic times."""
    # fmt: off
    test_activity_data = [
        ["Year", "Month", "Day", "Hour", "M_sanding", "Ploughing_road", "Cleaning_road"],
        [2023, 1, 15, 8, 10.0, 1.0, 0.5],
    ]
    # fmt: on

    activity_df = pd.DataFrame(test_activity_data)

    # Traffic timeline with different times
    traffic_year = np.array([2023, 2023])
    traffic_month = np.array([1, 1])
    traffic_day = np.array([16, 17])  # Different days
    traffic_hour = np.array([9, 10])
    traffic_minute = np.array([0, 0])

    result = read_input_activity(
        activity_df,
        traffic_year,
        traffic_month,
        traffic_day,
        traffic_hour,
        traffic_minute,
    )

    # Should have zeros since no times match
    assert len(result.M_sanding) == 2
    np.testing.assert_array_equal(result.M_sanding, [0.0, 0.0])
    np.testing.assert_array_equal(result.t_ploughing, [0.0, 0.0])
    np.testing.assert_array_equal(result.t_cleaning, [0.0, 0.0])
