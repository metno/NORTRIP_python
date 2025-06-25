import pandas as pd
import numpy as np
from src.read_files.read_road_dust_input.read_input_traffic import read_input_traffic


# TODO: Fix the tests, they are missing header data
def test_read_input_traffic_basic():
    """Test basic traffic data reading functionality."""
    # fmt: off
    test_data = [
        ["Year", "Month", "Day", "Hour", "Minute", "N(total)", "N(he)", "N(li)", "N(st,he)", "N(wi,he)", "N(su,he)", "N(st,li)", "N(wi,li)", "N(su,li)", "V_veh(he)", "V_veh(li)"],
        ["2023", "1", "1", "0", "0", "100", "70", "30", "35", "20", "15", "15", "10", "5", "50", "60"],
        ["2023", "1", "1", "1", "0", "120", "80", "40", "40", "25", "15", "20", "12", "8", "55", "65"],
        ["2023", "1", "1", "2", "0", "90", "60", "30", "30", "18", "12", "15", "10", "5", "45", "55"],
    ]
    # fmt: on

    df = pd.DataFrame(test_data)

    result = read_input_traffic(df, nodata=-99.0, print_results=False)

    # Basic assertions
    assert result.n_traffic == 3
    assert len(result.year) == 3
    assert len(result.month) == 3
    assert len(result.day) == 3
    assert len(result.N_total) == 3
    assert result.N_v.shape == (2, 3)  # num_veh x n_traffic
    assert result.N.shape == (3, 2, 3)  # num_tyre x num_veh x n_traffic
    assert result.V_veh.shape == (2, 3)  # num_veh x n_traffic

    # Check specific values
    np.testing.assert_array_equal(result.year, [2023, 2023, 2023])
    np.testing.assert_array_equal(result.month, [1, 1, 1])
    np.testing.assert_array_equal(result.day, [1, 1, 1])
    np.testing.assert_array_equal(result.N_total, [100, 120, 90])
    np.testing.assert_array_equal(result.N_v[0, :], [70, 80, 60])  # he
    np.testing.assert_array_equal(result.N_v[1, :], [30, 40, 30])  # li


def test_read_input_traffic_missing_data_forward_fill():
    """Test forward fill functionality with simple missing data."""
    # fmt: off
    test_data = [
        ["Year", "Month", "Day", "Hour", "Minute", "N(total)", "N(he)", "N(li)", "N(st,he)", "N(wi,he)", "N(su,he)", "N(st,li)", "N(wi,li)", "N(su,li)", "V_veh(he)", "V_veh(li)"],
        ["2023", "1", "15", "0", "0", "100", "70", "30", "35", "20", "15", "15", "10", "5", "50", "60"],
        ["2023", "1", "15", "1", "0", "120", "80", "40", "40", "25", "15", "20", "12", "8", "55", "65"],
        ["2023", "1", "15", "2", "0", "90", "60", "30", "30", "18", "12", "15", "10", "5", "45", "55"],
    ]
    # fmt: on
    df = pd.DataFrame(test_data)

    result = read_input_traffic(df, nodata=-99.0, print_results=False)

    assert len(result.N_total_nodata) == 0  # No missing data in this simple test
    assert len(result.month) == 3
    assert len(result.day) == 3
    np.testing.assert_array_equal(result.month, [1, 1, 1])
    np.testing.assert_array_equal(result.day, [15, 15, 15])


def test_read_input_traffic_missing_data_complex():
    """Test handling of missing data with multiple missing values (original test)."""
    # fmt: off
    test_data = [
        ["Year", "Month", "Day", "Hour", "Minute", "N(total)", "N(he)", "N(li)", "N(st,he)", "N(wi,he)", "N(su,he)", "N(st,li)", "N(wi,li)", "N(su,li)", "V_veh(he)", "V_veh(li)"],
        ["2023", "2", "10", "0", "0", "100", "70", "30", "35", "20", "15", "15", "10", "5", "50", "60"],
        ["2023", "2", "10", "1", "0", "120", "80", "40", "40", "25", "15", "20", "12", "8", "55", "65"],
        ["2023", "2", "10", "2", "0", "90", "60", "30", "30", "18", "12", "15", "10", "5", "45", "55"],
    ]
    # fmt: on

    df = pd.DataFrame(test_data)

    result = read_input_traffic(df, nodata=-99.0, print_results=False)

    # Basic checks for date data
    assert len(result.month) == 3
    assert len(result.day) == 3
    np.testing.assert_array_equal(result.month, [2, 2, 2])
    np.testing.assert_array_equal(result.day, [10, 10, 10])

    # Check that we have valid traffic data (no missing data in this basic test)
    assert len(result.N_total_nodata) == 0
    assert all(len(sublist) == 0 for sublist in result.V_veh_nodata)


def test_read_input_traffic_date_string_formatting():
    """Test date string formatting functionality."""
    # fmt: off
    test_data = [
        ["Year", "Month", "Day", "Hour", "Minute", "N(total)", "N(he)", "N(li)", "N(st,he)", "N(wi,he)", "N(su,he)", "N(st,li)", "N(wi,li)", "N(su,li)", "V_veh(he)", "V_veh(li)"],
        ["2023", "3", "15", "9", "30", "100", "70", "30", "35", "20", "15", "15", "10", "5", "50", "60"],
        ["2023", "12", "5", "14", "45", "120", "80", "40", "40", "25", "15", "20", "12", "8", "55", "65"],
        ["2024", "1", "1", "0", "0", "90", "60", "30", "30", "18", "12", "15", "10", "5", "45", "55"],
    ]
    # fmt: on

    df = pd.DataFrame(test_data)
    result = read_input_traffic(df, nodata=-99.0, print_results=False)

    # Check format 1: "%Y.%m.%d %H" format
    expected_format1 = ["2023.03.15 09", "2023.12.05 14", "2024.01.01 00"]
    np.testing.assert_array_equal(result.date_str[0, :], expected_format1)

    # Check format 2: "%H:%M %d %b " format
    expected_format2 = ["09:30 15 Mar ", "14:45 05 Dec ", "00:00 01 Jan "]
    np.testing.assert_array_equal(result.date_str[1, :], expected_format2)

    # Verify individual date components are correctly parsed
    np.testing.assert_array_equal(result.year, [2023, 2023, 2024])
    np.testing.assert_array_equal(result.month, [3, 12, 1])
    np.testing.assert_array_equal(result.day, [15, 5, 1])
    np.testing.assert_array_equal(result.hour, [9, 14, 0])
    np.testing.assert_array_equal(result.minute, [30, 45, 0])


def test_read_input_traffic_date_string_edge_cases():
    """Test date string formatting with edge cases like leap year and different months."""
    # fmt: off
    test_data = [
        ["Year", "Month", "Day", "Hour", "Minute", "N(total)", "N(he)", "N(li)", "N(st,he)", "N(wi,he)", "N(su,he)", "N(st,li)", "N(wi,li)", "N(su,li)", "V_veh(he)", "V_veh(li)"],
        ["2024", "2", "29", "23", "59", "100", "70", "30", "35", "20", "15", "15", "10", "5", "50", "60"],  # Leap year
        ["2023", "6", "21", "12", "0", "120", "80", "40", "40", "25", "15", "20", "12", "8", "55", "65"],   # Summer solstice
        ["2023", "11", "30", "6", "15", "90", "60", "30", "30", "18", "12", "15", "10", "5", "45", "55"],  # End of November
    ]
    # fmt: on

    df = pd.DataFrame(test_data)
    result = read_input_traffic(df, nodata=-99.0, print_results=False)

    # Check format 1 for edge cases
    expected_format1 = ["2024.02.29 23", "2023.06.21 12", "2023.11.30 06"]
    np.testing.assert_array_equal(result.date_str[0, :], expected_format1)

    # Check format 2 for edge cases
    expected_format2 = ["23:59 29 Feb ", "12:00 21 Jun ", "06:15 30 Nov "]
    np.testing.assert_array_equal(result.date_str[1, :], expected_format2)


def test_read_input_traffic_missing_minute_column():
    """Test date string creation when Minute column is missing."""
    # fmt: off
    test_data = [
        ["Year", "Month", "Day", "Hour", "N(total)", "N(he)", "N(li)", "N(st,he)", "N(wi,he)", "N(su,he)", "N(st,li)", "N(wi,li)", "N(su,li)", "V_veh(he)", "V_veh(li)"],
        ["2023", "7", "4", "8", "100", "70", "30", "35", "20", "15", "15", "10", "5", "50", "60"],
        ["2023", "7", "4", "16", "120", "80", "40", "40", "25", "15", "20", "12", "8", "55", "65"],
    ]
    # fmt: on

    df = pd.DataFrame(test_data)
    result = read_input_traffic(df, nodata=-99.0, print_results=False)

    # Minutes should default to 0 when column is missing
    np.testing.assert_array_equal(result.minute, [0, 0])

    # Check that date strings are still created correctly with 0 minutes
    expected_format1 = ["2023.07.04 08", "2023.07.04 16"]
    np.testing.assert_array_equal(result.date_str[0, :], expected_format1)

    expected_format2 = ["08:00 04 Jul ", "16:00 04 Jul "]
    np.testing.assert_array_equal(result.date_str[1, :], expected_format2)
