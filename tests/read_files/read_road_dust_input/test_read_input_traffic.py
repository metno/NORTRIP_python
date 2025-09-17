import pandas as pd
import numpy as np
from src.read_files.read_road_dust_input.read_input_traffic import read_input_traffic


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

    result = read_input_traffic(df, nodata=-99.0)

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

    result = read_input_traffic(df, nodata=-99.0)

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

    result = read_input_traffic(df, nodata=-99.0)

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
    result = read_input_traffic(df, nodata=-99.0)

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
    result = read_input_traffic(df, nodata=-99.0)

    # Minutes should default to 0 when column is missing
    np.testing.assert_array_equal(result.minute, [0, 0])

    # Check that date strings are still created correctly with 0 minutes
    expected_format1 = ["2023.07.04 08", "2023.07.04 16"]
    np.testing.assert_array_equal(result.date_str[0, :], expected_format1)

    expected_format2 = ["08:00 04 Jul ", "16:00 04 Jul "]
    np.testing.assert_array_equal(result.date_str[1, :], expected_format2)


def test_read_input_traffic_nodata_handling():
    """Test handling of nodata values (-99) in traffic data."""
    # fmt: off
    test_data = [
        ["Year", "Month", "Day", "Hour", "Minute", "N(total)", "N(he)", "N(li)", "N(st,he)", "N(wi,he)", "N(su,he)", "N(st,li)", "N(wi,li)", "N(su,li)", "V_veh(he)", "V_veh(li)"],
        ["2023", "1", "1", "0", "0", "100", "70", "30", "35", "20", "15", "15", "10", "5", "50", "60"],
        ["2023", "1", "1", "1", "0", "-99", "80", "40", "40", "25", "15", "20", "12", "8", "55", "65"],
        ["2023", "1", "1", "2", "0", "90", "-99", "30", "30", "18", "12", "15", "10", "5", "45", "55"],
        ["2023", "1", "1", "3", "0", "110", "75", "-99", "35", "22", "18", "-99", "-99", "-99", "50", "60"],
        ["2023", "1", "1", "4", "0", "95", "65", "30", "32", "20", "13", "14", "11", "5", "-99", "0"],
    ]
    # fmt: on

    df = pd.DataFrame(test_data)
    result = read_input_traffic(df, nodata=-99.0)

    # Basic structure checks
    assert result.n_traffic == 5
    assert len(result.year) == 5

    # Check nodata tracking
    assert len(result.N_total_nodata) == 1
    assert result.N_total_nodata[0] == 1

    assert len(result.N_v_nodata) == 2
    assert len(result.N_v_nodata[0]) == 1
    assert result.N_v_nodata[0][0] == 2
    assert len(result.N_v_nodata[1]) == 1
    assert result.N_v_nodata[1][0] == 3

    assert len(result.V_veh_nodata) == 2
    assert len(result.V_veh_nodata[0]) == 1
    assert result.V_veh_nodata[0][0] == 4
    assert len(result.V_veh_nodata[1]) == 1
    assert result.V_veh_nodata[1][0] == 4

    # Check that missing data is tracked correctly
    assert 2 in result.N_v_nodata[0]
    assert 3 in result.N_v_nodata[1]

    # Check vehicle speed forward fill
    assert result.V_veh[0, 4] == 50.0
    assert result.V_veh[1, 4] == 60.0

    # Verify good data is preserved
    np.testing.assert_array_equal(result.year, [2023, 2023, 2023, 2023, 2023])
    assert result.N_total[0] == 100.0
    assert result.N_v[0, 0] == 70.0
    assert result.N_v[1, 0] == 30.0

    # Check that -99 values were processed
    assert result.N_total[1] != -99.0

    # Check that zeros in speed data are treated as missing
    assert 4 in result.V_veh_nodata[1]
