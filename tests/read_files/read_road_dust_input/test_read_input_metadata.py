import pandas as pd
from src.read_files.read_road_dust_input.read_input_metadata import read_input_metadata


def test_read_input_metadata_basic():
    """Test basic metadata reading functionality with complete data."""
    test_data = [
        ["Parameter", "Value"],
        ["Road width", "15.0"],
        ["Latitude", "59.17"],
        ["Longitude", "18.30"],
        ["Height obs wind", "2.0"],
        ["Height obs temperature", "2.0"],
        ["Time difference", "-1.0"],
        ["Elevation", "0.0"],
        ["Surface albedo", "0.15"],
        ["Surface pressure", "1000.0"],
        ["Missing data", "-99.0"],
        ["Number of lanes", "4"],
        ["Width of lane", "3.5"],
        ["Street canyon width", "23.0"],
        ["Street canyon height north", "25.0"],
        ["Street canyon height south", "24.0"],
        ["Street orientation", "76.0"],
        ["Start date", "2023-01-01"],
        ["End date", "2023-12-31"],
        ["Exhaust EF (he)", "0.5"],
        ["Exhaust EF (li)", "0.3"],
        ["NOX EF (he)", "2.5"],
        ["NOX EF (li)", "1.8"],
    ]

    metadata_df = pd.DataFrame(test_data)
    result = read_input_metadata(metadata_df)

    # Check basic numeric fields
    assert result.b_road == 15.0
    assert result.LAT == 59.17
    assert result.LON == 18.30
    assert result.z_FF == 2.0
    assert result.z_T == 2.0
    assert result.DIFUTC_H == -1.0
    assert result.Z_SURF == 0.0
    assert result.albedo_road == 0.15
    assert result.Pressure == 1000.0
    assert result.nodata == -99.0
    assert result.n_lanes == 4
    assert result.b_lane == 3.5
    assert result.b_canyon == 23.0
    assert result.ang_road == 76.0

    # Check calculated field
    assert result.b_road_lanes == 4 * 3.5

    # Check canyon heights
    assert result.h_canyon == [25.0, 24.0]

    # Check dates
    assert result.start_date_str == "2023-01-01 00:00:00"
    assert result.end_date_str == "2023-12-31 00:00:00"

    # Check emission factors
    assert result.exhaust_EF == [0.5, 0.3]
    assert result.exhaust_EF_available == 1
    assert result.NOX_EF == [2.5, 1.8]
    assert result.NOX_EF_available == 1


def test_read_input_metadata_minimal_data():
    """Test with minimal required data only."""
    test_data = [
        ["Parameter", "Value"],
        ["Road width", "10.0"],
        ["Latitude", "60.0"],
        ["Longitude", "10.0"],
        ["Height obs wind", "3.0"],
        ["Height obs temperature", "2.5"],
        ["Time difference", "1.0"],
    ]

    metadata_df = pd.DataFrame(test_data)
    result = read_input_metadata(metadata_df)

    # Check that specified values are set
    assert result.b_road == 10.0
    assert result.LAT == 60.0
    assert result.LON == 10.0
    assert result.z_FF == 3.0
    assert result.z_T == 2.5
    assert result.DIFUTC_H == 1.0

    # Check that missing values use defaults
    assert result.Z_SURF == 0.0  # default
    assert result.albedo_road == 0.3  # default
    assert result.Pressure == 1000.0  # default
    assert result.nodata == -99.0  # default
    assert result.n_lanes == 2  # default
    assert result.b_lane == 3.5  # default

    # Check that b_canyon defaults to b_road when missing
    assert result.b_canyon == 10.0  # should equal b_road

    # Check default emission factors
    assert result.exhaust_EF == [0.0, 0.0]
    assert result.exhaust_EF_available == 0
    assert result.NOX_EF == [0.0, 0.0]
    assert result.NOX_EF_available == 0


def test_read_input_metadata_canyon_height_logic():
    """Test the logic for Street canyon height (north/south vs single value)."""
    # Test with north/south values
    test_data_north_south = [
        ["Parameter", "Value"],
        ["Road width", "15.0"],
        ["Street canyon height north", "25.0"],
        ["Street canyon height south", "24.0"],
        ["Latitude", "60.0"],
        ["Longitude", "10.0"],
        ["Height obs wind", "3.0"],
        ["Height obs temperature", "2.5"],
        ["Time difference", "1.0"],
    ]

    metadata_df = pd.DataFrame(test_data_north_south)
    result = read_input_metadata(metadata_df)
    assert result.h_canyon == [25.0, 24.0]

    # Test with single value
    test_data_single = [
        ["Parameter", "Value"],
        ["Road width", "15.0"],
        ["Street canyon height", "30.0"],
        ["Latitude", "60.0"],
        ["Longitude", "10.0"],
        ["Height obs wind", "3.0"],
        ["Height obs temperature", "2.5"],
        ["Time difference", "1.0"],
    ]

    metadata_df = pd.DataFrame(test_data_single)
    result = read_input_metadata(metadata_df)
    assert result.h_canyon == [30.0, 30.0]

    # Test with no canyon height values (should default to [0.0, 0.0])
    test_data_none = [
        ["Parameter", "Value"],
        ["Road width", "15.0"],
        ["Latitude", "60.0"],
        ["Longitude", "10.0"],
        ["Height obs wind", "3.0"],
        ["Height obs temperature", "2.5"],
        ["Time difference", "1.0"],
    ]

    metadata_df = pd.DataFrame(test_data_none)
    result = read_input_metadata(metadata_df)
    assert result.h_canyon == [0.0, 0.0]


def test_read_input_metadata_date_formatting():
    """Test date string formatting (adding time if missing)."""
    test_data = [
        ["Parameter", "Value"],
        ["Road width", "15.0"],
        ["Latitude", "60.0"],
        ["Longitude", "10.0"],
        ["Height obs wind", "3.0"],
        ["Height obs temperature", "2.5"],
        ["Time difference", "1.0"],
        ["Start date", "2023-01-01"],  # Should get " 00:00:00" appended
        ["End date", "2023-12-31 23:59:59"],  # Should remain unchanged
        ["Start save date", "2023-06-01"],  # Should get " 00:00:00" appended
        ["End save date", "2023-06-30 12:00:00"],  # Should remain unchanged
    ]

    metadata_df = pd.DataFrame(test_data)
    result = read_input_metadata(metadata_df)

    assert result.start_date_str == "2023-01-01 00:00:00"
    assert result.end_date_str == "2023-12-31 23:59:59"
    assert result.start_date_save_str == "2023-06-01 00:00:00"
    assert result.end_date_save_str == "2023-06-30 12:00:00"


def test_read_input_metadata_emission_factors():
    """Test emission factor handling and availability flags."""
    # Test with emission factors
    test_data_with_ef = [
        ["Parameter", "Value"],
        ["Road width", "15.0"],
        ["Latitude", "60.0"],
        ["Longitude", "10.0"],
        ["Height obs wind", "3.0"],
        ["Height obs temperature", "2.5"],
        ["Time difference", "1.0"],
        ["Exhaust EF (he)", "1.5"],
        ["Exhaust EF (li)", "0.8"],
        ["NOX EF (he)", "3.2"],
        ["NOX EF (li)", "2.1"],
    ]

    metadata_df = pd.DataFrame(test_data_with_ef)
    result = read_input_metadata(metadata_df)

    assert result.exhaust_EF == [1.5, 0.8]
    assert result.exhaust_EF_available == 1  # Should be 1 since sum is not zero
    assert result.NOX_EF == [3.2, 2.1]
    assert result.NOX_EF_available == 1

    # Test without emission factors (should default to [0.0, 0.0] and available=0)
    test_data_without_ef = [
        ["Parameter", "Value"],
        ["Road width", "15.0"],
        ["Latitude", "60.0"],
        ["Longitude", "10.0"],
        ["Height obs wind", "3.0"],
        ["Height obs temperature", "2.5"],
        ["Time difference", "1.0"],
    ]

    metadata_df = pd.DataFrame(test_data_without_ef)
    result = read_input_metadata(metadata_df)

    assert result.exhaust_EF == [0.0, 0.0]
    assert result.exhaust_EF_available == 0
    assert result.NOX_EF == [0.0, 0.0]
    assert result.NOX_EF_available == 0
