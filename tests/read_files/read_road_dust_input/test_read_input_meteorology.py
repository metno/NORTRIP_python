import pandas as pd
import numpy as np
import pytest
from src.read_files.read_road_dust_input.read_input_meteorology import (
    read_input_meteorology,
)


def test_read_input_meteorology_basic():
    """Test basic meteorology reading functionality."""
    # fmt: off
    test_data = [
        ["T2m", "FF", "Rain", "Snow", "RH"],
        ["20.5", "5.2", "0.0", "0.0", "65.0"],
        ["21.0", "6.1", "2.5", "0.0", "70.2"],
        ["19.8", "4.8", "0.0", "1.2", "58.9"],
    ]
    # fmt: on

    df = pd.DataFrame(test_data)
    result = read_input_meteorology(df)

    # Check basic fields
    assert result.n_meteo == 3
    assert len(result.T_a) == 3
    assert len(result.FF) == 3
    assert len(result.Rain) == 3
    assert len(result.Snow) == 3
    assert result.RH_available == 1


def test_read_input_meteorology_missing_required_field():
    """Test behavior when required field is missing."""
    # fmt: off
    test_data = [
        ["FF", "Rain", "Snow"],
        ["5.2", "0.0", "0.0"],
        ["6.1", "2.5", "0.0"],
        ["4.8", "0.0", "1.2"],
    ]
    # fmt: on

    df = pd.DataFrame(test_data)
    result = read_input_meteorology(df)

    # Should return early with empty data
    assert result.n_meteo == 3  # Data count is set before validation
    assert len(result.T_a) == 0  # But arrays should be empty due to early return


def test_read_input_meteorology_optional_fields():
    """Test handling of optional fields."""
    # fmt: off
    test_data = [
        ["T2m", "FF", "Rain", "Snow", "DD", "Global radiation", "Pressure"],
        ["20.5", "5.2", "0.0", "0.0", "180.0", "200.0", "101325"],
        ["21.0", "6.1", "2.5", "0.0", "190.0", "300.0", "101300"],
        ["19.8", "4.8", "0.0", "1.2", "170.0", "150.0", "101350"],
    ]
    # fmt: on

    df = pd.DataFrame(test_data)
    result = read_input_meteorology(df)

    # Check that optional fields are detected
    assert result.DD_available == 1
    assert result.short_rad_in_available == 1
    assert result.pressure_obs_available == 1
    assert len(result.DD) == 3
    assert len(result.short_rad_in) == 3
    assert len(result.Pressure_a) == 3


def test_read_input_meteorology_missing_data_handling():
    """Test handling of missing data (nodata values)."""
    # fmt: off
    test_data = [
        ["T2m", "FF", "Rain", "Snow", "RH"],
        ["20.5", "5.2", "0.0", "0.0", "65.0"],
        ["-99", "6.1", "2.5", "0.0", "-99"],
        ["19.8", "-99", "0.0", "1.2", "58.9"],
    ]
    # fmt: on

    df = pd.DataFrame(test_data)
    result = read_input_meteorology(df, nodata=-99.0)

    # Check that missing data indices are recorded
    assert len(result.T_a_nodata) == 1
    assert result.T_a_nodata[0] == 1  # Second row (0-indexed)
    assert len(result.FF_nodata) == 1
    assert result.FF_nodata[0] == 2  # Third row (0-indexed)

    # Check that forward filling occurred
    assert result.T_a[1] == 20.5  # Should be filled with previous value
    assert result.FF[2] == 6.1  # Should be filled with previous value


def test_read_input_meteorology_wind_speed_correction():
    """Test wind speed correction factor."""
    # fmt: off
    test_data = [
        ["T2m", "FF", "Rain", "Snow"],
        ["20.5", "10.0", "0.0", "0.0"],
        ["21.0", "12.0", "2.5", "0.0"],
        ["19.8", "8.0", "0.0", "1.2"],
    ]
    # fmt: on

    df = pd.DataFrame(test_data)
    correction_factor = 1.5
    result = read_input_meteorology(df, wind_speed_correction=correction_factor)

    # Check that wind speed is corrected
    expected_ff = np.array([10.0, 12.0, 8.0]) * correction_factor
    np.testing.assert_array_almost_equal(result.FF, expected_ff)


def test_read_input_meteorology_road_wetness_mm_detection():
    """Test detection of road wetness units in mm."""
    # fmt: off
    test_data = [
        ["T2m", "FF", "Rain", "Snow", "Road wetness (mm)"],
        ["20.5", "5.2", "0.0", "0.0", "0.5"],
        ["21.0", "6.1", "2.5", "0.0", "1.2"],
        ["19.8", "4.8", "0.0", "1.2", "0.0"],
    ]
    # fmt: on

    df = pd.DataFrame(test_data)
    result = read_input_meteorology(df)

    # Check that mm units are detected
    assert result.road_wetness_obs_available == 1
    assert result.road_wetness_obs_in_mm == 1
    assert result.max_road_wetness_obs == 1.2
    assert result.min_road_wetness_obs == 0.0
    assert result.mean_road_wetness_obs == pytest.approx(0.5667, rel=1e-3)


def test_read_input_meteorology_rh_dewpoint_conversion():
    """Test RH to dewpoint conversion when dewpoint is missing."""
    # fmt: off
    test_data = [
        ["T2m", "FF", "Rain", "Snow", "RH"],
        ["20.0", "5.2", "0.0", "0.0", "60.0"],
        ["25.0", "6.1", "2.5", "0.0", "70.0"],
        ["15.0", "4.8", "0.0", "1.2", "80.0"],
    ]
    # fmt: on

    df = pd.DataFrame(test_data)
    result = read_input_meteorology(df)

    # Check that dewpoint was calculated
    assert result.RH_available == 1
    assert result.T_dewpoint_available == 0  # Was not in input
    assert len(result.T_dewpoint) == 3  # But was calculated

    # Check that dewpoint values are reasonable (should be less than air temp)
    assert np.all(result.T_dewpoint < result.T_a)


def test_read_input_meteorology_negative_value_removal():
    """Test removal of negative values in Rain and Snow."""
    # fmt: off
    test_data = [
        ["T2m", "FF", "Rain", "Snow", "RH"],
        ["20.5", "5.2", "-1.0", "0.0", "-5.0"],
        ["21.0", "6.1", "2.5", "-0.5", "70.0"],
        ["19.8", "4.8", "0.0", "1.2", "80.0"],
    ]
    # fmt: on

    df = pd.DataFrame(test_data)
    result = read_input_meteorology(df)

    # Check that negative values are set to 0
    assert result.Rain[0] == 0.0
    assert result.Snow[1] == 0.0
    assert result.RH[0] == 0.0


def test_read_input_meteorology_empty_data():
    """Test handling of empty data."""
    # fmt: off
    test_data = [
        ["T2m", "FF", "Rain", "Snow"],
    ]
    # fmt: on

    df = pd.DataFrame(test_data)
    result = read_input_meteorology(df)

    # Should handle empty data gracefully
    assert result.n_meteo == 0
