import pandas as pd
import numpy as np
from read_files.read_road_dust_input.read_input_meteorology import (
    read_input_meteorology,
)


def test_read_input_meteorology_basic():
    """Test basic meteorology reading with required fields."""
    # Create test data with required fields - first row is headers
    data = [
        ["T2m", "FF", "Rain", "Snow"],  # Headers
        [15.5, 5.2, 0.0, 0.0],  # Data row 1
        [16.0, 4.8, 0.5, 0.0],  # Data row 2
        [14.2, 6.1, 0.0, 0.2],  # Data row 3
    ]
    df = pd.DataFrame(data)

    result = read_input_meteorology(df)

    assert result.n_meteo == 3
    assert len(result.T_a[0]) == 3
    assert len(result.FF[0]) == 3
    assert len(result.Rain[0]) == 3
    assert len(result.Snow[0]) == 3

    # Check basic values
    assert result.T_a[0][0] == 15.5
    assert result.FF[0][0] == 5.2
    assert result.Rain[0][0] == 0.0
    assert result.Snow[0][0] == 0.0


def test_read_input_meteorology_with_optional_fields():
    """Test meteorology reading with optional fields."""
    # Create test data with optional fields
    data = [
        [
            "T2m",
            "FF",
            "Rain",
            "Snow",
            "RH",
            "DD",
            "Global radiation",
            "Pressure",
        ],  # Headers
        [15.5, 5.2, 0.0, 0.0, 75.0, 180.0, 500.0, 101325.0],  # Data row 1
        [16.0, 4.8, 0.5, 0.0, 70.0, 175.0, 450.0, 101300.0],  # Data row 2
        [14.2, 6.1, 0.0, 0.2, 80.0, 185.0, 600.0, 101350.0],  # Data row 3
    ]
    df = pd.DataFrame(data)

    result = read_input_meteorology(df)

    assert result.n_meteo == 3
    assert result.RH_available == 1
    assert result.DD_available == 1
    assert result.short_rad_in_available == 1
    assert result.pressure_obs_available == 1

    # Check values
    assert result.RH[0][0] == 75.0
    assert result.DD[0][0] == 180.0
    assert result.short_rad_in[0][0] == 500.0
    assert result.Pressure_a[0][0] == 101325.0


def test_read_input_meteorology_missing_required_field():
    """Test error handling when required field is missing."""
    # Create test data missing T2m
    data = [
        ["FF", "Rain", "Snow"],  # Headers - missing T2m
        [5.2, 0.0, 0.0],  # Data row 1
        [4.8, 0.5, 0.0],  # Data row 2
        [6.1, 0.0, 0.2],  # Data row 3
    ]
    df = pd.DataFrame(data)

    result = read_input_meteorology(df)

    # Should return early with empty result
    assert result.n_meteo == 0


def test_read_input_meteorology_wind_speed_correction():
    """Test wind speed correction factor."""
    data = [
        ["T2m", "FF", "Rain", "Snow"],  # Headers
        [15.5, 5.0, 0.0, 0.0],  # Data row 1
        [16.0, 4.0, 0.5, 0.0],  # Data row 2
        [14.2, 6.0, 0.0, 0.2],  # Data row 3
    ]
    df = pd.DataFrame(data)

    # Test with wind speed correction of 1.5
    result = read_input_meteorology(df, wind_speed_correction=1.5)

    assert result.FF[0][0] == 7.5  # 5.0 * 1.5
    assert result.FF[0][1] == 6.0  # 4.0 * 1.5
    assert result.FF[0][2] == 9.0  # 6.0 * 1.5


def test_read_input_meteorology_road_wetness_units():
    """Test road wetness unit detection."""
    # Test with mm units
    data = [
        ["T2m", "FF", "Rain", "Snow", "Road wetness (mm)"],  # Headers
        [15.5, 5.2, 0.0, 0.0, 1.5],  # Data row 1
        [16.0, 4.8, 0.5, 0.0, 2.0],  # Data row 2
        [14.2, 6.1, 0.0, 0.2, 1.2],  # Data row 3
    ]
    df = pd.DataFrame(data)

    result = read_input_meteorology(df)

    assert result.road_wetness_obs_available == 1
    assert result.road_wetness_obs_in_mm == 1
    assert result.max_road_wetness_obs == 2.0
    assert result.min_road_wetness_obs == 1.2
    assert abs(result.mean_road_wetness_obs - 1.5666666666666667) < 0.01


def test_read_input_meteorology_missing_data_handling():
    """Test handling of missing data (nodata values)."""
    data = [
        ["T2m", "FF", "Rain", "Snow", "RH"],  # Headers
        [15.5, 5.2, 0.0, 0.0, 75.0],  # Data row 1
        [-99.0, 4.8, 0.5, 0.0, 70.0],  # Data row 2 - Missing temperature
        [14.2, 6.1, 0.0, 0.2, -99.0],  # Data row 3 - Missing RH
    ]
    df = pd.DataFrame(data)

    result = read_input_meteorology(df, nodata=-99.0)

    # Missing data should be forward filled
    assert result.T_a[0][1] == 15.5  # Forward filled from previous
    assert result.RH[0][2] == 70.0  # Forward filled from previous
    assert 1 in result.T_a_nodata  # Index 1 was missing
    assert 2 in result.RH_nodata  # Index 2 was missing


def test_read_input_meteorology_negative_value_removal():
    """Test removal of negative values for certain fields."""
    data = [
        ["T2m", "FF", "Rain", "Snow", "RH"],  # Headers
        [15.5, 5.2, -1.0, -0.5, -10.0],  # Data row 1 - Negative rain, snow, RH
        [16.0, 4.8, 0.5, 0.0, 70.0],  # Data row 2
        [14.2, 6.1, 0.0, 0.2, 80.0],  # Data row 3
    ]
    df = pd.DataFrame(data)

    result = read_input_meteorology(df)

    # Negative values should be set to 0
    assert result.Rain[0][0] == 0.0
    assert result.Snow[0][0] == 0.0
    assert result.RH[0][0] == 0.0


def test_read_input_meteorology_dewpoint_calculation():
    """Test automatic dewpoint calculation from RH."""
    data = [
        ["T2m", "FF", "Rain", "Snow", "RH"],  # Headers
        [20.0, 5.2, 0.0, 0.0, 50.0],  # Data row 1
        [15.0, 4.8, 0.5, 0.0, 60.0],  # Data row 2
        [25.0, 6.1, 0.0, 0.2, 40.0],  # Data row 3
    ]
    df = pd.DataFrame(data)

    result = read_input_meteorology(df)

    # Should calculate dewpoint from RH and temperature
    assert result.T_dewpoint_available == 1
    assert len(result.T_dewpoint[0]) == 3
    # At 20°C and 50% RH, dewpoint should be around 9.3°C
    assert abs(result.T_dewpoint[0][0] - 9.3) < 1.0


def test_read_input_meteorology_pressure_default():
    """Test default pressure when not provided."""
    data = [
        ["T2m", "FF", "Rain", "Snow"],  # Headers
        [15.5, 5.2, 0.0, 0.0],  # Data row 1
        [16.0, 4.8, 0.5, 0.0],  # Data row 2
        [14.2, 6.1, 0.0, 0.2],  # Data row 3
    ]
    df = pd.DataFrame(data)

    result = read_input_meteorology(df, pressure_default=102000.0)

    # Should use default pressure
    assert result.pressure_obs_available == 1
    assert result.Pressure_a[0][0] == 102000.0
    assert result.Pressure_a[0][1] == 102000.0
    assert result.Pressure_a[0][2] == 102000.0
