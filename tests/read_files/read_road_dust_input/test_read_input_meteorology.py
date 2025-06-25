import pandas as pd
from read_files.read_road_dust_input.read_input_meteorology import (
    read_input_meteorology,
)


def test_read_input_meteorology_basic():
    """Test basic functionality with required fields only."""
    data = [
        ["T2m", "FF", "Rain", "Snow"],
        [15.5, 5.2, 0.0, 0.0],
        [16.0, 4.8, 2.5, 0.0],
        [14.8, 6.1, 0.0, 1.2],
    ]
    df = pd.DataFrame(data)

    result = read_input_meteorology(df)

    # Check basic structure
    assert result.n_meteo == 3
    assert len(result.T_a) == 3
    assert len(result.FF) == 3
    assert len(result.Rain) == 3
    assert len(result.Snow) == 3

    # Check values
    assert result.T_a[0] == 15.5
    assert result.FF[0] == 5.2
    assert result.Rain[0] == 0.0
    assert result.Snow[0] == 0.0


def test_read_input_meteorology_optional_fields():
    """Test with optional fields present."""
    data = [
        ["T2m", "FF", "Rain", "Snow", "RH", "DD", "Global radiation", "Pressure"],
        [15.5, 5.2, 0.0, 0.0, 75.0, 180.0, 500.0, 101325.0],
        [16.0, 4.8, 2.5, 0.0, 70.0, 200.0, 450.0, 101325.0],
        [14.8, 6.1, 0.0, 1.2, 80.0, 160.0, 300.0, 101325.0],
    ]
    df = pd.DataFrame(data)

    result = read_input_meteorology(df)

    # Check availability flags
    assert result.RH_available == 1
    assert result.DD_available == 1
    assert result.short_rad_in_available == 1
    assert result.pressure_obs_available == 1

    # Check optional field values
    assert result.RH[0] == 75.0
    assert result.DD[0] == 180.0
    assert result.short_rad_in[0] == 500.0
    assert result.Pressure_a[0] == 101325.0


def test_read_input_meteorology_missing_required():
    """Test error handling when required fields are missing."""
    data = [
        ["T2m", "RH"],  # Missing FF, Rain, Snow
        [15.5, 75.0],
        [16.0, 70.0],
    ]
    df = pd.DataFrame(data)

    result = read_input_meteorology(df)
    assert result.n_meteo == 0  # Should fail due to missing required fields


def test_read_input_meteorology_wind_speed_correction():
    """Test wind speed correction factor."""
    data = [
        ["T2m", "FF", "Rain", "Snow"],
        [15.5, 5.0, 0.0, 0.0],
        [16.0, 4.0, 2.5, 0.0],
        [14.8, 6.0, 0.0, 1.2],
    ]
    df = pd.DataFrame(data)

    result = read_input_meteorology(df, wind_speed_correction=1.5)

    # Wind speed should be multiplied by correction factor
    assert result.FF[0] == 7.5  # 5.0 * 1.5
    assert result.FF[1] == 6.0  # 4.0 * 1.5
    assert result.FF[2] == 9.0  # 6.0 * 1.5


def test_read_input_meteorology_road_wetness_units():
    """Test road wetness unit detection."""
    data = [
        ["T2m", "FF", "Rain", "Snow", "Road wetness (mm)"],
        [15.5, 5.2, 0.0, 0.0, 2.5],
        [16.0, 4.8, 2.5, 0.0, 3.0],
        [14.8, 6.1, 0.0, 1.2, 1.5],
    ]
    df = pd.DataFrame(data)

    result = read_input_meteorology(df)

    assert result.road_wetness_obs_available == 1
    assert result.road_wetness_obs_in_mm == 1
    assert result.max_road_wetness_obs == 3.0
    assert result.min_road_wetness_obs == 1.5
    assert abs(result.mean_road_wetness_obs - 2.333) < 0.01  # (2.5 + 3.0 + 1.5) / 3


def test_read_input_meteorology_missing_data():
    """Test handling of missing data with forward filling."""
    data = [
        ["T2m", "FF", "Rain", "Snow", "RH"],
        [15.5, 5.2, 0.0, 0.0, 75.0],
        [-99.0, 4.8, 2.5, 0.0, 70.0],  # Missing T2m
        [14.8, 6.1, 0.0, 1.2, -99.0],  # Missing RH
    ]
    df = pd.DataFrame(data)

    result = read_input_meteorology(df, nodata=-99.0)

    # Missing data should be forward filled
    assert result.T_a[1] == 15.5  # Forward filled from previous
    assert result.RH[2] == 70.0  # Forward filled from previous
    assert 1 in result.T_a_nodata
    assert 2 in result.RH_nodata


def test_read_input_meteorology_negative_removal():
    """Test removal of negative values for Rain, Snow, RH."""
    data = [
        ["T2m", "FF", "Rain", "Snow", "RH"],
        [15.5, 5.2, -1.0, -0.5, -10.0],  # All negative
        [16.0, 4.8, 2.5, 0.0, 70.0],
        [14.8, 6.1, 0.0, 1.2, 80.0],
    ]
    df = pd.DataFrame(data)

    result = read_input_meteorology(df)

    # Negative values should be set to 0
    assert result.Rain[0] == 0.0
    assert result.Snow[0] == 0.0
    assert result.RH[0] == 0.0


def test_read_input_meteorology_dewpoint_calculation():
    """Test automatic dewpoint calculation from T and RH."""
    data = [
        ["T2m", "FF", "Rain", "Snow", "RH"],
        [15.5, 5.2, 0.0, 0.0, 75.0],
        [16.0, 4.8, 2.5, 0.0, 70.0],
        [14.8, 6.1, 0.0, 1.2, 80.0],
    ]
    df = pd.DataFrame(data)

    result = read_input_meteorology(df)

    assert result.T_dewpoint_available == 1
    assert len(result.T_dewpoint) == 3
    # Check dewpoint is reasonable (should be less than air temperature)
    assert abs(result.T_dewpoint[0] - 11.1) < 1.0  # At 15.5°C and 75% RH


def test_read_input_meteorology_default_pressure():
    """Test default pressure handling when not available."""
    data = [
        ["T2m", "FF", "Rain", "Snow"],
        [15.5, 5.2, 0.0, 0.0],
        [16.0, 4.8, 2.5, 0.0],
        [14.8, 6.1, 0.0, 1.2],
    ]
    df = pd.DataFrame(data)

    result = read_input_meteorology(df, pressure_default=102000.0)

    assert result.pressure_obs_available == 1
    assert result.Pressure_a[0] == 102000.0
    assert result.Pressure_a[1] == 102000.0
    assert result.Pressure_a[2] == 102000.0
