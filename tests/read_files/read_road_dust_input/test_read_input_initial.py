import pandas as pd
import numpy as np
from src.read_files.read_road_dust_input.read_input_initial import read_input_initial
from src.config_classes.model_parameters import model_parameters
from src.input_classes.input_metadata import input_metadata
import src.constants as constants


def test_read_input_initial_basic():
    """Test basic initial data reading functionality with all parameters."""
    # Create test DataFrame with header-value pairs
    test_data = [
        ["Parameter", "Value"],
        ["M_dust_road", "100.0"],
        ["M_sand_road", "50.0"],
        ["M_salt_road_na", "25.0"],
        ["g_road", "0.5"],
        ["s_road", "0.3"],
        ["i_road", "0.2"],
        ["long_rad_in_offset", "10.0"],
        ["RH_offset", "5.0"],
        ["T_2m_offset", "2.0"],
        ["P_fugitive", "0.1"],
        ["P2_fugitive", "0.05"],
    ]

    initial_df = pd.DataFrame(test_data)

    # Create test model parameters
    test_model_params = model_parameters()
    test_model_params.num_track = 2
    test_model_params.f_track = [0.7, 0.3]

    # Create test metadata
    test_metadata = input_metadata()
    test_metadata.b_road_lanes = 7.0  # 2 lanes * 3.5m

    result = read_input_initial(initial_df, test_model_params, test_metadata)

    # Check basic structure
    assert result.m_road_init.shape == (constants.num_source, constants.num_track_max)
    assert result.g_road_init.shape == (constants.num_moisture, constants.num_track_max)

    # Check that dust road mass is distributed correctly across tracks
    np.testing.assert_almost_equal(
        result.m_road_init[constants.road_index, 0], 100.0 * 0.7
    )
    np.testing.assert_almost_equal(
        result.m_road_init[constants.road_index, 1], 100.0 * 0.3
    )

    # Check sand mass distribution
    np.testing.assert_almost_equal(
        result.m_road_init[constants.sand_index, 0], 50.0 * 0.7
    )
    np.testing.assert_almost_equal(
        result.m_road_init[constants.sand_index, 1], 50.0 * 0.3
    )

    # Check salt mass distribution
    np.testing.assert_almost_equal(
        result.m_road_init[constants.salt_index[0], 0], 25.0 * 0.7
    )
    np.testing.assert_almost_equal(
        result.m_road_init[constants.salt_index[0], 1], 25.0 * 0.3
    )

    # Check moisture values (same across tracks)
    assert result.g_road_init[constants.water_index, 0] == 0.5
    assert result.g_road_init[constants.snow_index, 0] == 0.3
    assert result.g_road_init[constants.ice_index, 0] == 0.2
    assert result.g_road_init[constants.water_index, 1] == 0.5  # Same for all tracks
    assert result.g_road_init[constants.snow_index, 1] == 0.3
    assert result.g_road_init[constants.ice_index, 1] == 0.2

    # Check offset values
    assert result.long_rad_in_offset == 10.0
    assert result.RH_offset == 5.0
    assert result.T_2m_offset == 2.0
    assert result.P_fugitive == 0.1
    assert result.P2_fugitive == 0.05


def test_read_input_initial_with_m2_values():
    """Test reading with M2_ (mass per area) values instead of M_ values."""
    test_data = [
        ["Parameter", "Value"],
        ["M2_dust_road", "0.1"],  # Will be multiplied by b_road_lanes * 1000
        ["M2_sand_road", "0.05"],
        ["M2_salt_road_na", "0.025"],
        ["water_road", "0.8"],  # Alternative to g_road
        ["snow_road", "0.4"],  # Alternative to s_road
        ["ice_road", "0.3"],  # Alternative to i_road
    ]

    initial_df = pd.DataFrame(test_data)

    test_model_params = model_parameters()
    test_model_params.num_track = 1
    test_model_params.f_track = [1.0]

    test_metadata = input_metadata()
    test_metadata.b_road_lanes = 7.0

    result = read_input_initial(initial_df, test_model_params, test_metadata)

    # Check that M2_ values are converted correctly (M2 * b_road_lanes * 1000)
    expected_dust = 0.1 * 7.0 * 1000
    expected_sand = 0.05 * 7.0 * 1000
    expected_salt = 0.025 * 7.0 * 1000

    assert result.m_road_init[constants.road_index, 0] == expected_dust
    assert result.m_road_init[constants.sand_index, 0] == expected_sand
    assert result.m_road_init[constants.salt_index[0], 0] == expected_salt

    # Check that alternative moisture names are used
    assert result.g_road_init[constants.water_index, 0] == 0.8
    assert result.g_road_init[constants.snow_index, 0] == 0.4
    assert result.g_road_init[constants.ice_index, 0] == 0.3


def test_read_input_initial_salt_priority():
    """Test salt value priority: M2_salt_road_mg > M_salt_road_mg etc."""
    test_data = [
        ["Parameter", "Value"],
        ["M_salt_road_na", "10.0"],
        ["M2_salt_road_mg", "0.02"],  # Should take priority over M_salt_road_mg
        ["M_salt_road_mg", "15.0"],  # Should be ignored since M2_ is available
        ["M_salt_road_cma", "20.0"],  # Should be ignored since mg is found first
    ]

    initial_df = pd.DataFrame(test_data)

    test_model_params = model_parameters()
    test_model_params.num_track = 1
    test_model_params.f_track = [1.0]

    test_metadata = input_metadata()
    test_metadata.b_road_lanes = 7.0

    result = read_input_initial(initial_df, test_model_params, test_metadata)

    # First salt index should be NA salt
    assert result.m_road_init[constants.salt_index[0], 0] == 10.0

    # Second salt index should be M2_salt_road_mg (first non-zero M2_ value)
    expected_salt2 = 0.02 * 7.0 * 1000
    assert result.m_road_init[constants.salt_index[1], 0] == expected_salt2


def test_read_input_initial_salt_fallback():
    """Test salt value fallback: if no M2_ values, use M_ values."""
    test_data = [
        ["Parameter", "Value"],
        ["M_salt_road_na", "10.0"],
        ["M_salt_road_cma", "25.0"],  # Should be used for salt_index[1]
        ["M_salt_road_ca", "30.0"],  # Should be ignored since cma is found first
    ]

    initial_df = pd.DataFrame(test_data)

    test_model_params = model_parameters()
    test_model_params.num_track = 1
    test_model_params.f_track = [1.0]

    test_metadata = input_metadata()
    test_metadata.b_road_lanes = 7.0

    result = read_input_initial(initial_df, test_model_params, test_metadata)

    # First salt index should be NA salt
    assert result.m_road_init[constants.salt_index[0], 0] == 10.0

    # Second salt index should be first non-zero M_ value (cma)
    assert result.m_road_init[constants.salt_index[1], 0] == 25.0


def test_read_input_initial_missing_parameters():
    """Test handling of missing parameters - should use defaults from input_values_initial."""
    # Minimal test data with only one parameter
    test_data = [
        ["Parameter", "Value"],
        ["M_dust_road", "75.0"],
    ]

    initial_df = pd.DataFrame(test_data)

    test_model_params = model_parameters()
    test_model_params.num_track = 1
    test_model_params.f_track = [1.0]

    test_metadata = input_metadata()
    test_metadata.b_road_lanes = 7.0

    result = read_input_initial(initial_df, test_model_params, test_metadata)

    # Check that specified value is used
    assert result.m_road_init[constants.road_index, 0] == 75.0

    # Check that missing values default to 0.0 (from input_values_initial defaults)
    assert result.m_road_init[constants.sand_index, 0] == 0.0
    assert result.m_road_init[constants.salt_index[0], 0] == 0.0
    assert result.g_road_init[constants.water_index, 0] == 0.0
    assert result.long_rad_in_offset == 0.0
    assert result.RH_offset == 0.0


def test_read_input_initial_multiple_tracks():
    """Test distribution across multiple tracks."""
    test_data = [
        ["Parameter", "Value"],
        ["M_dust_road", "100.0"],
        ["M_sand_road", "60.0"],
        ["g_road", "0.5"],
    ]

    initial_df = pd.DataFrame(test_data)

    test_model_params = model_parameters()
    test_model_params.num_track = 3
    test_model_params.f_track = [0.5, 0.3, 0.2]

    test_metadata = input_metadata()
    test_metadata.b_road_lanes = 7.0

    result = read_input_initial(initial_df, test_model_params, test_metadata)

    # Check distribution of mass across tracks
    np.testing.assert_almost_equal(
        result.m_road_init[constants.road_index, 0], 100.0 * 0.5
    )
    np.testing.assert_almost_equal(
        result.m_road_init[constants.road_index, 1], 100.0 * 0.3
    )
    np.testing.assert_almost_equal(
        result.m_road_init[constants.road_index, 2], 100.0 * 0.2
    )

    np.testing.assert_almost_equal(
        result.m_road_init[constants.sand_index, 0], 60.0 * 0.5
    )
    np.testing.assert_almost_equal(
        result.m_road_init[constants.sand_index, 1], 60.0 * 0.3
    )
    np.testing.assert_almost_equal(
        result.m_road_init[constants.sand_index, 2], 60.0 * 0.2
    )

    # Check that moisture values are the same across all tracks
    assert result.g_road_init[constants.water_index, 0] == 0.5
    assert result.g_road_init[constants.water_index, 1] == 0.5
    assert result.g_road_init[constants.water_index, 2] == 0.5


def test_read_input_initial_european_decimal_format():
    """Test handling of European decimal format (comma as decimal separator)."""
    test_data = [
        ["Parameter", "Value"],
        ["M_dust_road", "100,5"],  # European format with comma
        ["long_rad_in_offset", "15,25"],
        ["RH_offset", "3,75"],
    ]

    initial_df = pd.DataFrame(test_data)

    test_model_params = model_parameters()
    test_model_params.num_track = 1
    test_model_params.f_track = [1.0]

    test_metadata = input_metadata()
    test_metadata.b_road_lanes = 7.0

    result = read_input_initial(initial_df, test_model_params, test_metadata)

    # Check that comma decimal separators are properly converted
    assert result.m_road_init[constants.road_index, 0] == 100.5
    assert result.long_rad_in_offset == 15.25
    assert result.RH_offset == 3.75


def test_read_input_initial_with_print_results():
    """Test that print_results parameter doesn't break functionality."""
    test_data = [
        ["Parameter", "Value"],
        ["M_dust_road", "50.0"],
        ["RH_offset", "2.5"],
    ]

    initial_df = pd.DataFrame(test_data)

    test_model_params = model_parameters()
    test_model_params.num_track = 1
    test_model_params.f_track = [1.0]

    test_metadata = input_metadata()
    test_metadata.b_road_lanes = 7.0

    # Test with print_results=True
    result = read_input_initial(
        initial_df, test_model_params, test_metadata, print_results=True
    )

    assert result.m_road_init[constants.road_index, 0] == 50.0
    assert result.RH_offset == 2.5


def test_read_input_initial_zero_track_multiplier():
    """Test edge case with zero track multiplier."""
    test_data = [
        ["Parameter", "Value"],
        ["M_dust_road", "100.0"],
        ["g_road", "0.5"],
    ]

    initial_df = pd.DataFrame(test_data)

    test_model_params = model_parameters()
    test_model_params.num_track = 2
    test_model_params.f_track = [0.0, 1.0]  # First track gets nothing

    test_metadata = input_metadata()
    test_metadata.b_road_lanes = 7.0

    result = read_input_initial(initial_df, test_model_params, test_metadata)

    # Check that zero multiplier results in zero mass
    assert result.m_road_init[constants.road_index, 0] == 0.0
    assert result.m_road_init[constants.road_index, 1] == 100.0

    # Moisture should still be the same
    assert result.g_road_init[constants.water_index, 0] == 0.5
    assert result.g_road_init[constants.water_index, 1] == 0.5
