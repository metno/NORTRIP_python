import numpy as np
from datetime import datetime
from calculations.calc_radiation import calc_radiation
from initialise import model_variables, time_config
from input_classes import (
    converted_data,
    input_metadata,
    input_initial,
    input_meteorology,
)
from config_classes import model_flags, model_parameters
import constants


def test_calc_radiation():
    """Test the main calc_radiation function."""
    # Create test data
    n_date = 24  # 24 hours
    n_roads = 1
    min_time = 0
    max_time = 23

    # Create time config
    test_time_config = time_config()
    test_time_config.min_time = min_time
    test_time_config.max_time = max_time
    test_time_config.dt = np.float64(1.0)

    # Create converted data
    test_converted_data = converted_data()
    test_converted_data.n_date = n_date
    test_converted_data.n_roads = n_roads
    test_converted_data.nodata = -99.0

    # Initialize arrays
    test_converted_data.date_data = np.zeros(
        (constants.num_date_index, n_date, n_roads)
    )
    test_converted_data.meteo_data = np.zeros(
        (constants.num_meteo_index, n_date, n_roads)
    )

    # Fill with sample data - use Unix timestamps
    base_timestamp = datetime(2023, 6, 21, 12, 0, 0).timestamp()
    test_converted_data.date_data[constants.datenum_index, :, 0] = np.arange(
        base_timestamp, base_timestamp + n_date * 3600, 3600
    )
    test_converted_data.meteo_data[constants.T_a_index, :, 0] = 10.0  # 10°C
    test_converted_data.meteo_data[constants.RH_index, :, 0] = 60.0  # 60% RH
    test_converted_data.meteo_data[constants.short_rad_in_index, :, 0] = 200.0  # W/m²
    test_converted_data.meteo_data[constants.cloud_cover_index, :, 0] = 0.5
    test_converted_data.meteo_data[constants.long_rad_in_index, :, 0] = 300.0  # W/m²

    # Create metadata
    test_metadata = input_metadata()
    test_metadata.nodata = -99.0
    test_metadata.LAT = 60.0
    test_metadata.LON = 10.0
    test_metadata.DIFUTC_H = 1.0
    test_metadata.Z_SURF = 100.0
    test_metadata.albedo_road = 0.1
    test_metadata.Pressure = 101325.0
    test_metadata.b_road = 10.0
    test_metadata.b_canyon = 20.0
    test_metadata.h_canyon = [5.0, 5.0]
    test_metadata.ang_road = 0.0

    # Create initial data
    test_initial = input_initial()
    test_initial.RH_offset = 0.0
    test_initial.T_2m_offset = 0.0
    test_initial.long_rad_in_offset = 0.0

    # Create model flags
    test_flags = model_flags()
    test_flags.canyon_shadow_flag = 0
    test_flags.canyon_long_rad_flag = 0

    # Create model parameters
    test_params = model_parameters()
    test_params.num_track = 1

    # Create meteorology data
    test_meteorology = input_meteorology()
    test_meteorology.short_rad_in_available = 1
    test_meteorology.cloud_cover_available = 1
    test_meteorology.long_rad_in_available = 1

    # Create model variables
    test_model_variables = model_variables()
    test_model_variables.road_meteo_data = np.zeros(
        (constants.num_road_meteo, n_date, constants.num_track_max, n_roads)
    )

    # Run the function
    calc_radiation(
        model_variables=test_model_variables,
        time_config=test_time_config,
        converted_data=test_converted_data,
        metadata=test_metadata,
        initial_data=test_initial,
        model_flags=test_flags,
        model_parameters=test_params,
        meteorology_data=test_meteorology,
    )

    # Check that some values have been set
    assert not np.all(
        test_model_variables.road_meteo_data[constants.short_rad_net_index, :, 0, 0]
        == 0
    )
    assert not np.all(
        test_converted_data.meteo_data[constants.short_rad_in_clearsky_index, :, 0]
        == -99.0
    )


def test_calc_radiation_with_canyon_effects():
    """Test calc_radiation with canyon shadow and longwave effects enabled."""
    # Create minimal test setup
    n_date = 5
    n_roads = 1

    test_time_config = time_config()
    test_time_config.min_time = 0
    test_time_config.max_time = 4

    test_converted_data = converted_data()
    test_converted_data.n_date = n_date
    test_converted_data.n_roads = n_roads
    test_converted_data.nodata = -99.0
    test_converted_data.date_data = np.zeros(
        (constants.num_date_index, n_date, n_roads)
    )
    test_converted_data.meteo_data = np.zeros(
        (constants.num_meteo_index, n_date, n_roads)
    )

    # Fill with sample data - use a realistic date in summer with daylight hours
    # Use July 15, 2023 around noon for better solar radiation
    base_timestamp = datetime(2023, 7, 15, 12, 0, 0).timestamp()
    test_converted_data.date_data[constants.datenum_index, :, 0] = np.arange(
        base_timestamp, base_timestamp + n_date * 3600, 3600
    )
    test_converted_data.meteo_data[constants.T_a_index, :, 0] = 15.0
    test_converted_data.meteo_data[constants.RH_index, :, 0] = 70.0
    test_converted_data.meteo_data[constants.cloud_cover_index, :, 0] = 0.3
    test_converted_data.meteo_data[constants.long_rad_in_index, :, 0] = 350.0

    test_metadata = input_metadata()
    test_metadata.nodata = -99.0
    test_metadata.LAT = 60.0
    test_metadata.LON = 10.0
    test_metadata.DIFUTC_H = 1.0
    test_metadata.Z_SURF = 100.0
    test_metadata.albedo_road = 0.1
    test_metadata.Pressure = 101325.0
    test_metadata.b_road = 10.0
    test_metadata.b_canyon = 20.0
    test_metadata.h_canyon = [8.0, 6.0]
    test_metadata.ang_road = 0.0

    test_initial = input_initial()
    test_initial.RH_offset = 0.0
    test_initial.T_2m_offset = 0.0
    test_initial.long_rad_in_offset = 10.0

    # Enable canyon effects
    test_flags = model_flags()
    test_flags.canyon_shadow_flag = 1
    test_flags.canyon_long_rad_flag = 1

    test_params = model_parameters()
    test_params.num_track = 1

    test_meteorology = input_meteorology()
    test_meteorology.short_rad_in_available = 0  # Force calculation
    test_meteorology.cloud_cover_available = 1
    test_meteorology.long_rad_in_available = 0  # Force calculation

    test_model_variables = model_variables()
    test_model_variables.road_meteo_data = np.zeros(
        (constants.num_road_meteo, n_date, constants.num_track_max, n_roads)
    )

    # Run the function
    calc_radiation(
        model_variables=test_model_variables,
        time_config=test_time_config,
        converted_data=test_converted_data,
        metadata=test_metadata,
        initial_data=test_initial,
        model_flags=test_flags,
        model_parameters=test_params,
        meteorology_data=test_meteorology,
    )

    # Check that calculations have been performed
    # The function should have run without errors
    assert True  # If we get here, the function ran successfully

    # Check that clear sky radiation has been calculated
    assert not np.all(
        test_converted_data.meteo_data[constants.short_rad_in_clearsky_index, :, 0]
        == 0.0
    )

    # Check that longwave radiation with offset has been calculated
    assert not np.all(
        test_converted_data.meteo_data[constants.long_rad_in_index, :, 0] == 350.0
    )
