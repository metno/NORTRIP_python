import numpy as np
import constants
from initialise import road_dust_initialise_variables, time_config
from config_classes import model_parameters, model_flags
from input_classes import (
    input_metadata,
    input_airquality,
    input_initial,
    converted_data,
)


def test_road_dust_initialise_variables_with_forecast():
    """Test the road_dust_initialise_variables function with forecast enabled."""

    # Create test time configuration
    test_time_config = time_config()
    test_time_config.min_time = 0
    test_time_config.max_time = 23
    test_time_config.dt = np.float64(1.0)

    # Create minimal test data
    n_date = 24
    n_roads = 1
    test_converted_data = converted_data()
    test_converted_data.n_date = n_date
    test_converted_data.n_roads = n_roads
    test_converted_data.nodata = -99.0

    # Initialize minimal required arrays
    test_converted_data.meteo_data = np.zeros(
        (constants.num_meteo_index, n_date, n_roads)
    )
    test_converted_data.traffic_data = np.zeros(
        (constants.num_traffic_index, n_date, n_roads)
    )
    test_converted_data.meteo_data[constants.T_a_index, :, 0] = 10.0
    test_converted_data.meteo_data[constants.RH_index, :, 0] = 60.0

    test_initial = input_initial()
    test_initial.m_road_init = np.zeros((constants.num_source, constants.num_track_max))
    test_initial.g_road_init = np.zeros(
        (constants.num_moisture, constants.num_track_max)
    )

    test_metadata = input_metadata()
    test_metadata.nodata = -99.0
    test_metadata.exhaust_EF_available = 0
    test_metadata.exhaust_EF = [0.0, 0.0]

    test_airquality = input_airquality()
    test_airquality.EP_emis_available = 0

    test_model_params = model_parameters()
    test_model_params.num_track = 1
    test_model_params.f_PM = np.zeros(
        (constants.num_source_all_extra, constants.num_size, constants.num_tyre)
    )

    # Test with forecast enabled
    test_model_flags = model_flags()
    test_model_flags.exhaust_flag = 0
    test_model_flags.forecast_hour = 6  # 6-hour forecast

    result = road_dust_initialise_variables(
        test_time_config,
        test_converted_data,
        test_initial,
        test_metadata,
        test_airquality,
        test_model_params,
        test_model_flags,
    )

    # Check forecast arrays are properly sized
    expected_forecast_steps = (
        int(test_model_flags.forecast_hour / test_time_config.dt) + 1
    )
    assert result.forecast_hours.shape == (expected_forecast_steps, n_date)
    assert result.E_corr_array.shape == (
        int(test_model_flags.forecast_hour / test_time_config.dt),
        n_date,
    )
