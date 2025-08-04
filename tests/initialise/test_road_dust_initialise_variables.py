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


def test_road_dust_initialise_variables():
    """Test the road_dust_initialise_variables function."""

    # Create test time configuration
    test_time_config = time_config()
    test_time_config.min_time = 0
    test_time_config.max_time = 23
    test_time_config.dt = np.float64(1.0)

    # Create test converted data
    n_date = 24
    n_roads = 1
    test_converted_data = converted_data()
    test_converted_data.n_date = n_date
    test_converted_data.n_roads = n_roads
    test_converted_data.nodata = -99.0

    # Initialize arrays
    test_converted_data.date_data = np.zeros(
        (constants.num_date_index, n_date, n_roads)
    )
    test_converted_data.traffic_data = np.zeros(
        (constants.num_traffic_index, n_date, n_roads)
    )
    test_converted_data.meteo_data = np.zeros(
        (constants.num_meteo_index, n_date, n_roads)
    )

    # Fill with sample meteorological data
    test_converted_data.meteo_data[constants.T_a_index, :, 0] = 10.0  # 10°C
    test_converted_data.meteo_data[constants.RH_index, :, 0] = 60.0  # 60% RH
    test_converted_data.meteo_data[constants.road_temperature_obs_input_index, :, 0] = (
        12.0
    )
    test_converted_data.meteo_data[constants.road_wetness_obs_input_index, :, 0] = 0.5

    # Fill traffic data
    test_converted_data.traffic_data[constants.N_v_index[0], :, 0] = (
        100.0  # Heavy vehicles
    )
    test_converted_data.traffic_data[constants.N_v_index[1], :, 0] = (
        500.0  # Light vehicles
    )

    # Create test initial data
    test_initial = input_initial()
    test_initial.m_road_init = np.zeros((constants.num_source, constants.num_track_max))
    test_initial.g_road_init = np.zeros(
        (constants.num_moisture, constants.num_track_max)
    )

    # Set some initial values
    test_initial.m_road_init[constants.road_index, 0] = 100.0  # Initial road dust
    test_initial.g_road_init[constants.water_index, 0] = 1.0  # Initial water

    # Create test metadata
    test_metadata = input_metadata()
    test_metadata.nodata = -99.0
    test_metadata.exhaust_EF_available = 1
    test_metadata.exhaust_EF = [1.5, 0.8]  # [heavy, light] emission factors

    # Create test air quality data
    test_airquality = input_airquality()
    test_airquality.EP_emis_available = 0
    test_airquality.EP_emis = np.zeros(n_date)

    # Create test model parameters
    test_model_params = model_parameters()
    test_model_params.num_track = 1
    test_model_params.f_track = [1.0]

    # Initialize f_PM array with sample values
    test_model_params.f_PM = np.zeros(
        (constants.num_source_all_extra, constants.num_size, constants.num_tyre)
    )
    # Set some sample fractional distributions
    test_model_params.f_PM[constants.road_index, constants.pm_all, constants.su] = 1.0
    test_model_params.f_PM[constants.road_index, constants.pm_200, constants.su] = 0.8
    test_model_params.f_PM[constants.road_index, constants.pm_10, constants.su] = 0.3
    test_model_params.f_PM[constants.road_index, constants.pm_25, constants.su] = 0.1

    # Create test model flags
    test_model_flags = model_flags()
    test_model_flags.exhaust_flag = 1
    test_model_flags.forecast_hour = 0

    # Run the function
    result = road_dust_initialise_variables(
        test_time_config,
        test_converted_data,
        test_initial,
        test_metadata,
        test_airquality,
        test_model_params,
        test_model_flags,
    )

    # Verify the results
    assert isinstance(result.M_road_data, np.ndarray)
    assert result.M_road_data.shape == (
        constants.num_source_all,
        constants.num_size,
        n_date,
        constants.num_track_max,
        n_roads,
    )

    assert isinstance(result.M_road_bin_data, np.ndarray)
    assert result.M_road_bin_data.shape == (
        constants.num_source_all,
        constants.num_size,
        n_date,
        constants.num_track_max,
        n_roads,
    )

    assert isinstance(result.road_meteo_data, np.ndarray)
    assert result.road_meteo_data.shape == (
        constants.num_road_meteo,
        n_date,
        constants.num_track_max,
        n_roads,
    )

    assert isinstance(result.g_road_data, np.ndarray)
    assert result.g_road_data.shape == (
        constants.num_moisture,
        n_date,
        constants.num_track_max,
        n_roads,
    )

    # Check that initial conditions were set correctly
    ti = test_time_config.min_time

    # Check initial mass loading
    expected_mass = (
        test_initial.m_road_init[constants.road_index, 0]
        * test_model_params.f_PM[constants.road_index, constants.pm_all, constants.su]
    )
    assert (
        result.M_road_data[constants.road_index, constants.pm_all, ti, 0, 0]
        == expected_mass
    )

    # Check initial moisture
    assert (
        result.g_road_data[constants.water_index, ti, 0, 0]
        == test_initial.g_road_init[constants.water_index, 0]
    )

    # Check initial temperature and humidity
    assert (
        result.road_meteo_data[constants.T_s_index, ti, 0, 0]
        == test_converted_data.meteo_data[constants.T_a_index, ti, 0]
    )
    assert (
        result.road_meteo_data[constants.RH_s_index, ti, 0, 0]
        == test_converted_data.meteo_data[constants.RH_index, ti, 0]
    )

    # Check that observed road temperature and wetness were copied
    for t in range(n_date):
        assert (
            result.road_meteo_data[constants.road_temperature_obs_index, t, 0, 0]
            == 12.0
        )
        assert result.road_meteo_data[constants.road_wetness_obs_index, t, 0, 0] == 0.5

    # Check quality factor initialization
    assert isinstance(result.f_q, np.ndarray)
    assert np.all(result.f_q == 1.0)  # Should be initialized to 1

    assert isinstance(result.f_q_obs, np.ndarray)
    assert np.all(
        result.f_q_obs == test_metadata.nodata
    )  # Should be initialized to nodata

    # Check exhaust emissions were initialized
    x = constants.pm_25
    assert (
        result.E_road_bin_data[
            constants.exhaust_index, x, constants.E_total_index, ti, 0, 0
        ]
        > 0
    )

    # Verify binned mass calculation
    for x in range(constants.num_size - 1):
        expected_bin = (
            result.M_road_data[constants.road_index, x, ti, 0, 0]
            - result.M_road_data[constants.road_index, x + 1, ti, 0, 0]
        )
        assert result.M_road_bin_data[constants.road_index, x, ti, 0, 0] == expected_bin

    # Check last size fraction
    x = constants.num_size - 1
    assert (
        result.M_road_bin_data[constants.road_index, x, ti, 0, 0]
        == result.M_road_data[constants.road_index, x, ti, 0, 0]
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
