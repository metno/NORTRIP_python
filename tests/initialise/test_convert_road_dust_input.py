import numpy as np
from initialise.convert_road_dust_input import (
    convert_input_data_to_consolidated_structure,
    convert_road_dust_input,
)
from input_classes import (
    converted_data,
    input_traffic,
    input_meteorology,
    input_activity,
    input_airquality,
)
import constants


def test_convert_input_data_to_consolidated_structure():
    """Test conversion from individual input data classes to consolidated structure."""
    # Create mock input data
    n_traffic = 24  # 24 hours of data

    # Mock traffic data
    traffic_data = input_traffic()
    traffic_data.n_traffic = n_traffic
    traffic_data.year = np.full(n_traffic, 2023)
    traffic_data.month = np.full(n_traffic, 6)
    traffic_data.day = np.full(n_traffic, 15)
    traffic_data.hour = np.arange(24)
    traffic_data.minute = np.zeros(n_traffic)
    traffic_data.date_num = np.arange(738000, 738000 + n_traffic / 24, 1 / 24)
    traffic_data.N_total = np.full(n_traffic, 100.0)
    traffic_data.N_v = np.full((constants.num_veh, n_traffic), 50.0)
    traffic_data.N = np.full((constants.num_tyre, constants.num_veh, n_traffic), 20.0)
    traffic_data.V_veh = np.full((constants.num_veh, n_traffic), 60.0)

    # Mock meteorology data
    meteorology_data = input_meteorology()
    meteorology_data.n_meteo = n_traffic
    meteorology_data.T_a = np.full(n_traffic, 15.0)
    meteorology_data.T2_a = np.full(n_traffic, 14.0)
    meteorology_data.FF = np.full(n_traffic, 3.0)
    meteorology_data.DD = np.full(n_traffic, 180.0)
    meteorology_data.RH = np.full(n_traffic, 70.0)
    meteorology_data.Rain = np.zeros(n_traffic)
    meteorology_data.Snow = np.zeros(n_traffic)
    meteorology_data.short_rad_in = np.full(n_traffic, 400.0)
    meteorology_data.long_rad_in = np.full(n_traffic, 300.0)
    meteorology_data.cloud_cover = np.full(n_traffic, 0.5)
    meteorology_data.road_temperature_obs = np.full(n_traffic, 16.0)
    meteorology_data.road_wetness_obs = np.full(n_traffic, 0.1)
    meteorology_data.T_dewpoint = np.full(n_traffic, 10.0)
    meteorology_data.Pressure_a = np.full(n_traffic, 1013.25)
    meteorology_data.T_sub = np.full(n_traffic, 12.0)

    # Mock activity data
    activity_data = input_activity()
    activity_data.M_sanding = np.zeros(n_traffic)
    activity_data.t_ploughing = np.zeros(n_traffic)
    activity_data.t_cleaning = np.zeros(n_traffic)
    activity_data.g_road_wetting = np.zeros(n_traffic)
    activity_data.M_salting = np.zeros((constants.num_salt, n_traffic))
    activity_data.M_fugitive = np.zeros(n_traffic)

    # Mock air quality data
    airquality_data = input_airquality()
    airquality_data.n_date = n_traffic
    airquality_data.f_dis_input = np.full(n_traffic, 1.0)

    # Perform conversion
    converted = convert_input_data_to_consolidated_structure(
        traffic_data, meteorology_data, activity_data, airquality_data, nodata=-99.0
    )

    # Check basic properties
    assert converted.n_date == n_traffic
    assert converted.n_roads == constants.n_roads
    assert converted.nodata == -99.0

    # Check array dimensions
    assert converted.date_data.shape == (
        constants.num_date_index,
        n_traffic,
        constants.n_roads,
    )
    assert converted.traffic_data.shape == (
        constants.num_traffic_index,
        n_traffic,
        constants.n_roads,
    )
    assert converted.meteo_data.shape == (
        constants.num_meteo_index,
        n_traffic,
        constants.n_roads,
    )
    assert converted.activity_data.shape == (
        constants.num_activity_index,
        n_traffic,
        constants.n_roads,
    )

    # Check that date data was properly transferred
    np.testing.assert_array_equal(
        converted.date_data[constants.year_index, :, 0], traffic_data.year
    )
    np.testing.assert_array_equal(
        converted.date_data[constants.month_index, :, 0], traffic_data.month
    )
    np.testing.assert_array_equal(
        converted.date_data[constants.hour_index, :, 0], traffic_data.hour
    )

    # Check that traffic data was properly transferred
    np.testing.assert_array_equal(
        converted.traffic_data[constants.N_total_index, :, 0], traffic_data.N_total
    )

    # Check that meteorological data was properly transferred
    np.testing.assert_array_equal(
        converted.meteo_data[constants.T_a_index, :, 0], meteorology_data.T_a
    )
    np.testing.assert_array_equal(
        converted.meteo_data[constants.FF_index, :, 0], meteorology_data.FF
    )

    # Check that activity data was properly transferred
    np.testing.assert_array_equal(
        converted.activity_data[constants.M_sanding_index, :, 0],
        activity_data.M_sanding,
    )

    # Check that activity_data_input is a copy of activity_data
    np.testing.assert_array_equal(
        converted.activity_data, converted.activity_data_input
    )

    # Check dispersion factor data
    np.testing.assert_array_equal(converted.f_dis[:, 0], airquality_data.f_dis_input)


def test_convert_input_data_with_different_lengths():
    """Test conversion when input data have different lengths."""
    n_traffic = 24
    n_meteo = 20  # Shorter meteorology data
    n_activity = 12  # Even shorter activity data

    # Create mock data with different lengths
    traffic_data = input_traffic()
    traffic_data.n_traffic = n_traffic
    traffic_data.year = np.full(n_traffic, 2023)
    traffic_data.month = np.full(n_traffic, 6)
    traffic_data.day = np.full(n_traffic, 15)
    traffic_data.hour = np.arange(24)
    traffic_data.minute = np.zeros(n_traffic)
    traffic_data.date_num = np.arange(738000, 738000 + n_traffic / 24, 1 / 24)
    traffic_data.N_total = np.full(n_traffic, 100.0)
    traffic_data.N_v = np.full((constants.num_veh, n_traffic), 50.0)
    traffic_data.N = np.full((constants.num_tyre, constants.num_veh, n_traffic), 20.0)
    traffic_data.V_veh = np.full((constants.num_veh, n_traffic), 60.0)

    meteorology_data = input_meteorology()
    meteorology_data.n_meteo = n_meteo
    meteorology_data.T_a = np.full(n_meteo, 15.0)
    meteorology_data.T2_a = np.full(n_meteo, 14.0)
    meteorology_data.FF = np.full(n_meteo, 3.0)
    meteorology_data.DD = np.full(n_meteo, 180.0)
    meteorology_data.RH = np.full(n_meteo, 70.0)
    meteorology_data.Rain = np.zeros(n_meteo)
    meteorology_data.Snow = np.zeros(n_meteo)
    meteorology_data.short_rad_in = np.full(n_meteo, 400.0)
    meteorology_data.long_rad_in = np.full(n_meteo, 300.0)
    meteorology_data.cloud_cover = np.full(n_meteo, 0.5)
    meteorology_data.road_temperature_obs = np.full(n_meteo, 16.0)
    meteorology_data.road_wetness_obs = np.full(n_meteo, 0.1)
    meteorology_data.T_dewpoint = np.full(n_meteo, 10.0)
    meteorology_data.Pressure_a = np.full(n_meteo, 1013.25)
    meteorology_data.T_sub = np.full(n_meteo, 12.0)

    activity_data = input_activity()
    activity_data.M_sanding = np.zeros(n_activity)
    activity_data.t_ploughing = np.zeros(n_activity)
    activity_data.t_cleaning = np.zeros(n_activity)
    activity_data.g_road_wetting = np.zeros(n_activity)
    activity_data.M_salting = np.zeros((constants.num_salt, n_activity))
    activity_data.M_fugitive = np.zeros(n_activity)

    airquality_data = input_airquality()
    airquality_data.n_date = n_traffic
    airquality_data.f_dis_input = np.full(n_traffic, 1.0)

    # Perform conversion
    converted = convert_input_data_to_consolidated_structure(
        traffic_data, meteorology_data, activity_data, airquality_data, nodata=-99.0
    )

    # Check that conversion handles different lengths properly
    assert converted.n_date == n_traffic

    # Check that meteorology data is filled for available length
    np.testing.assert_array_equal(
        converted.meteo_data[constants.T_a_index, :n_meteo, 0], meteorology_data.T_a
    )
    # Check that remaining values are nodata
    assert np.all(converted.meteo_data[constants.T_a_index, n_meteo:, 0] == -99.0)

    # Check that activity data is filled for available length
    np.testing.assert_array_equal(
        converted.activity_data[constants.M_sanding_index, :n_activity, 0],
        activity_data.M_sanding,
    )
    # Check that remaining values are nodata
    assert np.all(
        converted.activity_data[constants.M_sanding_index, n_activity:, 0] == -99.0
    )


def test_convert_road_dust_input():
    """Test the wrapper function convert_road_dust_input."""
    # Create mock input data
    n_traffic = 12

    # Mock traffic data
    traffic_data = input_traffic()
    traffic_data.n_traffic = n_traffic
    traffic_data.year = np.full(n_traffic, 2023)
    traffic_data.month = np.full(n_traffic, 6)
    traffic_data.day = np.full(n_traffic, 15)
    traffic_data.hour = np.arange(12)
    traffic_data.minute = np.zeros(n_traffic)
    traffic_data.date_num = np.arange(738000, 738000 + n_traffic / 24, 1 / 24)
    traffic_data.N_total = np.full(n_traffic, 100.0)
    traffic_data.N_v = np.full((constants.num_veh, n_traffic), 50.0)
    traffic_data.N = np.full((constants.num_tyre, constants.num_veh, n_traffic), 20.0)
    traffic_data.V_veh = np.full((constants.num_veh, n_traffic), 60.0)

    # Mock other data with minimal content
    meteorology_data = input_meteorology()
    meteorology_data.n_meteo = n_traffic
    meteorology_data.T_a = np.full(n_traffic, 15.0)
    meteorology_data.T2_a = np.full(n_traffic, 14.0)
    meteorology_data.FF = np.full(n_traffic, 3.0)
    meteorology_data.DD = np.full(n_traffic, 180.0)
    meteorology_data.RH = np.full(n_traffic, 70.0)
    meteorology_data.Rain = np.zeros(n_traffic)
    meteorology_data.Snow = np.zeros(n_traffic)
    meteorology_data.short_rad_in = np.full(n_traffic, 400.0)
    meteorology_data.long_rad_in = np.full(n_traffic, 300.0)
    meteorology_data.cloud_cover = np.full(n_traffic, 0.5)
    meteorology_data.road_temperature_obs = np.full(n_traffic, 16.0)
    meteorology_data.road_wetness_obs = np.full(n_traffic, 0.1)
    meteorology_data.T_dewpoint = np.full(n_traffic, 10.0)
    meteorology_data.Pressure_a = np.full(n_traffic, 1013.25)
    meteorology_data.T_sub = np.full(n_traffic, 12.0)

    activity_data = input_activity()
    activity_data.M_sanding = np.zeros(n_traffic)
    activity_data.t_ploughing = np.zeros(n_traffic)
    activity_data.t_cleaning = np.zeros(n_traffic)
    activity_data.g_road_wetting = np.zeros(n_traffic)
    activity_data.M_salting = np.zeros((constants.num_salt, n_traffic))
    activity_data.M_fugitive = np.zeros(n_traffic)

    airquality_data = input_airquality()
    airquality_data.n_date = n_traffic
    airquality_data.f_dis_input = np.full(n_traffic, 1.0)

    # Create input tuple (as returned by read_road_dust_input)
    input_data_tuple = (
        activity_data,
        airquality_data,
        meteorology_data,
        traffic_data,
        None,  # initial_data
        None,  # metadata_data
    )

    # Test the wrapper function
    converted = convert_road_dust_input(input_data_tuple, nodata=-99.0)

    # Check basic properties
    assert converted.n_date == n_traffic
    assert converted.n_roads == constants.n_roads
    assert converted.nodata == -99.0

    # Verify it calls the underlying function correctly
    np.testing.assert_array_equal(
        converted.traffic_data[constants.N_total_index, :, 0], traffic_data.N_total
    )
