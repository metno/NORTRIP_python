"""
Tests for set_activity_data module.
"""

import numpy as np
from initialise import (
    set_activity_data_v2,
    activity_state,
    time_config,
    model_variables,
)
from input_classes import converted_data
from config_classes import model_flags, model_activities, model_parameters
import constants


def test_activity_state_creation():
    """Test that activity_state can be created and initialized correctly."""
    state = activity_state()

    assert state.last_salting_time.shape == (constants.n_roads,)
    assert state.last_sanding_time.shape == (constants.n_roads,)
    assert state.last_binding_time.shape == (constants.n_roads,)
    assert state.time_since_last_ploughing.shape == (constants.n_roads,)
    assert state.time_since_last_cleaning.shape == (constants.n_roads,)

    # Should all be initialized to zero
    assert np.all(state.last_salting_time == 0)
    assert np.all(state.last_sanding_time == 0)
    assert np.all(state.last_binding_time == 0)
    assert np.all(state.time_since_last_ploughing == 0)
    assert np.all(state.time_since_last_cleaning == 0)


def test_set_activity_data_v2_basic():
    """Test basic functionality of set_activity_data_v2 function."""
    # Create minimal test data
    state = activity_state()

    # Create a minimal time_config
    time_config_mock = time_config()
    time_config_mock.min_time = 0
    time_config_mock.max_time = 10
    time_config_mock.dt = np.float64(1.0)

    # Create minimal converted_data
    converted_data_mock = converted_data()
    n_date = 11
    converted_data_mock.n_date = n_date

    # Initialize arrays with proper shapes
    converted_data_mock.activity_data = np.zeros(
        (constants.num_activity_index, n_date, constants.n_roads)
    )
    converted_data_mock.meteo_data = np.zeros(
        (constants.num_meteo_index, n_date, constants.n_roads)
    )
    converted_data_mock.date_data = np.zeros(
        (constants.num_date_index, n_date, constants.n_roads)
    )

    # Set some basic date data
    converted_data_mock.date_data[constants.hour_index, :, :] = 12  # Hour 12
    converted_data_mock.date_data[constants.month_index, :, :] = 1  # January
    converted_data_mock.date_data[constants.datenum_index, :, :] = np.arange(n_date)[
        :, np.newaxis
    ]

    # Set some meteorological data
    converted_data_mock.meteo_data[constants.T_a_index, :, :] = -3.0  # Cold temperature
    converted_data_mock.meteo_data[constants.Rain_precip_index, :, :] = (
        0.5  # Some precipitation
    )
    converted_data_mock.meteo_data[constants.Snow_precip_index, :, :] = 0.5  # Some snow
    converted_data_mock.meteo_data[constants.RH_index, :, :] = 95.0  # High humidity

    # Create minimal model variables with g_road_data
    model_variables_mock = model_variables()
    model_variables_mock.g_road_data = np.zeros(
        (
            constants.num_moisture,
            n_date,
            constants.num_track_max,
            constants.n_roads,
        )
    )

    # Create minimal model flags
    flags = model_flags()
    flags.auto_salting_flag = 1  # Enable auto salting

    # Create minimal model activities
    activities = model_activities()
    activities.salting_hour = [6, 18]  # Allow salting at 6 and 18
    activities.min_temp_salt = -10.0
    activities.max_temp_salt = 0.0
    activities.precip_rule_salt = 0.1
    activities.RH_rule_salt = 90.0

    # Create minimal model parameters
    parameters = model_parameters()
    parameters.ploughing_thresh_moisture = np.ones(constants.num_moisture) * 0.1

    # Test the function doesn't crash
    ti = 5
    ro = 0

    # This should not raise an exception
    set_activity_data_v2(
        ti=ti,
        ro=ro,
        time_config=time_config_mock,
        converted_data=converted_data_mock,
        model_variables=model_variables_mock,
        model_flags=flags,
        model_activities=activities,
        model_parameters=parameters,
        state=state,
    )

    # Basic check that activity data remains numeric (not NaN)
    assert np.isfinite(
        converted_data_mock.activity_data[constants.M_salting_index[0], ti, ro]
    )
    assert np.isfinite(
        converted_data_mock.activity_data[constants.M_salting_index[1], ti, ro]
    )

    # Since hour=12 is not in salting_hour=[6,18], no salting should occur
    assert (
        converted_data_mock.activity_data[constants.M_salting_index[0], ti, ro] == 0.0
    )
    assert (
        converted_data_mock.activity_data[constants.M_salting_index[1], ti, ro] == 0.0
    )
    assert (
        converted_data_mock.activity_data[constants.g_road_wetting_index, ti, ro] == 0.0
    )

    # State should remain at initial value since no salting initialization or activity occurred
    assert state.last_salting_time[ro] == 0.0

    # Test with salting hour
    ti_salt = 6
    converted_data_mock.date_data[constants.hour_index, ti_salt, ro] = 6  # Salting hour

    set_activity_data_v2(
        ti=ti_salt,
        ro=ro,
        time_config=time_config_mock,
        converted_data=converted_data_mock,
        model_variables=model_variables_mock,
        model_flags=flags,
        model_activities=activities,
        model_parameters=parameters,
        state=state,
    )

    # Now salting should occur (weather conditions are met and it's salting hour)
    salt_mass_1 = activities.salt_mass * activities.salt_type_distribution
    salt_mass_2 = activities.salt_mass * (1 - activities.salt_type_distribution)

    assert (
        converted_data_mock.activity_data[constants.M_salting_index[0], ti_salt, ro]
        == salt_mass_1
    )
    assert (
        converted_data_mock.activity_data[constants.M_salting_index[1], ti_salt, ro]
        == salt_mass_2
    )

    # State should be updated
    assert state.last_salting_time[ro] == ti_salt

    # Test wetting occurs when surface is dry enough
    # Current g_salting_rule=0.1, g_road_0_data should be 0 (dry surface)
    expected_wetting = (
        activities.salt_mass
        * (1 - activities.salt_dilution)
        / activities.salt_dilution
        * 1e-3
    )
    assert (
        converted_data_mock.activity_data[constants.g_road_wetting_index, ti_salt, ro]
        == expected_wetting
    )


def test_set_activity_data_v2_ploughing():
    """Test ploughing functionality."""
    state = activity_state()

    # Create test setup
    time_config_mock = time_config()
    time_config_mock.min_time = 0
    time_config_mock.max_time = 10
    time_config_mock.dt = np.float64(1.0)

    converted_data_mock = converted_data()
    n_date = 11
    converted_data_mock.n_date = n_date

    # Initialize arrays
    converted_data_mock.activity_data = np.zeros(
        (constants.num_activity_index, n_date, constants.n_roads)
    )
    converted_data_mock.meteo_data = np.zeros(
        (constants.num_meteo_index, n_date, constants.n_roads)
    )
    converted_data_mock.date_data = np.zeros(
        (constants.num_date_index, n_date, constants.n_roads)
    )

    # Set date data
    converted_data_mock.date_data[constants.hour_index, :, :] = 12
    converted_data_mock.date_data[constants.month_index, :, :] = 1
    converted_data_mock.date_data[constants.datenum_index, :, :] = np.arange(n_date)[
        :, np.newaxis
    ]

    # Create model variables with snow above threshold
    model_variables_mock = model_variables()
    model_variables_mock.g_road_data = np.zeros(
        (constants.num_moisture, n_date, constants.num_track_max, constants.n_roads)
    )
    # Set snow depth above threshold (0.1) for moisture index 1 (snow)
    model_variables_mock.g_road_data[constants.snow_index, :, :, :] = 0.2

    # Enable ploughing
    flags = model_flags()
    flags.auto_ploughing_flag = 1
    flags.use_ploughing_data_flag = 1

    activities = model_activities()
    activities.delay_ploughing_hour = 2.0

    parameters = model_parameters()
    parameters.ploughing_thresh_moisture = np.ones(constants.num_moisture) * 0.1
    parameters.ploughing_thresh_moisture[constants.snow_index] = 0.1  # Snow threshold

    ti = 5
    ro = 0

    # Set initial time to meet delay requirement for first ploughing
    state.time_since_last_ploughing[ro] = 3.0  # > delay of 2.0

    # First call - should trigger ploughing since snow > threshold and delay is met
    set_activity_data_v2(
        ti=ti,
        ro=ro,
        time_config=time_config_mock,
        converted_data=converted_data_mock,
        model_variables=model_variables_mock,
        model_flags=flags,
        model_activities=activities,
        model_parameters=parameters,
        state=state,
    )

    # Should have triggered ploughing
    assert converted_data_mock.activity_data[constants.t_ploughing_index, ti, ro] == 1.0
    assert state.time_since_last_ploughing[ro] == 0.0

    # Second call immediately after - should NOT trigger ploughing (delay not met: 0.0 < 2.0)
    ti_next = 6
    set_activity_data_v2(
        ti=ti_next,
        ro=ro,
        time_config=time_config_mock,
        converted_data=converted_data_mock,
        model_variables=model_variables_mock,
        model_flags=flags,
        model_activities=activities,
        model_parameters=parameters,
        state=state,
    )

    # Should NOT trigger ploughing (delay not met) - tests the else branch
    assert (
        converted_data_mock.activity_data[constants.t_ploughing_index, ti_next, ro]
        == 0.0
    )
    assert state.time_since_last_ploughing[ro] == 1.0  # dt added


def test_set_activity_data_v2_weather_conditions():
    """Test that weather conditions properly control salting."""
    state = activity_state()

    time_config_mock = time_config()
    time_config_mock.min_time = 0
    time_config_mock.max_time = 10
    time_config_mock.dt = np.float64(1.0)

    converted_data_mock = converted_data()
    n_date = 11
    converted_data_mock.n_date = n_date

    # Initialize arrays
    converted_data_mock.activity_data = np.zeros(
        (constants.num_activity_index, n_date, constants.n_roads)
    )
    converted_data_mock.meteo_data = np.zeros(
        (constants.num_meteo_index, n_date, constants.n_roads)
    )
    converted_data_mock.date_data = np.zeros(
        (constants.num_date_index, n_date, constants.n_roads)
    )

    # Set salting hour
    converted_data_mock.date_data[constants.hour_index, :, :] = 6  # Salting hour
    converted_data_mock.date_data[constants.month_index, :, :] = 1
    converted_data_mock.date_data[constants.datenum_index, :, :] = np.arange(n_date)[
        :, np.newaxis
    ]

    # Set weather conditions that DON'T meet salting criteria
    converted_data_mock.meteo_data[constants.T_a_index, :, :] = (
        5.0  # Too warm (> max_temp_salt=0)
    )
    converted_data_mock.meteo_data[constants.Rain_precip_index, :, :] = (
        0.05  # Too little precip (< 0.1)
    )
    converted_data_mock.meteo_data[constants.Snow_precip_index, :, :] = 0.05
    converted_data_mock.meteo_data[constants.RH_index, :, :] = 80.0  # Too low RH (< 90)

    model_variables_mock = model_variables()
    model_variables_mock.g_road_data = np.zeros(
        (constants.num_moisture, n_date, constants.num_track_max, constants.n_roads)
    )

    flags = model_flags()
    flags.auto_salting_flag = 1

    activities = model_activities()
    activities.salting_hour = [6, 18]
    activities.min_temp_salt = -10.0
    activities.max_temp_salt = 0.0
    activities.precip_rule_salt = 0.1
    activities.RH_rule_salt = 90.0

    parameters = model_parameters()
    parameters.ploughing_thresh_moisture = np.ones(constants.num_moisture) * 0.1

    ti = 5
    ro = 0

    set_activity_data_v2(
        ti=ti,
        ro=ro,
        time_config=time_config_mock,
        converted_data=converted_data_mock,
        model_variables=model_variables_mock,
        model_flags=flags,
        model_activities=activities,
        model_parameters=parameters,
        state=state,
    )

    # Should NOT salt because weather conditions don't meet criteria
    assert (
        converted_data_mock.activity_data[constants.M_salting_index[0], ti, ro] == 0.0
    )
    assert (
        converted_data_mock.activity_data[constants.M_salting_index[1], ti, ro] == 0.0
    )
    assert (
        converted_data_mock.activity_data[constants.g_road_wetting_index, ti, ro] == 0.0
    )


if __name__ == "__main__":
    test_activity_state_creation()
    test_set_activity_data_v2_basic()
    test_set_activity_data_v2_ploughing()
    test_set_activity_data_v2_weather_conditions()
    print("All tests passed!")
