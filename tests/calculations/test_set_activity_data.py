"""
Tests for set_activity_data module.
"""

import numpy as np
from initialise import time_config, model_variables
from calculations import activity_state, set_activity_data
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


def test_set_activity_data_basic():
    """Test basic functionality of set_activity_data function."""
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
    set_activity_data(
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

    set_activity_data(
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


def test_set_activity_data_ploughing():
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
    set_activity_data(
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
    set_activity_data(
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


def test_set_activity_data_weather_conditions():
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

    set_activity_data(
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


def test_set_activity_data_sanding():
    """Test sanding functionality."""
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

    # Set date data with sanding hour
    converted_data_mock.date_data[constants.hour_index, :, :] = 7  # Sanding hour
    converted_data_mock.date_data[constants.month_index, :, :] = 1
    converted_data_mock.date_data[constants.datenum_index, :, :] = np.arange(n_date)[
        :, np.newaxis
    ]

    # Set weather conditions that meet sanding criteria
    converted_data_mock.meteo_data[constants.T_a_index, :, :] = -2.0  # Cold enough
    converted_data_mock.meteo_data[constants.Rain_precip_index, :, :] = (
        0.2  # Enough precip
    )
    converted_data_mock.meteo_data[constants.Snow_precip_index, :, :] = 0.2
    converted_data_mock.meteo_data[constants.RH_index, :, :] = 95.0  # High humidity

    model_variables_mock = model_variables()
    model_variables_mock.g_road_data = np.zeros(
        (constants.num_moisture, n_date, constants.num_track_max, constants.n_roads)
    )
    # Set snow/ice conditions for sanding
    model_variables_mock.g_road_data[constants.snow_index, :, :, :] = 0.2
    model_variables_mock.g_road_data[constants.ice_index, :, :, :] = 0.1

    flags = model_flags()
    flags.auto_sanding_flag = 1

    activities = model_activities()
    activities.sanding_hour = [7, 19]  # Allow sanding at 7 and 19
    activities.min_temp_sand = -10.0
    activities.max_temp_sand = 0.0
    activities.precip_rule_sand = 0.1
    activities.RH_rule_sand = 90.0
    activities.g_sanding_rule = 0.1
    activities.sand_mass = 0.15
    activities.sand_dilution = 0.3
    activities.delay_sanding_day = 0.5

    parameters = model_parameters()
    parameters.ploughing_thresh_moisture = np.ones(constants.num_moisture) * 0.1

    ti = 5
    ro = 0

    set_activity_data(
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

    # Should trigger sanding
    assert (
        converted_data_mock.activity_data[constants.M_sanding_index, ti, ro]
        == activities.sand_mass
    )
    assert state.last_sanding_time[ro] == ti

    # Check wetting occurs due to snow/ice conditions
    expected_wetting = activities.sand_mass / activities.sand_dilution * 1e-3
    assert (
        converted_data_mock.activity_data[constants.g_road_wetting_index, ti, ro]
        == expected_wetting
    )


def test_set_activity_data_cleaning():
    """Test cleaning functionality."""
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

    # Set date data
    converted_data_mock.date_data[constants.hour_index, :, :] = 10
    converted_data_mock.date_data[constants.month_index, :, :] = (
        6  # June (cleaning season)
    )
    converted_data_mock.date_data[constants.datenum_index, :, :] = np.arange(n_date)[
        :, np.newaxis
    ]

    # Set weather conditions that meet cleaning criteria
    converted_data_mock.meteo_data[constants.T_a_index, :, :] = 15.0  # Warm enough

    model_variables_mock = model_variables()
    model_variables_mock.g_road_data = np.zeros(
        (constants.num_moisture, n_date, constants.num_track_max, constants.n_roads)
    )

    flags = model_flags()
    flags.auto_cleaning_flag = 1
    flags.use_cleaning_data_flag = 1

    activities = model_activities()
    activities.delay_cleaning_hour = 168.0  # 7 days
    activities.min_temp_cleaning = 5.0
    activities.start_month_cleaning = 4  # April
    activities.end_month_cleaning = 10  # October
    activities.efficiency_of_cleaning = 0.8
    activities.wetting_with_cleaning = 0.5
    activities.clean_with_salting = 0

    parameters = model_parameters()
    parameters.ploughing_thresh_moisture = np.ones(constants.num_moisture) * 0.1

    ti = 5
    ro = 0

    # Set time since last cleaning to meet delay requirement
    state.time_since_last_cleaning[ro] = 200.0  # > 168 hours

    set_activity_data(
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

    # Should trigger cleaning
    assert (
        converted_data_mock.activity_data[constants.t_cleaning_index, ti, ro]
        == activities.efficiency_of_cleaning
    )
    assert state.time_since_last_cleaning[ro] == 0.0

    # Check wetting from cleaning
    assert (
        converted_data_mock.activity_data[constants.g_road_wetting_index, ti, ro]
        == activities.wetting_with_cleaning
    )


def test_set_activity_data_binding():
    """Test binding functionality."""
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

    # Set date data with binding hour
    converted_data_mock.date_data[constants.hour_index, :, :] = 8  # Binding hour
    converted_data_mock.date_data[constants.month_index, :, :] = (
        5  # May (binding season)
    )
    converted_data_mock.date_data[constants.datenum_index, :, :] = np.arange(n_date)[
        :, np.newaxis
    ]

    # Set weather conditions that meet binding criteria
    converted_data_mock.meteo_data[constants.T_a_index, :, :] = -1.0  # Right temp range
    converted_data_mock.meteo_data[constants.Rain_precip_index, :, :] = (
        0.05  # Low precip
    )
    converted_data_mock.meteo_data[constants.Snow_precip_index, :, :] = 0.05
    converted_data_mock.meteo_data[constants.RH_index, :, :] = 95.0  # High humidity

    model_variables_mock = model_variables()
    model_variables_mock.g_road_data = np.zeros(
        (constants.num_moisture, n_date, constants.num_track_max, constants.n_roads)
    )
    # Set very dry surface conditions for binding (total moisture sum < g_binding_rule=0.1)
    model_variables_mock.g_road_data[: constants.num_moisture, :, :, :] = 0.01

    flags = model_flags()
    flags.auto_binding_flag = 1

    activities = model_activities()
    activities.binding_hour = [8, 20]  # Allow binding at 8 and 20
    activities.min_temp_binding = -5.0
    activities.max_temp_binding = 2.0
    activities.precip_rule_binding = 0.1
    activities.RH_rule_binding = 90.0
    activities.g_binding_rule = 0.1
    activities.binding_mass = 0.12
    activities.binding_dilution = 0.25
    activities.delay_binding_day = 0.5
    activities.start_month_binding = 3  # March
    activities.end_month_binding = 9  # September
    activities.check_binding_day = 0.5

    parameters = model_parameters()
    parameters.ploughing_thresh_moisture = np.ones(constants.num_moisture) * 0.1

    ti = 5
    ro = 0

    set_activity_data(
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

    # Should trigger binding
    assert (
        converted_data_mock.activity_data[constants.M_salting_index[1], ti, ro]
        == activities.binding_mass
    )
    assert state.last_binding_time[ro] == ti

    # Check wetting from binding
    expected_wetting = (
        activities.binding_mass
        * (1 - activities.binding_dilution)
        / activities.binding_dilution
        * 1e-3
    )
    assert (
        converted_data_mock.activity_data[constants.g_road_wetting_index, ti, ro]
        == expected_wetting
    )


def test_set_activity_data_salting_mode_2():
    """Test salting with auto_salting_flag = 2 (builds on existing data)."""
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

    # Set existing salting data
    ti = 5
    ro = 0
    converted_data_mock.activity_data[constants.M_salting_index[0], ti, ro] = 0.05
    converted_data_mock.activity_data[constants.M_salting_index[1], ti, ro] = 0.03
    converted_data_mock.activity_data[constants.g_road_wetting_index, ti, ro] = 0.002

    # Set date data with salting hour
    converted_data_mock.date_data[constants.hour_index, :, :] = 6  # Salting hour
    converted_data_mock.date_data[constants.month_index, :, :] = 1
    converted_data_mock.date_data[constants.datenum_index, :, :] = np.arange(n_date)[
        :, np.newaxis
    ]

    # Set weather conditions that meet salting criteria
    converted_data_mock.meteo_data[constants.T_a_index, :, :] = -3.0
    converted_data_mock.meteo_data[constants.Rain_precip_index, :, :] = 0.2
    converted_data_mock.meteo_data[constants.Snow_precip_index, :, :] = 0.2
    converted_data_mock.meteo_data[constants.RH_index, :, :] = 95.0

    model_variables_mock = model_variables()
    model_variables_mock.g_road_data = np.zeros(
        (constants.num_moisture, n_date, constants.num_track_max, constants.n_roads)
    )

    flags = model_flags()
    flags.auto_salting_flag = 2  # Mode 2: build on existing data

    activities = model_activities()
    activities.salting_hour = [6, 18]
    activities.min_temp_salt = -10.0
    activities.max_temp_salt = 0.0
    activities.precip_rule_salt = 0.1
    activities.RH_rule_salt = 90.0
    activities.salt_mass = 0.1
    activities.salt_type_distribution = 1.0
    activities.delay_salting_day = 0.5

    parameters = model_parameters()
    parameters.ploughing_thresh_moisture = np.ones(constants.num_moisture) * 0.1

    set_activity_data(
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

    # Should add to existing salting data
    expected_salt_1 = 0.05 + activities.salt_mass * activities.salt_type_distribution
    expected_salt_2 = 0.03 + activities.salt_mass * (
        1 - activities.salt_type_distribution
    )

    assert (
        converted_data_mock.activity_data[constants.M_salting_index[0], ti, ro]
        == expected_salt_1
    )
    assert (
        converted_data_mock.activity_data[constants.M_salting_index[1], ti, ro]
        == expected_salt_2
    )


def test_set_activity_data_initialization_at_min_time():
    """Test initialization logic when ti == time_config.min_time."""
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

    # Set date data
    converted_data_mock.date_data[constants.hour_index, :, :] = 12
    converted_data_mock.date_data[constants.month_index, :, :] = 1
    converted_data_mock.date_data[constants.datenum_index, :, :] = np.arange(n_date)[
        :, np.newaxis
    ]

    model_variables_mock = model_variables()
    model_variables_mock.g_road_data = np.zeros(
        (constants.num_moisture, n_date, constants.num_track_max, constants.n_roads)
    )

    flags = model_flags()
    flags.auto_salting_flag = 1
    flags.auto_sanding_flag = 1
    flags.auto_binding_flag = 1
    flags.auto_cleaning_flag = 1
    flags.auto_ploughing_flag = 1
    flags.use_cleaning_data_flag = 1
    flags.use_ploughing_data_flag = 1

    activities = model_activities()
    parameters = model_parameters()
    parameters.ploughing_thresh_moisture = np.ones(constants.num_moisture) * 0.1

    ti = 0  # min_time
    ro = 0

    # Store initial datenum for comparison
    initial_datenum = converted_data_mock.date_data[constants.datenum_index, ti, ro]

    set_activity_data(
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

    # Should initialize timestamps at min_time
    assert state.last_salting_time[ro] == initial_datenum
    assert state.last_sanding_time[ro] == initial_datenum
    assert state.last_binding_time[ro] == initial_datenum
    # Note: ploughing timer gets incremented by dt even at initialization if no ploughing occurs
    assert state.time_since_last_ploughing[ro] == time_config_mock.dt
    assert state.time_since_last_cleaning[ro] == time_config_mock.dt
