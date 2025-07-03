"""
Set activity data module for NORTRIP model.

This module implements automatic road maintenance activity rules based on
weather conditions and timing constraints.
"""

from dataclasses import dataclass, field
import numpy as np
import logging
import constants
from config_classes import model_flags, model_activities, model_parameters
from initialise.road_dust_initialise_variables import model_variables
from input_classes import converted_data
from initialise import time_config

logger = logging.getLogger(__name__)


@dataclass
class activity_state:
    """
    Dataclass for tracking activity state variables across time steps.

    This tracks the timing of last activities and durations since activities
    to implement delay rules for road maintenance activities.
    """

    # Last activity times (in datenum format)
    last_salting_time: np.ndarray = field(
        default_factory=lambda: np.zeros(constants.n_roads)
    )
    last_sanding_time: np.ndarray = field(
        default_factory=lambda: np.zeros(constants.n_roads)
    )
    last_binding_time: np.ndarray = field(
        default_factory=lambda: np.zeros(constants.n_roads)
    )

    # Time since last activities (in hours)
    time_since_last_ploughing: np.ndarray = field(
        default_factory=lambda: np.zeros(constants.n_roads)
    )
    time_since_last_cleaning: np.ndarray = field(
        default_factory=lambda: np.zeros(constants.n_roads)
    )


def set_activity_data(
    ti: int,
    ro: int,
    time_config: time_config,
    converted_data: converted_data,
    model_variables: model_variables,
    model_flags: model_flags,
    model_activities: model_activities,
    model_parameters: model_parameters,
    state: activity_state,
) -> None:
    """
    Determine if road maintenance activities are undertaken based on rules.

    This function implements automatic road maintenance activity rules including
    salting, sanding, ploughing, cleaning, and binding based on weather conditions,
    timing constraints, and previous activity history.

    Args:
        ti: Current time index
        ro: Current road index
        time_config: Time configuration with dt, min_time, max_time
        converted_data: Consolidated input data with activity_data, meteo_data, date_data
        model_variables: Model variables containing g_road_data
        model_flags: Model flags for auto_* activity settings
        model_activities: Activity parameters (timing, thresholds, masses)
        model_parameters: Model parameters including ploughing thresholds
        state: Activity state tracking variables

    Returns:
        None (modifies converted_data.activity_data in place)
    """

    # Internal variables
    M_salting_0 = np.zeros(constants.num_salt)
    g_road_wetting_0 = 0.0
    M_sanding_0 = 0.0
    t_ploughing_0 = 0.0
    t_cleaning_0 = 0.0

    # Get mean road moisture data from previous time step
    g_road_0_data = np.zeros(constants.num_moisture)
    if ti > time_config.min_time:
        prev_ti = max(time_config.min_time, ti - 1)
        g_road_0_data = np.mean(
            model_variables.g_road_data[: constants.num_moisture, prev_ti, :, ro],
            axis=1,
        )

    # --------------------------------------------------------------------------
    # Automatically add salt
    # --------------------------------------------------------------------------
    if model_flags.auto_salting_flag:
        if model_flags.auto_salting_flag == 1:
            M_salting_0[: constants.num_salt] = 0
            g_road_wetting_0 = 0
        elif model_flags.auto_salting_flag == 2:
            M_salting_0[: constants.num_salt] = converted_data.activity_data[
                constants.M_salting_index, ti, ro
            ]
            g_road_wetting_0 = converted_data.activity_data[
                constants.g_road_wetting_index, ti, ro
            ]

        if ti == time_config.min_time:
            state.last_salting_time[ro] = converted_data.date_data[
                constants.datenum_index, time_config.min_time, ro
            ]

        # Check temperature within range within the given delay time
        check_day = min(
            time_config.max_time,
            round(ti + time_config.dt * model_activities.check_salting_day * 24),
        )
        salt_temperature_flag = False
        for i in range(ti, check_day + 1):
            if i <= time_config.max_time:
                temp = converted_data.meteo_data[constants.T_a_index, i, ro]
                if (
                    temp > model_activities.min_temp_salt
                    and temp < model_activities.max_temp_salt
                ):
                    salt_temperature_flag = True
                    break

        # Check precipitation within range within +/- the given delay time
        check_day_min = max(
            time_config.min_time,
            round(ti - time_config.dt * model_activities.check_salting_day * 24),
        )
        check_day_max = min(
            time_config.max_time,
            round(ti + time_config.dt * model_activities.check_salting_day * 24),
        )
        salt_precip_flag = False
        for i in range(check_day_min, check_day_max + 1):
            if i <= time_config.max_time:
                rain = converted_data.meteo_data[constants.Rain_precip_index, i, ro]
                snow = converted_data.meteo_data[constants.Snow_precip_index, i, ro]
                if rain + snow > model_activities.precip_rule_salt:
                    salt_precip_flag = True
                    break

        # Check relative humidity
        check_day_min = ti
        check_day_max = min(
            time_config.max_time,
            round(ti + time_config.dt * model_activities.check_salting_day * 24),
        )
        salt_RH_flag = False
        for i in range(check_day_min, check_day_max + 1):
            if i <= time_config.max_time:
                RH = converted_data.meteo_data[constants.RH_index, i, ro]
                if RH > model_activities.RH_rule_salt:
                    salt_RH_flag = True
                    break

        # Check if salting should occur
        current_hour = converted_data.date_data[constants.hour_index, ti, ro]
        time_since_salting = (
            converted_data.date_data[constants.datenum_index, ti, ro]
            - state.last_salting_time[ro]
        )

        hour_condition = (
            current_hour == model_activities.salting_hour[0]
            or current_hour == model_activities.salting_hour[1]
        )
        weather_condition = salt_temperature_flag and (salt_precip_flag or salt_RH_flag)
        delay_condition = time_since_salting >= model_activities.delay_salting_day

        if hour_condition and weather_condition and delay_condition:
            # Apply salting
            salt_mass_1 = (
                model_activities.salt_mass * model_activities.salt_type_distribution
            )
            salt_mass_2 = model_activities.salt_mass * (
                1 - model_activities.salt_type_distribution
            )

            converted_data.activity_data[constants.M_salting_index[0], ti, ro] = (
                M_salting_0[0] + salt_mass_1
            )
            converted_data.activity_data[constants.M_salting_index[1], ti, ro] = (
                M_salting_0[1] + salt_mass_2
            )

            state.last_salting_time[ro] = converted_data.date_data[
                constants.datenum_index, ti, ro
            ]

            # Add road wetting if surface is dry enough
            if (
                np.sum(g_road_0_data[: constants.num_moisture])
                < model_activities.g_salting_rule
                and model_activities.salt_dilution != 0
            ):
                wetting_amount = (
                    model_activities.salt_mass
                    * (1 - model_activities.salt_dilution)
                    / model_activities.salt_dilution
                    * 1e-3
                )
                converted_data.activity_data[constants.g_road_wetting_index, ti, ro] = (
                    g_road_wetting_0 + wetting_amount
                )
            else:
                converted_data.activity_data[constants.g_road_wetting_index, ti, ro] = (
                    g_road_wetting_0
                )
        else:
            # No salting
            converted_data.activity_data[constants.M_salting_index[0], ti, ro] = (
                M_salting_0[0]
            )
            converted_data.activity_data[constants.M_salting_index[1], ti, ro] = (
                M_salting_0[1]
            )
            converted_data.activity_data[constants.g_road_wetting_index, ti, ro] = (
                g_road_wetting_0
            )

    # --------------------------------------------------------------------------
    # Automatically add sand
    # --------------------------------------------------------------------------
    if model_flags.auto_sanding_flag:
        if model_flags.auto_sanding_flag == 1:
            M_sanding_0 = 0
            g_road_wetting_0 = converted_data.activity_data[
                constants.g_road_wetting_index, ti, ro
            ]
        elif model_flags.auto_sanding_flag == 2:
            M_sanding_0 = converted_data.activity_data[
                constants.M_sanding_index, ti, ro
            ]
            g_road_wetting_0 = converted_data.activity_data[
                constants.g_road_wetting_index, ti, ro
            ]

        if ti == time_config.min_time:
            state.last_sanding_time[ro] = converted_data.date_data[
                constants.datenum_index, time_config.min_time, ro
            ]

        # Check temperature within range within the given delay time
        check_day = min(
            time_config.max_time,
            round(ti + time_config.dt * model_activities.check_sanding_day * 24),
        )
        sand_temperature_flag = False
        for i in range(ti, check_day + 1):
            if i <= time_config.max_time:
                temp = converted_data.meteo_data[constants.T_a_index, i, ro]
                if (
                    temp > model_activities.min_temp_sand
                    and temp < model_activities.max_temp_sand
                ):
                    sand_temperature_flag = True
                    break

        # Check precipitation within range
        check_day_min = max(
            time_config.min_time,
            round(ti - time_config.dt * model_activities.check_sanding_day * 24),
        )
        check_day_max = min(
            time_config.max_time,
            round(ti + time_config.dt * model_activities.check_sanding_day * 24),
        )
        sand_precip_flag = False
        for i in range(check_day_min, check_day_max + 1):
            if i <= time_config.max_time:
                rain = converted_data.meteo_data[constants.Rain_precip_index, i, ro]
                snow = converted_data.meteo_data[constants.Snow_precip_index, i, ro]
                if rain + snow > model_activities.precip_rule_sand:
                    sand_precip_flag = True
                    break

        # Check relative humidity
        check_day_min = ti
        check_day_max = min(
            time_config.max_time,
            round(ti + time_config.dt * model_activities.check_sanding_day * 24),
        )
        sand_RH_flag = False
        for i in range(check_day_min, check_day_max + 1):
            if i <= time_config.max_time:
                RH = converted_data.meteo_data[constants.RH_index, i, ro]
                if RH > model_activities.RH_rule_sand:
                    sand_RH_flag = True
                    break

        # Check if sanding should occur
        current_hour = converted_data.date_data[constants.hour_index, ti, ro]
        time_since_sanding = (
            converted_data.date_data[constants.datenum_index, ti, ro]
            - state.last_sanding_time[ro]
        )

        hour_condition = (
            current_hour == model_activities.sanding_hour[0]
            or current_hour == model_activities.sanding_hour[1]
        )
        weather_condition = sand_temperature_flag and (sand_precip_flag or sand_RH_flag)
        delay_condition = time_since_sanding >= model_activities.delay_sanding_day

        if hour_condition and weather_condition and delay_condition:
            # Apply sanding
            converted_data.activity_data[constants.M_sanding_index, ti, ro] = (
                M_sanding_0 + model_activities.sand_mass
            )

            state.last_sanding_time[ro] = converted_data.date_data[
                constants.datenum_index, ti, ro
            ]

            # Add road wetting if snow/ice conditions and dilution is set
            snow_ice_sum = np.sum(g_road_0_data[constants.snow_ice_index])
            if (
                snow_ice_sum > model_activities.g_sanding_rule
                and model_activities.sand_dilution != 0
            ):
                wetting_amount = (
                    model_activities.sand_mass / model_activities.sand_dilution * 1e-3
                )
                converted_data.activity_data[constants.g_road_wetting_index, ti, ro] = (
                    g_road_wetting_0 + wetting_amount
                )
            else:
                converted_data.activity_data[constants.g_road_wetting_index, ti, ro] = (
                    g_road_wetting_0
                )
        else:
            # No sanding
            converted_data.activity_data[constants.M_sanding_index, ti, ro] = 0
            converted_data.activity_data[constants.g_road_wetting_index, ti, ro] = (
                g_road_wetting_0
            )

    # --------------------------------------------------------------------------
    # Automatically carry out ploughing based on previous hours
    # --------------------------------------------------------------------------
    if model_flags.auto_ploughing_flag and model_flags.use_ploughing_data_flag:
        if model_flags.auto_ploughing_flag == 1:
            t_ploughing_0 = 0
        elif model_flags.auto_ploughing_flag == 2:
            t_ploughing_0 = converted_data.activity_data[
                constants.t_ploughing_index, ti, ro
            ]

        if ti == time_config.min_time:
            state.time_since_last_ploughing[ro] = 0

        # Check moisture conditions for ploughing
        prev_ti = max(time_config.min_time, ti - 1)
        plough_temp = np.mean(
            model_variables.g_road_data[: constants.num_moisture, prev_ti, :, ro],
            axis=1,
        )

        plough_moisture_flag = False
        for m in range(constants.num_moisture):
            if plough_temp[m] > model_parameters.ploughing_thresh_moisture[m]:
                plough_moisture_flag = True
                break

        # Check if ploughing should occur
        if (
            plough_moisture_flag
            and state.time_since_last_ploughing[ro]
            >= model_activities.delay_ploughing_hour
        ):
            converted_data.activity_data[constants.t_ploughing_index, ti, ro] = (
                t_ploughing_0 + 1
            )
            state.time_since_last_ploughing[ro] = 0
        else:
            converted_data.activity_data[constants.t_ploughing_index, ti, ro] = (
                t_ploughing_0
            )
            state.time_since_last_ploughing[ro] += time_config.dt

    # --------------------------------------------------------------------------
    # Automatically carry out cleaning based on salting activity
    # --------------------------------------------------------------------------
    if model_flags.auto_cleaning_flag and model_flags.use_cleaning_data_flag:
        if model_flags.auto_cleaning_flag == 1:
            t_cleaning_0 = 0
            g_road_wetting_0 = converted_data.activity_data[
                constants.g_road_wetting_index, ti, ro
            ]
        elif model_flags.auto_cleaning_flag == 2:
            t_cleaning_0 = converted_data.activity_data[
                constants.t_cleaning_index, ti, ro
            ]
            g_road_wetting_0 = converted_data.activity_data[
                constants.g_road_wetting_index, ti, ro
            ]

        if ti == time_config.min_time:
            state.time_since_last_cleaning[ro] = 0

        # Check if cleaning is allowed based on salting
        cleaning_allowed = 1
        if model_activities.clean_with_salting:
            salt_sum = np.sum(
                converted_data.activity_data[constants.M_salting_index, ti, ro]
            )
            if salt_sum > 0:
                cleaning_allowed = 1
            else:
                cleaning_allowed = 0

        # Check month
        current_month = converted_data.date_data[constants.month_index, ti, ro]
        if model_activities.start_month_cleaning <= model_activities.end_month_cleaning:
            if not (
                model_activities.start_month_cleaning
                <= current_month
                <= model_activities.end_month_cleaning
            ):
                cleaning_allowed = 0
        else:
            if not (
                current_month >= model_activities.start_month_cleaning
                or current_month <= model_activities.end_month_cleaning
            ):
                cleaning_allowed = 0

        # Check if cleaning should occur
        temp_condition = (
            converted_data.meteo_data[constants.T_a_index, ti, ro]
            > model_activities.min_temp_cleaning
        )
        time_condition = (
            state.time_since_last_cleaning[ro] >= model_activities.delay_cleaning_hour
        )

        if time_condition and temp_condition and cleaning_allowed:
            converted_data.activity_data[constants.t_cleaning_index, ti, ro] = (
                t_cleaning_0 + model_activities.efficiency_of_cleaning
            )
            state.time_since_last_cleaning[ro] = 0
            converted_data.activity_data[constants.g_road_wetting_index, ti, ro] = (
                g_road_wetting_0 + model_activities.wetting_with_cleaning
            )
        else:
            converted_data.activity_data[constants.t_cleaning_index, ti, ro] = (
                t_cleaning_0
            )
            state.time_since_last_cleaning[ro] += time_config.dt
            converted_data.activity_data[constants.g_road_wetting_index, ti, ro] = (
                g_road_wetting_0
            )

    # --------------------------------------------------------------------------
    # Automatically add second salt for binding
    # --------------------------------------------------------------------------
    if model_flags.auto_binding_flag:
        if model_flags.auto_binding_flag == 1:
            M_salting_0[1] = 0
            g_road_wetting_0 = 0
        elif model_flags.auto_binding_flag == 2:
            M_salting_0[1] = converted_data.activity_data[
                constants.M_salting_index[1], ti, ro
            ]
            g_road_wetting_0 = converted_data.activity_data[
                constants.g_road_wetting_index, ti, ro
            ]

        if ti == time_config.min_time:
            state.last_binding_time[ro] = converted_data.date_data[
                constants.datenum_index, time_config.min_time, ro
            ]

        # Start with no binding allowed
        binding_allowed = 0

        # Check temperature within range within the given delay time
        check_day = min(
            time_config.max_time,
            round(ti + time_config.dt * model_activities.check_binding_day * 24),
        )
        for i in range(ti, check_day + 1):
            if i <= time_config.max_time:
                temp = converted_data.meteo_data[constants.T_a_index, i, ro]
                if (
                    temp > model_activities.min_temp_binding
                    and temp < model_activities.max_temp_binding
                ):
                    binding_allowed = 1
                    break

        # Check precipitation within range (disallow if too much precip)
        check_day_min = max(
            time_config.min_time,
            round(ti - time_config.dt * model_activities.check_binding_day * 24),
        )
        check_day_max = min(
            time_config.max_time,
            round(ti + time_config.dt * model_activities.check_binding_day * 24),
        )
        for i in range(check_day_min, check_day_max + 1):
            if i <= time_config.max_time:
                rain = converted_data.meteo_data[constants.Rain_precip_index, i, ro]
                snow = converted_data.meteo_data[constants.Snow_precip_index, i, ro]
                if rain + snow > model_activities.precip_rule_binding:
                    binding_allowed = 0
                    break

        # Check month
        current_month = converted_data.date_data[constants.month_index, ti, ro]
        if model_activities.start_month_binding <= model_activities.end_month_binding:
            if not (
                model_activities.start_month_binding
                <= current_month
                <= model_activities.end_month_binding
            ):
                binding_allowed = 0
        else:
            if not (
                current_month >= model_activities.start_month_binding
                or current_month <= model_activities.end_month_binding
            ):
                binding_allowed = 0

        # Check current surface conditions
        if (
            np.sum(g_road_0_data[: constants.num_moisture])
            > model_activities.g_binding_rule
        ):
            binding_allowed = 0

        # Check relative humidity
        check_day_min = ti
        check_day_max = min(
            time_config.max_time,
            round(ti + time_config.dt * model_activities.check_binding_day * 24),
        )
        binding_RH_flag = False
        for i in range(check_day_min, check_day_max + 1):
            if i <= time_config.max_time:
                RH = converted_data.meteo_data[constants.RH_index, i, ro]
                if RH > model_activities.RH_rule_binding:
                    binding_RH_flag = True
                    break

        # Check if binding should occur
        current_hour = converted_data.date_data[constants.hour_index, ti, ro]
        time_since_binding = (
            converted_data.date_data[constants.datenum_index, ti, ro]
            - state.last_binding_time[ro]
        )

        hour_condition = (
            current_hour == model_activities.binding_hour[0]
            or current_hour == model_activities.binding_hour[1]
        )
        delay_condition = time_since_binding >= model_activities.delay_binding_day

        if hour_condition and delay_condition and binding_RH_flag and binding_allowed:
            # Apply binding
            converted_data.activity_data[constants.M_salting_index[1], ti, ro] = (
                M_salting_0[1] + model_activities.binding_mass
            )

            state.last_binding_time[ro] = converted_data.date_data[
                constants.datenum_index, ti, ro
            ]

            # Add road wetting if dilution is set
            if model_activities.binding_dilution != 0:
                wetting_amount = (
                    model_activities.binding_mass
                    * (1 - model_activities.binding_dilution)
                    / model_activities.binding_dilution
                    * 1e-3
                )
                converted_data.activity_data[constants.g_road_wetting_index, ti, ro] = (
                    g_road_wetting_0 + wetting_amount
                )
            else:
                converted_data.activity_data[constants.g_road_wetting_index, ti, ro] = (
                    g_road_wetting_0
                )
        else:
            # No binding
            converted_data.activity_data[constants.M_salting_index[1], ti, ro] = (
                M_salting_0[1]
            )
            converted_data.activity_data[constants.g_road_wetting_index, ti, ro] = (
                g_road_wetting_0
            )
