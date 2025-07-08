from dataclasses import dataclass, field
import numpy as np
import logging
import constants
from initialise import time_config
from config_classes import model_parameters, model_flags
from input_classes import (
    input_metadata,
    input_airquality,
    converted_data,
    input_initial,
)

logger = logging.getLogger(__name__)


@dataclass
class model_variables:
    """
    Dataclass containing all model variables and arrays for NORTRIP execution.
    """

    # Forecast missing data array
    road_temperature_forecast_missing: list = field(default_factory=list)

    # Main model data arrays
    M_road_data: np.ndarray = field(default_factory=lambda: np.array([]))
    M_road_bin_data: np.ndarray = field(default_factory=lambda: np.array([]))
    M_road_bin_balance_data: np.ndarray = field(default_factory=lambda: np.array([]))
    M_road_balance_data: np.ndarray = field(default_factory=lambda: np.array([]))
    WR_time_data: np.ndarray = field(default_factory=lambda: np.array([]))
    road_salt_data: np.ndarray = field(default_factory=lambda: np.array([]))
    C_bin_data: np.ndarray = field(default_factory=lambda: np.array([]))
    C_data: np.ndarray = field(default_factory=lambda: np.array([]))
    E_road_data: np.ndarray = field(default_factory=lambda: np.array([]))
    E_road_bin_data: np.ndarray = field(default_factory=lambda: np.array([]))
    road_meteo_data: np.ndarray = field(default_factory=lambda: np.array([]))
    g_road_balance_data: np.ndarray = field(default_factory=lambda: np.array([]))
    g_road_data: np.ndarray = field(default_factory=lambda: np.array([]))

    # Forecast and correction arrays
    forecast_hours: np.ndarray = field(default_factory=lambda: np.array([]))
    E_corr_array: np.ndarray = field(default_factory=lambda: np.array([]))

    # Quality factor arrays
    f_q: np.ndarray = field(default_factory=lambda: np.array([]))
    f_q_obs: np.ndarray = field(default_factory=lambda: np.array([]))

    # Initial mass data
    M_road_0_data: np.ndarray = field(default_factory=lambda: np.array([]))


def road_dust_initialise_variables(
    time_config: time_config,
    converted_data: converted_data,
    initial_data: input_initial,
    metadata: input_metadata,
    airquality_data: input_airquality,
    model_parameters: model_parameters,
    model_flags: model_flags,
) -> model_variables:
    """
    Initialize all model variables and arrays for NORTRIP execution.

    Args:
        time_config: Time configuration with min_time, max_time, dt, etc.
        converted_data: Consolidated input data arrays
        initial_data: Initial conditions data
        metadata: Metadata including nodata value
        airquality_data: Air quality input data
        model_parameters: Model parameters including f_PM array
        model_flags: Model flags including exhaust_flag and forecast_hour

    Returns:
        model_variables: Dataclass containing all initialized arrays
    """

    logger.info("Initialising model variables...")

    # Extract commonly used values
    min_time = time_config.min_time
    max_time = time_config.max_time
    n_date = converted_data.n_date
    n_roads = converted_data.n_roads
    nodata = metadata.nodata
    forecast_hour = model_flags.forecast_hour
    dt = time_config.dt
    exhaust_flag = model_flags.exhaust_flag

    # Initialize model variables dataclass
    variables = model_variables()

    # Initialize forecast missing if forecast mode is used
    variables.road_temperature_forecast_missing = []

    # Initialize all main arrays
    variables.M_road_data = np.zeros(
        (
            constants.num_source_all,
            constants.num_size,
            n_date,
            constants.num_track_max,
            n_roads,
        )
    )

    variables.M_road_bin_data = np.zeros(
        (
            constants.num_source_all,
            constants.num_size,
            n_date,
            constants.num_track_max,
            n_roads,
        )
    )

    variables.M_road_bin_balance_data = np.zeros(
        (
            constants.num_source_all,
            constants.num_size,
            constants.num_dustbalance,
            n_date,
            constants.num_track_max,
            n_roads,
        )
    )

    variables.M_road_balance_data = np.zeros(
        (
            constants.num_source_all,
            constants.num_size,
            constants.num_dustbalance,
            n_date,
            constants.num_track_max,
            n_roads,
        )
    )

    variables.WR_time_data = np.zeros(
        (constants.num_wear, n_date, constants.num_track_max, n_roads)
    )

    variables.road_salt_data = np.zeros(
        (
            constants.num_saltdata,
            constants.num_salt,
            n_date,
            constants.num_track_max,
            n_roads,
        )
    )

    variables.C_bin_data = np.zeros(
        (
            constants.num_source_all,
            constants.num_size,
            constants.num_process,
            n_date,
            constants.num_track_max,
            n_roads,
        )
    )

    variables.C_data = np.zeros(
        (
            constants.num_source_all,
            constants.num_size,
            constants.num_process,
            n_date,
            constants.num_track_max,
            n_roads,
        )
    )

    variables.E_road_data = np.zeros(
        (
            constants.num_source_all,
            constants.num_size,
            constants.num_process,
            n_date,
            constants.num_track_max,
            n_roads,
        )
    )

    variables.E_road_bin_data = np.zeros(
        (
            constants.num_source_all,
            constants.num_size,
            constants.num_process,
            n_date,
            constants.num_track_max,
            n_roads,
        )
    )

    variables.road_meteo_data = np.zeros(
        (constants.num_road_meteo, n_date, constants.num_track_max, n_roads)
    )

    variables.g_road_balance_data = np.zeros(
        (
            constants.num_moisture,
            constants.num_moistbalance,
            n_date,
            constants.num_track_max,
            n_roads,
        )
    )

    # Initialize forecast arrays
    if forecast_hour > 0:
        forecast_steps = int(forecast_hour / dt) + 1
        variables.forecast_hours = np.zeros((forecast_steps, n_date))
        variables.E_corr_array = np.zeros((int(forecast_hour / dt), n_date))
    else:
        variables.forecast_hours = np.zeros((1, n_date))
        variables.E_corr_array = np.zeros((1, n_date))

    # NOTE: +2 in g_road_data dimension (as in MATLAB comment "WHY IS THIS +2 ???")
    variables.g_road_data = np.zeros(
        (constants.num_moisture + 2, n_date, constants.num_track_max, n_roads)
    )

    # Initialize quality factor arrays
    variables.f_q = np.ones(
        (constants.num_source_all, n_date, constants.num_track_max, n_roads)
    )

    variables.f_q_obs = np.full((n_date, constants.num_track_max, n_roads), nodata)

    # Initialize specific mass balance data indices to zero (from min_time to max_time)
    time_slice = slice(min_time, max_time + 1)

    # Mass balance data - set specific indices to zero
    balance_indices = [
        constants.S_total_index,
        constants.P_total_index,
        constants.P_wear_index,
        constants.S_dustspray_index,
        constants.P_dustspray_index,
        constants.S_dustdrainage_index,
        constants.S_windblown_index,
        constants.S_suspension_index,
        constants.S_cleaning_index,
        constants.S_dustploughing_index,
        constants.P_crushing_index,
        constants.S_crushing_index,
        constants.P_abrasion_index,
        constants.P_depo_index,
    ]

    for balance_idx in balance_indices:
        variables.M_road_bin_balance_data[
            : constants.num_source_all,
            : constants.num_size,
            balance_idx,
            time_slice,
            : constants.num_track_max,
            :n_roads,
        ] = 0

    # Emission data initialization
    emission_indices = [
        constants.E_direct_index,
        constants.E_suspension_index,
        constants.E_windblown_index,
        constants.E_total_index,
    ]

    for emis_idx in emission_indices:
        variables.E_road_data[
            : constants.num_source_all,
            : constants.num_size,
            emis_idx,
            time_slice,
            : constants.num_track_max,
            :n_roads,
        ] = 0
        variables.E_road_bin_data[
            : constants.num_source_all,
            : constants.num_size,
            emis_idx,
            time_slice,
            : constants.num_track_max,
            :n_roads,
        ] = 0

    # Concentration data initialization
    concentration_indices = [
        constants.C_direct_index,
        constants.C_suspension_index,
        constants.C_windblown_index,
        constants.C_total_index,
    ]

    for conc_idx in concentration_indices:
        variables.C_data[
            : constants.num_source_all,
            : constants.num_size,
            conc_idx,
            time_slice,
            : constants.num_track_max,
            :n_roads,
        ] = 0

    # Road salt data initialization
    salt_indices = [
        constants.RH_salt_index,
        constants.melt_temperature_salt_index,
        constants.dissolved_ratio_index,
    ]

    for salt_idx in salt_indices:
        variables.road_salt_data[
            salt_idx,
            : constants.num_salt,
            time_slice,
            : constants.num_track_max,
            :n_roads,
        ] = 0

    # Road meteorological data initialization
    meteo_indices = [
        constants.T_s_index,
        constants.T_melt_index,
        constants.r_aero_index,
        constants.r_aero_notraffic_index,
        constants.RH_s_index,
        constants.RH_salt_final_index,
        constants.L_index,
        constants.H_index,
        constants.G_index,
        constants.G_sub_index,
        constants.G_freeze_index,
        constants.G_melt_index,
        constants.evap_index,
        constants.evap_pot_index,
        constants.rad_net_index,
        constants.short_rad_net_index,
        constants.short_rad_net_clearsky_index,
        constants.long_rad_net_index,
        constants.long_rad_out_index,
        constants.H_traffic_index,
        constants.T_sub_index,
        constants.road_temperature_obs_index,
        constants.road_wetness_obs_index,
        constants.E_index,
        constants.E_correction_index,
    ]

    for meteo_idx in meteo_indices:
        variables.road_meteo_data[
            meteo_idx, time_slice, : constants.num_track_max, :n_roads
        ] = 0

    # Road moisture mass balance data initialization
    moisture_balance_indices = [
        constants.S_melt_index,
        constants.P_melt_index,
        constants.P_freeze_index,
        constants.P_evap_index,
        constants.S_evap_index,
        constants.S_drainage_index,
        constants.S_spray_index,
        constants.R_spray_index,
        constants.P_spray_index,
        constants.S_total_index,
        constants.P_total_index,
        constants.P_precip_index,
        constants.P_roadwetting_index,
    ]

    for moisture_idx in moisture_balance_indices:
        variables.g_road_balance_data[
            : constants.num_moisture,
            moisture_idx,
            time_slice,
            : constants.num_track_max,
            :n_roads,
        ] = 0

    # Road moisture data initialization
    variables.g_road_data[
        : constants.num_moisture, time_slice, : constants.num_track_max, :n_roads
    ] = 0

    # Set initial mass loading values
    ti = min_time
    t = constants.su  # summer tire type

    for ro in range(n_roads):
        for tr in range(model_parameters.num_track):
            for s in range(constants.num_source):
                for x in range(constants.num_size):
                    variables.M_road_data[s, x, ti, tr, ro] = (
                        initial_data.m_road_init[s, tr] * model_parameters.f_PM[s, x, t]
                    )

    # Set initial surface moisture
    for ro in range(n_roads):
        for tr in range(model_parameters.num_track):
            for m in range(constants.num_moisture):
                variables.g_road_data[m, ti, tr, ro] = initial_data.g_road_init[m, tr]

    # Initialize surface temperature and humidity
    for ro in range(n_roads):
        for tr in range(model_parameters.num_track):
            variables.road_meteo_data[constants.T_s_index, ti, tr, ro] = (
                converted_data.meteo_data[constants.T_a_index, ti, ro]
            )
            variables.road_meteo_data[constants.RH_s_index, ti, tr, ro] = (
                converted_data.meteo_data[constants.RH_index, ti, ro]
            )
            variables.road_meteo_data[constants.T_sub_index, ti, tr, ro] = (
                converted_data.meteo_data[constants.T_a_index, ti, ro]
            )

    # Initialize M_road_0_data
    variables.M_road_0_data = np.full(
        (constants.num_source, constants.num_size), nodata
    )
    for x in range(constants.num_size):
        variables.M_road_0_data[: constants.num_source, x] = variables.M_road_data[
            : constants.num_source,
            x,
            max(min_time, ti - 1),
            model_parameters.num_track - 1,
            n_roads - 1,
        ]

    # Convert initial road mass to binned road mass
    for ro in range(n_roads):
        for tr in range(model_parameters.num_track):
            for s in range(constants.num_source):
                # Handle all size fractions except the last one
                for x in range(constants.num_size - 1):
                    variables.M_road_bin_data[s, x, ti, tr, ro] = (
                        variables.M_road_data[s, x, ti, tr, ro]
                        - variables.M_road_data[s, x + 1, ti, tr, ro]
                    )
                # Handle the last size fraction
                x = constants.num_size - 1
                variables.M_road_bin_data[s, x, ti, tr, ro] = variables.M_road_data[
                    s, x, ti, tr, ro
                ]

    # Set observed surface temperature and wetness as road_meteo_data
    for ro in range(n_roads):
        for tr in range(model_parameters.num_track):
            variables.road_meteo_data[
                constants.road_temperature_obs_index, time_slice, tr, ro
            ] = converted_data.meteo_data[
                constants.road_temperature_obs_input_index, time_slice, ro
            ]
            variables.road_meteo_data[
                constants.road_wetness_obs_index, time_slice, tr, ro
            ] = converted_data.meteo_data[
                constants.road_wetness_obs_input_index, time_slice, ro
            ]

    # Initialize exhaust emission into PM2.5 and all tracks
    x = constants.pm_25
    if metadata.exhaust_EF_available and exhaust_flag:
        for ti in range(min_time, max_time + 1):
            for tr in range(model_parameters.num_track):
                for v in range(constants.num_veh):
                    variables.E_road_bin_data[
                        constants.exhaust_index, x, constants.E_total_index, ti, tr, 0
                    ] += (
                        # TODO: Something wring with traffic data, can be in read_input_traffic.py
                        converted_data.traffic_data[constants.N_v_index[v], ti, 0]
                        * metadata.exhaust_EF[v]
                    )
    elif airquality_data.EP_emis_available and exhaust_flag:
        for tr in range(model_parameters.num_track):
            ep_data_slice = airquality_data.EP_emis[min_time : max_time + 1]
            variables.E_road_bin_data[
                constants.exhaust_index, x, constants.E_total_index, time_slice, tr, 0
            ] = ep_data_slice

    logger.info("Model variables initialised")

    return variables
