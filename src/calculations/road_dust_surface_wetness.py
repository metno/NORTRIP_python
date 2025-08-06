import numpy as np
import logging
import constants as C
from functions import (
    r_aero_func_with_stability,
    r_aero_func,
    dewpoint_from_rh_func,
    f_spray_func,
    mass_balance_func,
    surface_energy_model_4_func,
    e_diff_func,
    relax_func,
)
from initialise.road_dust_initialise_time import time_config
from initialise.road_dust_initialise_variables import model_variables
from input_classes import (
    converted_data,
    input_activity,
    input_metadata,
    input_meteorology,
)
from config_classes import (
    model_parameters,
    model_flags,
)

logger = logging.getLogger(__name__)


def road_dust_surface_wetness(
    ti: int,
    tr: int,
    ro: int,
    time_config: time_config,
    converted_data: converted_data,
    model_variables: model_variables,
    model_parameters: model_parameters,
    model_flags: model_flags,
    metadata: input_metadata,
    input_activity: input_activity,
    tf: int = None,
    meteorology_input: "input_meteorology" = None,
):
    """
    Calculate road surface moisture and retention in the NORTRIP model.

    This function calculates surface moisture, evaporation, drainage, spray,
    and retention factors for road dust modeling.

    Args:
        ti: Current time index
        tr: Track index
        ro: Road index
        time_config: Time configuration object
        converted_data: Converted input data
        model_variables: Model variables object (modified in place)
        model_parameters: Model parameters
        model_flags: Model flags
        metadata: Input metadata
        input_activity: Activity input data
        tf: Forecast time index (optional)
        meteorology_input: Meteorology input data (optional)
    """

    surface_moisture_min = 0.000001

    dz_snow_albedo = 3  # Depth of snow required before implementing snow albedo mm.w.e.
    Z_CLOUD = 100  # Only used when no global radiation is available
    z0t = model_parameters.z0 / 10

    length_veh = np.zeros(C.num_veh)
    length_veh[C.li] = 5.0
    length_veh[C.he] = 15.0

    retain_water_by_snow = 1
    b_factor = 1 / (1000 * metadata.b_road_lanes * model_parameters.f_track[tr])

    g_road_0_data = np.full(C.num_moisture, metadata.nodata)
    S_melt_temp = metadata.nodata
    g_road_fraction = np.full(C.num_moisture, metadata.nodata)
    M2_road_salt_0 = np.full(C.num_salt, metadata.nodata)
    g_road_temp = metadata.nodata
    R_evaporation = np.zeros(C.num_moisture)
    R_ploughing = np.zeros(C.num_moisture)
    R_road_drainage = np.zeros(C.num_moisture)
    R_spray = np.zeros(C.num_moisture)
    R_drainage = np.zeros(C.num_moisture)
    melt_temperature_salt_temp = metadata.nodata
    RH_salt_temp = metadata.nodata
    M_road_dissolved_ratio_temp = metadata.nodata
    T_s_0 = metadata.nodata
    T_a_0 = metadata.nodata
    FF_0 = metadata.nodata
    RH_s_0 = metadata.nodata
    short_rad_net_temp = metadata.nodata
    g_road_drainable_withrain = metadata.nodata
    g_road_drainable_min_temp = metadata.nodata
    g_road_total = metadata.nodata
    g_ratio_road = metadata.nodata
    g_ratio_brake = metadata.nodata
    g_ratio_binder = 0
    g_ratio_obs = 0
    g_road_sprayable = 0

    g_road_0_data[: C.num_moisture] = (
        model_variables.g_road_data[
            : C.num_moisture, max(time_config.min_time, ti - 1), tr, ro
        ]
        + surface_moisture_min * 0.5
    )
    T_s_0 = model_variables.road_meteo_data[
        C.T_s_index, max(time_config.min_time, ti - 1), tr, ro
    ]

    T_a_0 = converted_data.meteo_data[
        C.T_a_index, max(time_config.min_time, ti - 1), ro
    ]

    RH_s_0 = model_variables.road_meteo_data[
        C.RH_s_index, max(time_config.min_time, ti - 1), tr, ro
    ]

    FF_0 = converted_data.meteo_data[C.FF_index, max(time_config.min_time, ti - 1), ro]
    M2_road_salt_0[: C.num_salt] = (
        np.sum(
            model_variables.M_road_bin_data[
                C.salt_index, : C.num_size, max(time_config.min_time, ti - 1), tr, ro
            ],
            axis=1,
        )
        * b_factor
    )

    # # DEBUG: Print previous time step values (equivalent to MATLAB lines 71-92)
    # if ti >= 745 and ti <= 750:  # Print when real data starts
    #     logger.info("=== PYTHON PREVIOUS TIME STEP VALUES DEBUG ===")
    #     logger.info(
    #         f"ti={ti}, tr={tr}, ro={ro}, prev_time={max(time_config.min_time, ti - 1)}"
    #     )
    #     logger.info(
    #         f"g_road_0_data (prev moisture): [{g_road_0_data[0]:.6f} {g_road_0_data[1]:.6f} {g_road_0_data[2]:.6f}]"
    #     )
    #     logger.info(f"T_s_0 (prev road temp): {T_s_0:.6f}")
    #     logger.info(f"T_a_0 (prev air temp): {T_a_0:.6f}")
    #     logger.info(f"FF_0 (prev wind speed): {FF_0:.6f}")
    #     logger.info(f"RH_s_0 (prev surface RH): {RH_s_0:.6f}")
    #     logger.info(
    #         f"M2_road_salt_0 (prev salt mass): [{M2_road_salt_0[0]:.6f} {M2_road_salt_0[1]:.6f}]"
    #     )
    #     logger.info(f"b_factor: {b_factor:.6f}")
    #     logger.info(f"surface_moisture_min: {surface_moisture_min:.6f}")

    #     # Also print raw previous values
    #     prev_g_road_raw = model_variables.g_road_data[
    #         : C.num_moisture, max(time_config.min_time, ti - 1), tr, ro
    #     ]
    #     logger.info(
    #         f"Raw prev g_road_data: [{prev_g_road_raw[0]:.6f} {prev_g_road_raw[1]:.6f} {prev_g_road_raw[2]:.6f}]"
    #     )

    #     prev_T_s_raw = model_variables.road_meteo_data[
    #         C.T_s_index, max(time_config.min_time, ti - 1), tr, ro
    #     ]
    #     prev_RH_s_raw = model_variables.road_meteo_data[
    #         C.RH_s_index, max(time_config.min_time, ti - 1), tr, ro
    #     ]
    #     logger.info(
    #         f"Raw prev T_s: {prev_T_s_raw:.6f}, Raw prev RH_s: {prev_RH_s_raw:.6f}"
    #     )
    #     logger.info("=== END PYTHON PREVIOUS VALUES DEBUG ===")

    # --------------------------------------------------------------------------
    # Set precipitation production term
    # This assumes that the rain is in mm for the given dt period
    # --------------------------------------------------------------------------
    model_variables.g_road_balance_data[C.water_index, C.P_precip_index, ti, tr, ro] = (
        converted_data.meteo_data[C.Rain_precip_index, ti, ro] / time_config.dt
    )

    model_variables.g_road_balance_data[C.snow_index, C.P_precip_index, ti, tr, ro] = (
        converted_data.meteo_data[C.Snow_precip_index, ti, ro] / time_config.dt
    )

    model_variables.g_road_balance_data[C.ice_index, C.P_precip_index, ti, tr, ro] = 0

    # --------------------------------------------------------------------------
    # Sub surface temperature given as weighted sum of surface temperatures
    # when use_subsurface_flag=2 or air temperature =3
    # More realistic than using the running mean air temperature
    # --------------------------------------------------------------------------
    if ti > time_config.min_time and model_flags.use_subsurface_flag == 2:
        model_variables.road_meteo_data[C.T_sub_index, ti, tr, ro] = (
            model_variables.road_meteo_data[C.T_sub_index, max(1, ti - 1), tr, ro]
            * (1.0 - time_config.dt / model_parameters.sub_surf_average_time)
            + model_variables.road_meteo_data[C.T_s_index, max(1, ti - 1), tr, ro]
            * time_config.dt
            / model_parameters.sub_surf_average_time
        )

    if ti > time_config.min_time and model_flags.use_subsurface_flag == 3:
        model_variables.road_meteo_data[C.T_sub_index, ti, tr, ro] = (
            model_variables.road_meteo_data[C.T_sub_index, max(1, ti - 1), tr, ro]
            * (1.0 - time_config.dt / model_parameters.sub_surf_average_time)
            + converted_data.meteo_data[C.T_a_index, max(1, ti - 1), ro]
            * time_config.dt
            / model_parameters.sub_surf_average_time
        )

    # --------------------------------------------------------------------------
    # Ploughing road sinks rate
    # --------------------------------------------------------------------------
    R_ploughing[: C.num_moisture] = (
        -np.log(1 - model_parameters.h_ploughing_moisture[: C.num_moisture] + 0.0001)
        / time_config.dt
        * converted_data.activity_data[C.t_ploughing_index, ti, ro]
        * model_flags.use_ploughing_data_flag
    )

    # --------------------------------------------------------------------------
    # Wetting production
    # --------------------------------------------------------------------------
    model_variables.g_road_balance_data[
        C.water_index, C.P_roadwetting_index, ti, tr, ro
    ] = (
        converted_data.activity_data[C.g_road_wetting_index, ti, ro]
        / time_config.dt
        * model_flags.use_wetting_data_flag
    )

    # --------------------------------------------------------------------------
    # Evaporation
    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    # Calculate aerodynamic resistance
    # --------------------------------------------------------------------------
    use_stability = 1
    # Prepare vehicle arrays for aerodynamic resistance functions
    V_veh_array = converted_data.traffic_data[C.V_veh_index, ti, ro]
    N_v_array = converted_data.traffic_data[C.N_v_index, ti, ro] / metadata.n_lanes

    if use_stability:
        model_variables.road_meteo_data[C.r_aero_index, ti, tr, ro] = (
            r_aero_func_with_stability(
                FF_0,
                T_a_0,
                T_s_0,
                metadata.z_FF,
                metadata.z_T,
                model_parameters.z0,
                z0t,
                V_veh_array,
                N_v_array,
                C.num_veh,
                model_parameters.a_traffic,
            )
        )
        model_variables.road_meteo_data[C.r_aero_notraffic_index, ti, tr, ro] = (
            r_aero_func_with_stability(
                FF_0,
                T_a_0,
                T_s_0,
                metadata.z_FF,
                metadata.z_T,
                model_parameters.z0,
                z0t,
                V_veh_array * 0,  # No traffic
                N_v_array * 0,  # No traffic
                C.num_veh,
                model_parameters.a_traffic,
            )
        )
    else:
        model_variables.road_meteo_data[C.r_aero_index, ti, tr, ro] = r_aero_func(
            converted_data.meteo_data[C.FF_index, ti, ro],
            metadata.z_FF,
            metadata.z_T,
            model_parameters.z0,
            z0t,
            V_veh_array,
            N_v_array,
            C.num_veh,
            model_parameters.a_traffic,
        )
        model_variables.road_meteo_data[C.r_aero_notraffic_index, ti, tr, ro] = (
            r_aero_func(
                converted_data.meteo_data[C.FF_index, ti, ro],
                metadata.z_FF,
                metadata.z_T,
                model_parameters.z0,
                z0t,
                V_veh_array * 0,
                N_v_array * 0,
                C.num_veh,
                model_parameters.a_traffic,
            )
        )

    if model_flags.use_traffic_turb_flag == 0:
        model_variables.road_meteo_data[C.r_aero_index, ti, tr, ro] = (
            model_variables.road_meteo_data[C.r_aero_notraffic_index, ti, tr, ro]
        )

    # --------------------------------------------------------------------------
    # Calculate the traffic induced heat flux (W/m2)
    # --------------------------------------------------------------------------
    model_variables.road_meteo_data[C.H_traffic_index, ti, tr, ro] = 0
    if model_flags.use_traffic_turb_flag:
        for v in range(C.num_veh):
            if (
                converted_data.traffic_data[C.V_veh_index[v], ti, ro] != 0
                and metadata.n_lanes != 0
            ):
                model_variables.road_meteo_data[C.H_traffic_index, ti, tr, ro] += (
                    model_parameters.H_veh[v]
                    * min(
                        1,
                        length_veh[v]
                        / converted_data.traffic_data[C.V_veh_index[v], ti, ro]
                        * converted_data.traffic_data[C.N_v_index[v], ti, ro]
                        / (metadata.n_lanes * 1000),
                    )
                )

    # --------------------------------------------------------------------------
    # Set surface relative humidity
    # --------------------------------------------------------------------------
    if model_flags.surface_humidity_flag == 1:
        model_variables.road_meteo_data[C.RH_s_index, ti, tr, ro] = (
            min(
                1,
                np.sum(g_road_0_data[: C.num_moisture])
                / model_parameters.g_road_evaporation_thresh,
            )
            * 100
        )
    elif model_flags.surface_humidity_flag == 2:
        model_variables.road_meteo_data[C.RH_s_index, ti, tr, ro] = (
            1
            - np.exp(
                -np.sum(g_road_0_data[: C.num_moisture])
                / model_parameters.g_road_evaporation_thresh
                * 4
            )
        ) * 100
    else:
        model_variables.road_meteo_data[C.RH_s_index, ti, tr, ro] = 100

    # --------------------------------------------------------------------------
    # Calculate evaporation energy balance including melt of snow and ice
    # --------------------------------------------------------------------------
    if model_flags.evaporation_flag:
        short_rad_net_temp = model_variables.road_meteo_data[
            C.short_rad_net_index, ti, tr, ro
        ]

        # Adjust net radiation if snow is present on the road
        if g_road_0_data[C.snow_index] > dz_snow_albedo:
            short_rad_net_temp = (
                model_variables.road_meteo_data[C.short_rad_net_index, ti, tr, ro]
                * (1 - model_parameters.albedo_snow)
                / (1 - metadata.albedo_road)
            )

        # Forecast and energy correction handling (lines 175-228 in MATLAB)
        E_corr = 0
        if tf is not None and model_flags.forecast_hour > 0:
            forecast_index = max(
                0, round(model_flags.forecast_hour / time_config.dt - 1)
            )
            # This part calculates the energy correction
            if (
                model_flags.use_energy_correction_flag
                and tf > time_config.min_time + 1
                and tf == ti
            ):
                (
                    model_variables.road_meteo_data[C.E_index, tf - 1, tr, ro],
                    model_variables.road_meteo_data[C.E_correction_index, tf, tr, ro],
                    model_variables.road_meteo_data[C.T_s_index, tf - 1, tr, ro],
                ) = e_diff_func(
                    model_variables.road_meteo_data[
                        C.road_temperature_obs_index, tf - 1, tr, ro
                    ],
                    model_variables.road_meteo_data[C.T_s_index, tf - 2, tr, ro],
                    converted_data.meteo_data[C.T_a_index, tf - 1, ro],
                    model_variables.road_meteo_data[C.T_sub_index, tf - 1, tr, ro],
                    model_variables.road_meteo_data[
                        C.E_correction_index, tf - 1, tr, ro
                    ],
                    converted_data.meteo_data[C.pressure_index, tf - 1, ro],
                    model_parameters.dzs,
                    time_config.dt,
                    model_variables.road_meteo_data[C.r_aero_index, tf - 1, tr, ro],
                    short_rad_net_temp,
                    converted_data.meteo_data[C.long_rad_in_index, tf - 1, ro],
                    model_variables.road_meteo_data[C.H_traffic_index, tf - 1, tr, ro],
                    model_variables.road_meteo_data[C.L_index, tf - 1, tr, ro],
                    model_variables.road_meteo_data[C.G_freeze_index, tf - 1, tr, ro],
                    model_variables.road_meteo_data[C.G_melt_index, tf - 1, tr, ro],
                    model_parameters.sub_surf_param,
                    model_flags.use_subsurface_flag,
                )

            if tf > time_config.min_time + 1 and tf == ti:
                # Set the previous (initial) model surface temperature to the observed surface temperature in forecast mode
                model_variables.road_temperature_forecast_missing[
                    tf + forecast_index
                ] = 1

                # Find if road temperature observation is missing
                r_missing = (
                    tf - 1 in meteorology_input.road_temperature_obs_missing
                    if meteorology_input
                    else True
                )

                if not r_missing:
                    model_variables.original_bias_T_s = (
                        model_variables.road_meteo_data[C.T_s_index, tf - 1, tr, ro]
                        - model_variables.road_meteo_data[
                            C.road_temperature_obs_index, tf - 1, tr, ro
                        ]
                    )
                    if model_flags.use_observed_temperature_init_flag:
                        model_variables.road_meteo_data[C.T_s_index, tf - 1, tr, ro] = (
                            model_variables.road_meteo_data[
                                C.road_temperature_obs_index, tf - 1, tr, ro
                            ]
                        )
                    model_variables.road_temperature_forecast_missing[
                        tf + forecast_index
                    ] = 0
                # else: road_temperature_forecast_missing will be 1 (true) and the modelled temperature will be used

            if model_flags.use_energy_correction_flag:
                E_corr = model_variables.road_meteo_data[
                    C.E_correction_index, tf, tr, ro
                ] * relax_func(time_config.dt, (ti - tf) + 1)

            T_s_0 = model_variables.road_meteo_data[C.T_s_index, ti - 1, tr, ro]

        # DEBUG: Print surface energy model inputs
        if ti >= 745 and ti <= 750:
            logger.info("=== SURFACE ENERGY MODEL INPUTS ===")
            logger.info(f"short_rad_net_temp: {short_rad_net_temp:.6f}")
            logger.info(
                f"long_rad_in: {converted_data.meteo_data[C.long_rad_in_index, ti, ro]:.6f}"
            )
            logger.info(
                f"H_traffic: {model_variables.road_meteo_data[C.H_traffic_index, ti, tr, ro]:.6f}"
            )
            logger.info(
                f"r_aero: {model_variables.road_meteo_data[C.r_aero_index, ti, tr, ro]:.6f}"
            )
            logger.info(f"T_a: {converted_data.meteo_data[C.T_a_index, ti, ro]:.6f}")
            logger.info(f"T_s_0: {T_s_0:.6f}")
            logger.info(
                f"T_sub: {model_variables.road_meteo_data[C.T_sub_index, ti, tr, ro]:.6f}"
            )
            logger.info(f"RH: {converted_data.meteo_data[C.RH_index, ti, ro]:.6f}")
            logger.info(
                f"RH_s: {model_variables.road_meteo_data[C.RH_s_index, ti, tr, ro]:.6f}"
            )
            logger.info(f"RH_s_0: {RH_s_0:.6f}")
            logger.info(
                f"pressure: {converted_data.meteo_data[C.pressure_index, ti, ro]:.6f}"
            )
            logger.info(f"g_water: {g_road_0_data[C.water_index]:.6f}")
            logger.info(
                f"g_ice_snow: {g_road_0_data[C.ice_index] + g_road_0_data[C.snow_index]:.6f}"
            )
            logger.info(f"dt: {time_config.dt:.6f}")
            logger.info(f"E_corr: {E_corr:.6f}")

        # Call the surface energy model
        (
            model_variables.road_meteo_data[C.T_s_index, ti, tr, ro],
            model_variables.road_meteo_data[C.T_melt_index, ti, tr, ro],
            model_variables.road_meteo_data[C.RH_salt_final_index, ti, tr, ro],
            model_variables.road_meteo_data[C.RH_s_index, ti, tr, ro],
            model_variables.road_salt_data[C.dissolved_ratio_index, :, ti, tr, ro],
            model_variables.road_meteo_data[C.evap_index, ti, tr, ro],
            model_variables.road_meteo_data[C.evap_pot_index, ti, tr, ro],
            S_melt_temp,
            model_variables.g_road_balance_data[
                C.ice_index, C.P_freeze_index, ti, tr, ro
            ],
            model_variables.road_meteo_data[C.H_index, ti, tr, ro],
            model_variables.road_meteo_data[C.L_index, ti, tr, ro],
            model_variables.road_meteo_data[C.G_index, ti, tr, ro],
            model_variables.road_meteo_data[C.long_rad_out_index, ti, tr, ro],
            model_variables.road_meteo_data[C.long_rad_net_index, ti, tr, ro],
            model_variables.road_meteo_data[C.rad_net_index, ti, tr, ro],
            model_variables.road_meteo_data[C.G_sub_index, ti, tr, ro],
            model_variables.road_meteo_data[C.G_freeze_index, ti, tr, ro],
            model_variables.road_meteo_data[C.G_melt_index, ti, tr, ro],
        ) = surface_energy_model_4_func(
            short_rad_net_temp,
            converted_data.meteo_data[C.long_rad_in_index, ti, ro],
            model_variables.road_meteo_data[C.H_traffic_index, ti, tr, ro],
            model_variables.road_meteo_data[C.r_aero_index, ti, tr, ro],
            converted_data.meteo_data[C.T_a_index, ti, ro],
            T_s_0,
            model_variables.road_meteo_data[C.T_sub_index, ti, tr, ro],
            converted_data.meteo_data[C.RH_index, ti, ro],
            model_variables.road_meteo_data[C.RH_s_index, ti, tr, ro],
            RH_s_0,
            converted_data.meteo_data[C.pressure_index, ti, ro],
            model_parameters.dzs,
            time_config.dt,
            g_road_0_data[C.water_index],
            g_road_0_data[C.ice_index] + g_road_0_data[C.snow_index],
            model_parameters.g_road_evaporation_thresh,
            M2_road_salt_0,
            input_activity.salt_type,  # salt_type
            model_parameters.sub_surf_param,
            model_flags.surface_humidity_flag,
            model_flags.use_subsurface_flag,
            model_flags.use_salt_humidity_flag,
            E_corr,
        )

        # DEBUG: Print surface energy model outputs
        if ti >= 745 and ti <= 750:
            logger.info(f"=== SURFACE ENERGY MODEL OUTPUT ===")
            logger.info(
                f"evap_rate: {model_variables.road_meteo_data[C.evap_index, ti, tr, ro]:.6f}"
            )
            logger.info(
                f"T_s: {model_variables.road_meteo_data[C.T_s_index, ti, tr, ro]:.6f}"
            )
            logger.info(
                f"RH_s_final: {model_variables.road_meteo_data[C.RH_s_index, ti, tr, ro]:.6f}"
            )

        # Redistribute melting between snow and ice
        snow_ice_sum = np.sum(g_road_0_data[C.snow_ice_index])
        if snow_ice_sum > 0:
            model_variables.g_road_balance_data[
                C.snow_ice_index, C.S_melt_index, ti, tr, ro
            ] += S_melt_temp * g_road_0_data[C.snow_ice_index] / snow_ice_sum

    # --------------------------------------------------------------------------
    # Calculate surface dewpoint temperature
    # --------------------------------------------------------------------------
    model_variables.road_meteo_data[C.T_s_dewpoint_index, ti, tr, ro] = (
        dewpoint_from_rh_func(
            model_variables.road_meteo_data[C.T_s_index, ti, tr, ro],
            model_variables.road_meteo_data[C.RH_s_index, ti, tr, ro],
        )
    )

    converted_data.meteo_data[C.T_dewpoint_index, ti, ro] = dewpoint_from_rh_func(
        converted_data.meteo_data[C.T_a_index, ti, ro],
        converted_data.meteo_data[C.RH_index, ti, ro],
    )

    # --------------------------------------------------------------------------
    # Set the evaporation/condensation rates
    # Distribute evaporation between water and ice according to the share of water and ice
    # Distribute the condensation between water and ice according to temperature
    # --------------------------------------------------------------------------
    g_road_fraction[: C.num_moisture] = g_road_0_data[: C.num_moisture] / np.sum(
        g_road_0_data[: C.num_moisture]
    )

    if model_variables.road_meteo_data[C.evap_index, ti, tr, ro] > 0:  # Evaporation
        # # Debug the calculation to identify the issue
        # evap_rate = model_variables.road_meteo_data[C.evap_index, ti, tr, ro]
        # if ti >= 745 and ti <= 750:
        #     logger.info("DEBUG EVAPORATION CALCULATION:")
        #     logger.info(f"  evap_rate: {evap_rate:.6f}")
        #     logger.info(
        #         f"  g_road_0_data: [{g_road_0_data[0]:.6f} {g_road_0_data[1]:.6f} {g_road_0_data[2]:.6f}]"
        #     )
        #     logger.info(
        #         f"  g_road_fraction: [{g_road_fraction[0]:.6f} {g_road_fraction[1]:.6f} {g_road_fraction[2]:.6f}]"
        #     )

        # Calculate R_evaporation element by element to handle division by zero correctly
        # Use the raw g_road_data (before adding surface_moisture_min) to check for real moisture
        raw_g_road_data = model_variables.g_road_data[
            : C.num_moisture, max(time_config.min_time, ti - 1), tr, ro
        ]

        evap_rate = model_variables.road_meteo_data[C.evap_index, ti, tr, ro]
        for m in range(C.num_moisture):
            if raw_g_road_data[m] > surface_moisture_min:
                R_evaporation[m] = evap_rate / g_road_0_data[m] * g_road_fraction[m]
                # if ti >= 745 and ti <= 750:
                #     logger.info(
                #         f"  R_evaporation[{m}] = {evap_rate:.6f} / {g_road_0_data[m]:.6f} * {g_road_fraction[m]:.6f} = {R_evaporation[m]:.6f}"
                #     )
            else:
                R_evaporation[m] = 0
                # if ti >= 745 and ti <= 750:
                #     logger.info(
                #         f"  R_evaporation[{m}] = 0 (because raw_g_road_data[{m}] = {raw_g_road_data[m]:.6f} <= surface_moisture_min = {surface_moisture_min:.6f})"
                #     )

        # if ti >= 745 and ti <= 750:
        #     logger.info(
        #         f"  R_evaporation result: [{R_evaporation[0]:.6f} {R_evaporation[1]:.6f} {R_evaporation[2]:.6f}]"
        #     )

        model_variables.g_road_balance_data[
            : C.num_moisture, C.P_evap_index, ti, tr, ro
        ] = 0
    else:  # Condensation
        if (
            model_variables.road_meteo_data[C.T_s_index, ti, tr, ro]
            >= model_variables.road_meteo_data[C.T_melt_index, ti, tr, ro]
        ):  # Condensation to water
            model_variables.g_road_balance_data[
                C.water_index, C.P_evap_index, ti, tr, ro
            ] = -model_variables.road_meteo_data[C.evap_index, ti, tr, ro]
            model_variables.g_road_balance_data[
                C.snow_index, C.P_evap_index, ti, tr, ro
            ] = 0
            model_variables.g_road_balance_data[
                C.ice_index, C.P_evap_index, ti, tr, ro
            ] = 0
        elif model_flags.evaporation_flag == 2:
            # Condensation to snow. Hoar frost is more like snow than ice from melting
            model_variables.g_road_balance_data[
                C.snow_index, C.P_evap_index, ti, tr, ro
            ] = -model_variables.road_meteo_data[C.evap_index, ti, tr, ro]
            model_variables.g_road_balance_data[
                C.water_index, C.P_evap_index, ti, tr, ro
            ] = 0
            model_variables.g_road_balance_data[
                C.ice_index, C.P_evap_index, ti, tr, ro
            ] = 0
        elif model_flags.evaporation_flag == 1:
            # Condensation only to ice (not snow)
            model_variables.g_road_balance_data[
                C.snow_index, C.P_evap_index, ti, tr, ro
            ] = 0
            model_variables.g_road_balance_data[
                C.water_index, C.P_evap_index, ti, tr, ro
            ] = 0
            model_variables.g_road_balance_data[
                C.ice_index, C.P_evap_index, ti, tr, ro
            ] = -model_variables.road_meteo_data[C.evap_index, ti, tr, ro]

        R_evaporation[: C.num_moisture] = 0

    model_variables.g_road_balance_data[
        : C.num_moisture, C.S_evap_index, ti, tr, ro
    ] = R_evaporation[: C.num_moisture] * g_road_0_data[: C.num_moisture]

    # --------------------------------------------------------------------------
    # Set drainage rates
    # --------------------------------------------------------------------------
    # This drainage type reduces exponentially according to a time scale
    # Should only be used when the model is run at much shorter time scales than
    # 1 hour, e.g. 5 - 10 minutes
    if model_flags.drainage_type_flag == 1:
        # Exponential drainage based on time scale tau_road_drainage
        g_road_water_drainable = max(
            0, g_road_0_data[C.water_index] - model_parameters.g_road_drainable_min
        )
        g_road_drainable_withrain = max(
            0,
            converted_data.meteo_data[C.Rain_precip_index, ti, ro]
            + g_road_0_data[C.water_index]
            - model_parameters.g_road_drainable_min,
        )

        if g_road_drainable_withrain > 0:
            R_drainage[C.water_index] = 1 / model_parameters.tau_road_drainage
        else:
            R_drainage[C.water_index] = 0

        # Diagnostic only. Not correct mathematically
        model_variables.g_road_balance_data[
            C.water_index, C.S_drainage_tau_index, ti, tr, ro
        ] = g_road_drainable_withrain * R_drainage[C.water_index]

    elif model_flags.drainage_type_flag == 2:
        R_drainage[C.water_index] = 0
        model_variables.g_road_balance_data[
            C.water_index, C.S_drainage_tau_index, ti, tr, ro
        ] = 0

    elif model_flags.drainage_type_flag == 3:
        # Combined drainage, first instantaneous removal to g_road_drainable_min
        # Then exponential drainage based on time scale tau_road_drainage to g_road_drainable_thresh
        g_road_water_drainable = max(
            0,
            g_road_0_data[C.water_index] - model_parameters.g_road_drainable_thresh,
        )
        g_road_drainable_withrain = max(
            0,
            converted_data.meteo_data[C.Rain_precip_index, ti, ro]
            + g_road_0_data[C.water_index]
            - model_parameters.g_road_drainable_min,
        )

        if g_road_drainable_withrain == 0 and g_road_water_drainable > 0:
            R_drainage[C.water_index] = 1 / model_parameters.tau_road_drainage
        else:
            R_drainage[C.water_index] = 0

        # Diagnostic only. Not correct mathematically
        model_variables.g_road_balance_data[
            C.water_index, C.S_drainage_tau_index, ti, tr, ro
        ] = g_road_water_drainable * R_drainage[C.water_index]

    # Set the drainage rate to be used in the dust/salt module
    model_variables.g_road_balance_data[
        C.water_index, C.R_drainage_index, ti, tr, ro
    ] = R_drainage[C.water_index]

    # --------------------------------------------------------------------------
    # Splash and spray sinks and production. Also for snow but not defined yet
    # --------------------------------------------------------------------------
    R_spray[: C.num_moisture] = 0
    for m in range(C.num_moisture):
        g_road_sprayable = max(
            0, g_road_0_data[m] - model_parameters.g_road_sprayable_min[m]
        )
        if g_road_sprayable > 0 and model_flags.water_spray_flag:
            for v in range(C.num_veh):
                R_spray[m] += (
                    converted_data.traffic_data[C.N_v_index[v], ti, ro]
                    / metadata.n_lanes
                    * model_parameters.veh_track[tr]
                    * f_spray_func(
                        model_parameters.R_0_spray[v, m],
                        converted_data.traffic_data[C.V_veh_index[v], ti, ro],
                        model_parameters.V_ref_spray[m],
                        model_parameters.V_thresh_spray[m],
                        model_parameters.a_spray[m],
                        model_flags.water_spray_flag,
                    )
                )
            # Adjust according to minimum
            R_spray[m] = (
                R_spray[m]
                * g_road_sprayable
                / (g_road_0_data[m] + surface_moisture_min)
            )

        model_variables.g_road_balance_data[m, C.S_spray_index, ti, tr, ro] = (
            R_spray[m] * g_road_0_data[m]
        )
        model_variables.g_road_balance_data[m, C.R_spray_index, ti, tr, ro] = R_spray[m]

    # --------------------------------------------------------------------------
    # Add production terms
    # --------------------------------------------------------------------------
    model_variables.g_road_balance_data[
        : C.num_moisture, C.P_total_index, ti, tr, ro
    ] = (
        model_variables.g_road_balance_data[
            : C.num_moisture, C.P_precip_index, ti, tr, ro
        ]
        + model_variables.g_road_balance_data[
            : C.num_moisture, C.P_evap_index, ti, tr, ro
        ]
        + model_variables.g_road_balance_data[
            : C.num_moisture, C.P_roadwetting_index, ti, tr, ro
        ]
    )

    # --------------------------------------------------------------------------
    # Add sink rate terms
    # --------------------------------------------------------------------------
    R_total = R_evaporation + R_drainage + R_spray + R_ploughing

    # --------------------------------------------------------------------------
    # Calculate change in water and snow
    # --------------------------------------------------------------------------
    for m in range(C.num_moisture):
        model_variables.g_road_data[m, ti, tr, ro] = mass_balance_func(
            g_road_0_data[m],
            model_variables.g_road_balance_data[m, C.P_total_index, ti, tr, ro],
            R_total[m],
            time_config.dt,
        )

    # # DEBUG: Print final calculated g_road_data values (equivalent to MATLAB lines 403-415)
    # if ti >= 745 and ti <= 750:  # Print when real data starts
    #     logger.info("=== PYTHON FINAL G_ROAD_DATA VALUES DEBUG ===")
    #     logger.info(f"ti={ti}, tr={tr}, ro={ro}")
    #     final_g_road = model_variables.g_road_data[: C.num_moisture, ti, tr, ro]
    #     logger.info(
    #         f"Final g_road_data: [{final_g_road[0]:.6f} {final_g_road[1]:.6f} {final_g_road[2]:.6f}]"
    #     )

    #     production_terms = model_variables.g_road_balance_data[
    #         : C.num_moisture, C.P_total_index, ti, tr, ro
    #     ]
    #     logger.info(
    #         f"Production terms (P_total): [{production_terms[0]:.6f} {production_terms[1]:.6f} {production_terms[2]:.6f}]"
    #     )
    #     logger.info(
    #         f"Sink rates (R_total): [{R_total[0]:.6f} {R_total[1]:.6f} {R_total[2]:.6f}]"
    #     )

    #     # Debug individual R components
    #     logger.info(
    #         f"R_evaporation: [{R_evaporation[0]:.6f} {R_evaporation[1]:.6f} {R_evaporation[2]:.6f}]"
    #     )
    #     logger.info(
    #         f"R_drainage: [{R_drainage[0]:.6f} {R_drainage[1]:.6f} {R_drainage[2]:.6f}]"
    #     )
    #     logger.info(f"R_spray: [{R_spray[0]:.6f} {R_spray[1]:.6f} {R_spray[2]:.6f}]")
    #     logger.info(
    #         f"R_ploughing: [{R_ploughing[0]:.6f} {R_ploughing[1]:.6f} {R_ploughing[2]:.6f}]"
    #     )

    #     # Debug evaporation components
    #     evap_rate = model_variables.road_meteo_data[C.evap_index, ti, tr, ro]
    #     logger.info(f"Evaporation rate: {evap_rate:.6f}")
    #     logger.info(
    #         f"g_road_fraction: [{g_road_fraction[0]:.6f} {g_road_fraction[1]:.6f} {g_road_fraction[2]:.6f}]"
    #     )

    #     logger.info(f"Time step (dt): {time_config.dt:.6f}")
    #     logger.info("=== END PYTHON FINAL G_ROAD_DATA DEBUG ===")

    # --------------------------------------------------------------------------
    # Recalculate spray and evaporation diagnostics based on average moisture
    # --------------------------------------------------------------------------
    for m in range(C.num_moisture):
        model_variables.g_road_balance_data[m, C.S_spray_index, ti, tr, ro] = (
            R_spray[m]
            * (model_variables.g_road_data[m, ti, tr, ro] + g_road_0_data[m])
            / 2
        )
        model_variables.g_road_balance_data[m, C.S_evap_index, ti, tr, ro] = (
            R_evaporation[m]
            * (model_variables.g_road_data[m, ti, tr, ro] + g_road_0_data[m])
            / 2
        )

    # --------------------------------------------------------------------------
    # Remove and add snow melt after the rest of the calculations
    # --------------------------------------------------------------------------
    # Can't melt more ice or snow than there is
    for m in C.snow_ice_index:
        model_variables.g_road_balance_data[m, C.S_melt_index, ti, tr, ro] = min(
            model_variables.g_road_data[m, ti, tr, ro] / time_config.dt,
            model_variables.g_road_balance_data[m, C.S_melt_index, ti, tr, ro],
        )

    # Sink of melt is the same as production of water
    model_variables.g_road_balance_data[C.water_index, C.P_melt_index, ti, tr, ro] = (
        np.sum(
            model_variables.g_road_balance_data[
                C.snow_ice_index, C.S_melt_index, ti, tr, ro
            ]
        )
    )

    for m in range(C.num_moisture):
        model_variables.g_road_data[m, ti, tr, ro] = max(
            0,
            model_variables.g_road_data[m, ti, tr, ro]
            - model_variables.g_road_balance_data[m, C.S_melt_index, ti, tr, ro]
            * time_config.dt,
        )
        model_variables.g_road_data[m, ti, tr, ro] = (
            model_variables.g_road_data[m, ti, tr, ro]
            + model_variables.g_road_balance_data[m, C.P_melt_index, ti, tr, ro]
            * time_config.dt
        )

    # --------------------------------------------------------------------------
    # Remove water through drainage for drainage_type_flag=2 or 3
    # --------------------------------------------------------------------------
    g_road_water_drainable = 0
    if model_flags.drainage_type_flag == 2 or model_flags.drainage_type_flag == 3:
        if retain_water_by_snow:
            g_road_drainable_min_temp = max(
                model_parameters.g_road_drainable_min,
                model_variables.g_road_data[C.snow_index, ti, tr, ro],
            )
        else:
            g_road_drainable_min_temp = model_parameters.g_road_drainable_min

        g_road_water_drainable = max(
            0,
            model_variables.g_road_data[C.water_index, ti, tr, ro]
            - g_road_drainable_min_temp,
        )
        model_variables.g_road_data[C.water_index, ti, tr, ro] = min(
            model_variables.g_road_data[C.water_index, ti, tr, ro],
            g_road_drainable_min_temp,
        )
        model_variables.g_road_balance_data[
            C.water_index, C.S_drainage_index, ti, tr, ro
        ] = g_road_water_drainable / time_config.dt

    # --------------------------------------------------------------------------
    # Freeze after the rest of the calculations
    # --------------------------------------------------------------------------
    # Limit the amount of freezing to the amount of available water
    model_variables.g_road_balance_data[C.ice_index, C.P_freeze_index, ti, tr, ro] = (
        min(
            model_variables.g_road_data[C.water_index, ti, tr, ro],
            model_variables.g_road_balance_data[
                C.ice_index, C.P_freeze_index, ti, tr, ro
            ]
            * time_config.dt,
        )
        / time_config.dt
    )
    model_variables.g_road_balance_data[C.water_index, C.S_freeze_index, ti, tr, ro] = (
        model_variables.g_road_balance_data[C.ice_index, C.P_freeze_index, ti, tr, ro]
    )
    model_variables.g_road_data[C.water_index, ti, tr, ro] -= (
        model_variables.g_road_balance_data[C.water_index, C.S_freeze_index, ti, tr, ro]
    )
    model_variables.g_road_data[C.ice_index, ti, tr, ro] += (
        model_variables.g_road_balance_data[C.ice_index, C.P_freeze_index, ti, tr, ro]
    )

    # --------------------------------------------------------------------------
    # Set moisture content to be always>=0. Avoiding round off errors
    # --------------------------------------------------------------------------
    for m in range(C.num_moisture):
        model_variables.g_road_data[m, ti, tr, ro] = max(
            0, model_variables.g_road_data[m, ti, tr, ro]
        )

    # --------------------------------------------------------------------------
    # Calculate inhibition/retention factors
    # --------------------------------------------------------------------------
    g_road_total = np.sum(model_variables.g_road_data[: C.num_moisture, ti, tr, ro])
    g_ratio_road = (g_road_total - model_parameters.g_retention_min[C.road_index]) / (
        model_parameters.g_retention_thresh[C.road_index]
        - model_parameters.g_retention_min[C.road_index]
    )
    g_ratio_brake = (
        model_variables.g_road_data[C.water_index, ti, tr, ro]
        - model_parameters.g_retention_min[C.brake_index]
    ) / (
        model_parameters.g_retention_thresh[C.brake_index]
        - model_parameters.g_retention_min[C.brake_index]
    )
    g_ratio_binder = (
        M2_road_salt_0[1] - model_parameters.g_retention_min[C.salt_index[1]]
    ) / (
        model_parameters.g_retention_thresh[C.salt_index[1]]
        - model_parameters.g_retention_min[C.salt_index[1]]
    )

    if model_flags.retention_flag == 1:
        model_variables.f_q[: C.num_source, ti, tr, ro] = np.maximum(
            0, np.minimum(1, 1 - g_ratio_road)
        )
        model_variables.f_q[: C.num_source, ti, tr, ro] = (
            np.maximum(0, np.minimum(1, 1 - g_ratio_binder))
            * model_variables.f_q[: C.num_source, ti, tr, ro]
        )
        model_variables.f_q[C.brake_index, ti, tr, ro] = max(
            0, min(1, 1 - g_ratio_brake)
        )
    elif model_flags.retention_flag == 2:
        model_variables.f_q[: C.num_source, ti, tr, ro] = np.exp(
            -2 * np.maximum(0, g_ratio_road)
        )
        model_variables.f_q[: C.num_source, ti, tr, ro] = (
            np.exp(-2 * np.maximum(0, g_ratio_binder))
            * model_variables.f_q[: C.num_source, ti, tr, ro]
        )
        model_variables.f_q[C.brake_index, ti, tr, ro] = np.exp(
            -2 * max(0, g_ratio_brake)
        )
    else:
        model_variables.f_q[: C.num_source, ti, tr, ro] = 1.0

    # --------------------------------------------------------------------------
    # Set observed retention parameter if available
    # --------------------------------------------------------------------------
    if (
        meteorology_input is not None
        and meteorology_input.road_wetness_obs_available
        and meteorology_input.road_wetness_obs_in_mm
    ):
        g_ratio_obs = (
            converted_data.meteo_data[C.road_wetness_obs_input_index, ti, ro]
            - model_parameters.g_retention_min[C.road_index]
        ) / (
            model_parameters.g_retention_thresh[C.road_index]
            - model_parameters.g_retention_min[C.road_index]
        )

        if (
            converted_data.meteo_data[C.road_wetness_obs_input_index, ti, ro]
            == metadata.nodata
        ):
            model_variables.f_q_obs[ti, tr, ro] = 1  # No data then the road is dry
        elif model_flags.retention_flag == 1:
            model_variables.f_q_obs[ti, tr, ro] = max(0, min(1, 1 - g_ratio_obs))
        elif model_flags.retention_flag == 2:
            model_variables.f_q_obs[ti, tr, ro] = np.exp(-2 * max(0, g_ratio_obs))
        else:
            model_variables.f_q_obs[ti, tr, ro] = 1

    # Handle non-mm road wetness observations
    if (
        meteorology_input is not None
        and meteorology_input.road_wetness_obs_available
        and meteorology_input.road_wetness_obs_in_mm == 0
    ):
        # f_q_obs=1-(road_wetness_obs-min(road_wetness_obs))./(max(road_wetness_obs)-min(road_wetness_obs));
        middle_max_road_wetness_obs = (
            meteorology_input.max_road_wetness_obs
            - meteorology_input.min_road_wetness_obs
        ) / 2

        if metadata.observed_moisture_cutoff_value == 0:
            observed_moisture_cutoff_value_temp = middle_max_road_wetness_obs
        else:
            observed_moisture_cutoff_value_temp = (
                metadata.observed_moisture_cutoff_value
            )

        if (
            model_variables.road_meteo_data[C.road_wetness_obs_index, ti, tr, ro]
            == metadata.nodata
        ):
            model_variables.f_q_obs[ti, tr, ro] = 1  # No data then dry road
        elif (
            model_variables.road_meteo_data[C.road_wetness_obs_index, ti, tr, ro]
            < observed_moisture_cutoff_value_temp
        ):
            model_variables.f_q_obs[ti, tr, ro] = 1
        else:
            model_variables.f_q_obs[ti, tr, ro] = 0

    # Set retention based on observed wetness if required
    if (
        model_flags.use_obs_retention_flag
        and meteorology_input is not None
        and meteorology_input.road_wetness_obs_available
        and model_flags.retention_flag != 0
    ):
        model_variables.f_q[: C.num_source, ti, tr, ro] = model_variables.f_q_obs[
            ti, tr, ro
        ]
        model_variables.f_q[C.brake_index, ti, tr, ro] = 1  # No retention for brakes
        model_variables.f_q[C.exhaust_index, ti, tr, ro] = 1  # No retention for exhaust
