import numpy as np
import constants
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

    # Set minimum allowable total surface wetness (avoid division by 0)
    surface_moisture_min = 1e-6

    # Fixed parameters
    dz_snow_albedo = 3  # Depth of snow required before implementing snow albedo mm.w.e.
    z0t = (
        model_parameters.z0 / 10
    )  # Roughness length for temperature relative to momentum
    length_veh = np.array(
        [5.0, 15.0]
    )  # Vehicle lengths [li, he] for heat flux calculation

    retain_water_by_snow = (
        1  # Decides if water is allowed to drain off normally when snow is present
    )
    b_factor = 1 / (
        1000 * metadata.b_road_lanes * model_parameters.f_track[tr]
    )  # Converts g/km to g/m^2

    # Local arrays and variables initialization
    g_road_0_data = np.full(constants.num_moisture, metadata.nodata)
    S_melt_temp = metadata.nodata
    g_road_fraction = np.full(constants.num_moisture, metadata.nodata)
    M2_road_salt_0 = np.full(constants.num_salt, metadata.nodata)
    R_evaporation = np.zeros(constants.num_moisture)
    R_ploughing = np.zeros(constants.num_moisture)
    R_spray = np.zeros(constants.num_moisture)
    R_drainage = np.zeros(constants.num_moisture)
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

    # Set initial values for the time step
    prev_time = max(time_config.min_time, ti - 1)
    g_road_0_data[: constants.num_moisture] = (
        model_variables.g_road_data[: constants.num_moisture, prev_time, tr, ro]
        + surface_moisture_min * 0.5
    )
    T_s_0 = model_variables.road_meteo_data[constants.T_s_index, prev_time, tr, ro]
    T_a_0 = converted_data.meteo_data[constants.T_a_index, prev_time, ro]
    FF_0 = converted_data.meteo_data[constants.FF_index, prev_time, ro]
    RH_s_0 = model_variables.road_meteo_data[constants.RH_s_index, prev_time, tr, ro]

    # Calculate salt mass
    M2_road_salt_0[: constants.num_salt] = (
        np.sum(
            model_variables.M_road_bin_data[constants.salt_index, :, prev_time, tr, ro],
            axis=1,
        )
        * b_factor
    )

    # Set precipitation production term
    # This assumes that the rain is in mm for the given dt period
    dt = time_config.dt
    model_variables.g_road_balance_data[
        constants.water_index, constants.P_precip_index, ti, tr, ro
    ] = converted_data.meteo_data[constants.Rain_precip_index, ti, ro] / dt
    model_variables.g_road_balance_data[
        constants.snow_index, constants.P_precip_index, ti, tr, ro
    ] = converted_data.meteo_data[constants.Snow_precip_index, ti, ro] / dt
    model_variables.g_road_balance_data[
        constants.ice_index, constants.P_precip_index, ti, tr, ro
    ] = 0

    # Sub surface temperature given as weighted sum of surface temperatures
    # when use_subsurface_flag=2 or air temperature =3
    if ti > time_config.min_time and model_flags.use_subsurface_flag == 2:
        model_variables.road_meteo_data[constants.T_sub_index, ti, tr, ro] = (
            model_variables.road_meteo_data[
                constants.T_sub_index, max(1, ti - 1), tr, ro
            ]
            * (1 - dt / model_parameters.sub_surf_average_time)
            + model_variables.road_meteo_data[
                constants.T_s_index, max(1, ti - 1), tr, ro
            ]
            * dt
            / model_parameters.sub_surf_average_time
        )

    if ti > time_config.min_time and model_flags.use_subsurface_flag == 3:
        model_variables.road_meteo_data[constants.T_sub_index, ti, tr, ro] = (
            model_variables.road_meteo_data[
                constants.T_sub_index, max(1, ti - 1), tr, ro
            ]
            * (1 - dt / model_parameters.sub_surf_average_time)
            + converted_data.meteo_data[constants.T_a_index, max(1, ti - 1), ro]
            * dt
            / model_parameters.sub_surf_average_time
        )

    # Ploughing road sinks rate
    R_ploughing[: constants.num_moisture] = (
        -np.log(
            1 - model_parameters.h_ploughing_moisture[: constants.num_moisture] + 0.0001
        )
        / dt
        * converted_data.activity_data[constants.t_ploughing_index, ti, ro]
        * model_flags.use_ploughing_data_flag
    )

    # Wetting production
    model_variables.g_road_balance_data[
        constants.water_index, constants.P_roadwetting_index, ti, tr, ro
    ] = (
        converted_data.activity_data[constants.g_road_wetting_index, ti, ro]
        / dt
        * model_flags.use_wetting_data_flag
    )

    # Calculate aerodynamic resistance
    use_stability = 1
    if use_stability:
        # Build arrays for all vehicle types
        V_veh_array = np.array(
            [
                converted_data.traffic_data[constants.V_veh_index[v], ti, ro]
                for v in range(constants.num_veh)
            ]
        )
        N_v_array = np.array(
            [
                converted_data.traffic_data[constants.N_v_index[v], ti, ro]
                / metadata.n_lanes
                for v in range(constants.num_veh)
            ]
        )

        model_variables.road_meteo_data[constants.r_aero_index, ti, tr, ro] = (
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
                constants.num_veh,
                model_parameters.a_traffic,
            )
        )
        model_variables.road_meteo_data[
            constants.r_aero_notraffic_index, ti, tr, ro
        ] = r_aero_func_with_stability(
            FF_0,
            T_a_0,
            T_s_0,
            metadata.z_FF,
            metadata.z_T,
            model_parameters.z0,
            z0t,
            np.zeros(constants.num_veh),
            np.zeros(constants.num_veh),
            constants.num_veh,
            model_parameters.a_traffic,
        )
    else:
        # Build arrays for all vehicle types
        V_veh_array = np.array(
            [
                converted_data.traffic_data[constants.V_veh_index[v], ti, ro]
                for v in range(constants.num_veh)
            ]
        )
        N_v_array = np.array(
            [
                converted_data.traffic_data[constants.N_v_index[v], ti, ro]
                / metadata.n_lanes
                for v in range(constants.num_veh)
            ]
        )

        model_variables.road_meteo_data[constants.r_aero_index, ti, tr, ro] = (
            r_aero_func(
                converted_data.meteo_data[constants.FF_index, ti, ro],
                metadata.z_FF,
                metadata.z_T,
                model_parameters.z0,
                z0t,
                V_veh_array,
                N_v_array,
                constants.num_veh,
                model_parameters.a_traffic,
            )
        )
        model_variables.road_meteo_data[
            constants.r_aero_notraffic_index, ti, tr, ro
        ] = r_aero_func(
            converted_data.meteo_data[constants.FF_index, ti, ro],
            metadata.z_FF,
            metadata.z_T,
            model_parameters.z0,
            z0t,
            np.zeros(constants.num_veh),
            np.zeros(constants.num_veh),
            constants.num_veh,
            model_parameters.a_traffic,
        )

    if model_flags.use_traffic_turb_flag == 0:
        model_variables.road_meteo_data[constants.r_aero_index, ti, tr, ro] = (
            model_variables.road_meteo_data[
                constants.r_aero_notraffic_index, ti, tr, ro
            ]
        )

    # Calculate the traffic induced heat flux (W/m2)
    model_variables.road_meteo_data[constants.H_traffic_index, ti, tr, ro] = 0
    if model_flags.use_traffic_turb_flag:
        for v in range(constants.num_veh):
            if (
                converted_data.traffic_data[constants.V_veh_index[v], ti, ro] != 0
                and metadata.n_lanes != 0
            ):
                model_variables.road_meteo_data[
                    constants.H_traffic_index, ti, tr, ro
                ] += model_parameters.H_veh[v] * min(
                    1,
                    length_veh[v]
                    / converted_data.traffic_data[constants.V_veh_index[v], ti, ro]
                    * converted_data.traffic_data[constants.N_v_index[v], ti, ro]
                    / (metadata.n_lanes * 1000),
                )

    # Set surface relative humidity
    if model_flags.surface_humidity_flag == 1:
        model_variables.road_meteo_data[constants.RH_s_index, ti, tr, ro] = (
            min(
                1,
                np.sum(g_road_0_data[: constants.num_moisture])
                / model_parameters.g_road_evaporation_thresh,
            )
            * 100
        )
    elif model_flags.surface_humidity_flag == 2:
        model_variables.road_meteo_data[constants.RH_s_index, ti, tr, ro] = (
            1
            - np.exp(
                -np.sum(g_road_0_data[: constants.num_moisture])
                / model_parameters.g_road_evaporation_thresh
                * 4
            )
        ) * 100
    else:
        model_variables.road_meteo_data[constants.RH_s_index, ti, tr, ro] = 100

    # Calculate evaporation energy balance including melt of snow and ice
    if model_flags.evaporation_flag:
        short_rad_net_temp = model_variables.road_meteo_data[
            constants.short_rad_net_index, ti, tr, ro
        ]

        # Adjust net radiation if snow is present on the road
        if g_road_0_data[constants.snow_index] > dz_snow_albedo:
            short_rad_net_temp = (
                model_variables.road_meteo_data[
                    constants.short_rad_net_index, ti, tr, ro
                ]
                * (1 - model_parameters.albedo_snow)
                / (1 - metadata.albedo_road)
            )

        # Handle forecast mode
        E_corr = 0
        if model_flags.forecast_hour > 0:
            # Energy correction calculations
            if (
                model_flags.use_energy_correction_flag
                and tf is not None
                and tf > time_config.min_time + 1
                and tf == ti
            ):
                (
                    model_variables.road_meteo_data[constants.E_index, tf - 1, tr, ro],
                    model_variables.road_meteo_data[
                        constants.E_correction_index, tf, tr, ro
                    ],
                    model_variables.road_meteo_data[
                        constants.T_s_index, tf - 1, tr, ro
                    ],
                ) = e_diff_func(
                    model_variables.road_meteo_data[
                        constants.road_temperature_obs_index, tf - 1, tr, ro
                    ],
                    model_variables.road_meteo_data[
                        constants.T_s_index, tf - 2, tr, ro
                    ],
                    converted_data.meteo_data[constants.T_a_index, tf - 1, ro],
                    model_variables.road_meteo_data[
                        constants.T_sub_index, tf - 1, tr, ro
                    ],
                    model_variables.road_meteo_data[
                        constants.E_correction_index, tf - 1, tr, ro
                    ],
                    converted_data.meteo_data[constants.pressure_index, tf - 1, ro],
                    model_parameters.dzs,
                    dt,
                    model_variables.road_meteo_data[
                        constants.r_aero_index, tf - 1, tr, ro
                    ],
                    short_rad_net_temp,
                    converted_data.meteo_data[constants.long_rad_in_index, tf - 1, ro],
                    model_variables.road_meteo_data[
                        constants.H_traffic_index, tf - 1, tr, ro
                    ],
                    model_variables.road_meteo_data[constants.L_index, tf - 1, tr, ro],
                    model_variables.road_meteo_data[
                        constants.G_freeze_index, tf - 1, tr, ro
                    ],
                    model_variables.road_meteo_data[
                        constants.G_melt_index, tf - 1, tr, ro
                    ],
                    model_parameters.sub_surf_param,
                    model_flags.use_subsurface_flag,
                )

            # Forecast temperature initialization
            if tf is not None and tf > time_config.min_time + 1 and tf == ti:
                forecast_idx = tf + model_flags.forecast_hour
                if forecast_idx < len(
                    model_variables.road_temperature_forecast_missing
                ):
                    model_variables.road_temperature_forecast_missing[forecast_idx] = 1

                # Check if observed temperature is available
                if meteorology_input and hasattr(
                    meteorology_input, "road_temperature_obs_missing"
                ):
                    if (tf - 1) not in meteorology_input.road_temperature_obs_missing:
                        model_variables.original_bias_T_s = (
                            model_variables.road_meteo_data[
                                constants.T_s_index, tf - 1, tr, ro
                            ]
                            - model_variables.road_meteo_data[
                                constants.road_temperature_obs_index, tf - 1, tr, ro
                            ]
                        )
                        if model_flags.use_observed_temperature_init_flag:
                            model_variables.road_meteo_data[
                                constants.T_s_index, tf - 1, tr, ro
                            ] = model_variables.road_meteo_data[
                                constants.road_temperature_obs_index, tf - 1, tr, ro
                            ]
                        if forecast_idx < len(
                            model_variables.road_temperature_forecast_missing
                        ):
                            model_variables.road_temperature_forecast_missing[
                                forecast_idx
                            ] = 0

            if model_flags.use_energy_correction_flag and tf is not None:
                E_corr = model_variables.road_meteo_data[
                    constants.E_correction_index, tf, tr, ro
                ] * relax_func(dt, (ti - tf) + 1)

            T_s_0 = model_variables.road_meteo_data[constants.T_s_index, ti - 1, tr, ro]

        # Call surface energy model
        (
            model_variables.road_meteo_data[constants.T_s_index, ti, tr, ro],
            model_variables.road_meteo_data[constants.T_melt_index, ti, tr, ro],
            model_variables.road_meteo_data[constants.RH_salt_final_index, ti, tr, ro],
            model_variables.road_meteo_data[constants.RH_s_index, ti, tr, ro],
            model_variables.road_salt_data[
                constants.dissolved_ratio_index, :, ti, tr, ro
            ],
            model_variables.road_meteo_data[constants.evap_index, ti, tr, ro],
            model_variables.road_meteo_data[constants.evap_pot_index, ti, tr, ro],
            S_melt_temp,
            model_variables.g_road_balance_data[
                constants.ice_index, constants.P_freeze_index, ti, tr, ro
            ],
            model_variables.road_meteo_data[constants.H_index, ti, tr, ro],
            model_variables.road_meteo_data[constants.L_index, ti, tr, ro],
            model_variables.road_meteo_data[constants.G_index, ti, tr, ro],
            model_variables.road_meteo_data[constants.long_rad_out_index, ti, tr, ro],
            model_variables.road_meteo_data[constants.long_rad_net_index, ti, tr, ro],
            model_variables.road_meteo_data[constants.rad_net_index, ti, tr, ro],
            model_variables.road_meteo_data[constants.G_sub_index, ti, tr, ro],
            model_variables.road_meteo_data[constants.G_freeze_index, ti, tr, ro],
            model_variables.road_meteo_data[constants.G_melt_index, ti, tr, ro],
        ) = surface_energy_model_4_func(
            short_rad_net_temp,
            converted_data.meteo_data[constants.long_rad_in_index, ti, ro],
            model_variables.road_meteo_data[constants.H_traffic_index, ti, tr, ro],
            model_variables.road_meteo_data[constants.r_aero_index, ti, tr, ro],
            converted_data.meteo_data[constants.T_a_index, ti, ro],
            T_s_0,
            model_variables.road_meteo_data[constants.T_sub_index, ti, tr, ro],
            converted_data.meteo_data[constants.RH_index, ti, ro],
            model_variables.road_meteo_data[constants.RH_s_index, ti, tr, ro],
            RH_s_0,
            converted_data.meteo_data[constants.pressure_index, ti, ro],
            model_parameters.dzs,
            dt,
            g_road_0_data[constants.water_index],
            g_road_0_data[constants.ice_index] + g_road_0_data[constants.snow_index],
            model_parameters.g_road_evaporation_thresh,
            M2_road_salt_0,
            input_activity.salt_type,
            model_parameters.sub_surf_param,
            model_flags.surface_humidity_flag,
            model_flags.use_subsurface_flag,
            model_flags.use_salt_humidity_flag,
            E_corr,
        )

        # Redistribute melting between snow and ice
        if np.sum(g_road_0_data[constants.snow_ice_index]) > 0:
            model_variables.g_road_balance_data[
                constants.snow_ice_index, constants.S_melt_index, ti, tr, ro
            ] = model_variables.g_road_balance_data[
                constants.snow_ice_index, constants.S_melt_index, ti, tr, ro
            ] + S_melt_temp * g_road_0_data[constants.snow_ice_index] / np.sum(
                g_road_0_data[constants.snow_ice_index]
            )

    # Calculate surface dewpoint temperature
    model_variables.road_meteo_data[constants.T_s_dewpoint_index, ti, tr, ro] = (
        dewpoint_from_rh_func(
            model_variables.road_meteo_data[constants.T_s_index, ti, tr, ro],
            model_variables.road_meteo_data[constants.RH_s_index, ti, tr, ro],
        )
    )

    converted_data.meteo_data[constants.T_dewpoint_index, ti, ro] = (
        dewpoint_from_rh_func(
            converted_data.meteo_data[constants.T_a_index, ti, ro],
            converted_data.meteo_data[constants.RH_index, ti, ro],
        )
    )

    # Set the evaporation/condensation rates
    total_moisture = np.sum(g_road_0_data[: constants.num_moisture])
    if total_moisture > 0:
        g_road_fraction[: constants.num_moisture] = (
            g_road_0_data[: constants.num_moisture] / total_moisture
        )
    else:
        g_road_fraction[: constants.num_moisture] = 1.0 / constants.num_moisture

    if (
        model_variables.road_meteo_data[constants.evap_index, ti, tr, ro] > 0
    ):  # Evaporation
        for m in range(constants.num_moisture):
            if g_road_0_data[m] > 0:
                R_evaporation[m] = (
                    model_variables.road_meteo_data[constants.evap_index, ti, tr, ro]
                    / g_road_0_data[m]
                    * g_road_fraction[m]
                )
        model_variables.g_road_balance_data[
            : constants.num_moisture, constants.P_evap_index, ti, tr, ro
        ] = 0
    else:  # Condensation
        if (
            model_variables.road_meteo_data[constants.T_s_index, ti, tr, ro]
            >= model_variables.road_meteo_data[constants.T_melt_index, ti, tr, ro]
        ):  # Condensation to water
            model_variables.g_road_balance_data[
                constants.water_index, constants.P_evap_index, ti, tr, ro
            ] = -model_variables.road_meteo_data[constants.evap_index, ti, tr, ro]
            model_variables.g_road_balance_data[
                constants.snow_index, constants.P_evap_index, ti, tr, ro
            ] = 0
            model_variables.g_road_balance_data[
                constants.ice_index, constants.P_evap_index, ti, tr, ro
            ] = 0
        elif model_flags.evaporation_flag == 2:
            # Condensation to snow
            model_variables.g_road_balance_data[
                constants.snow_index, constants.P_evap_index, ti, tr, ro
            ] = -model_variables.road_meteo_data[constants.evap_index, ti, tr, ro]
            model_variables.g_road_balance_data[
                constants.water_index, constants.P_evap_index, ti, tr, ro
            ] = 0
            model_variables.g_road_balance_data[
                constants.ice_index, constants.P_evap_index, ti, tr, ro
            ] = 0
        elif model_flags.evaporation_flag == 1:
            # Condensation only to ice
            model_variables.g_road_balance_data[
                constants.snow_index, constants.P_evap_index, ti, tr, ro
            ] = 0
            model_variables.g_road_balance_data[
                constants.water_index, constants.P_evap_index, ti, tr, ro
            ] = 0
            model_variables.g_road_balance_data[
                constants.ice_index, constants.P_evap_index, ti, tr, ro
            ] = -model_variables.road_meteo_data[constants.evap_index, ti, tr, ro]
        R_evaporation[: constants.num_moisture] = 0

    model_variables.g_road_balance_data[
        : constants.num_moisture, constants.S_evap_index, ti, tr, ro
    ] = (
        R_evaporation[: constants.num_moisture]
        * g_road_0_data[: constants.num_moisture]
    )

    # Set drainage rates
    if model_flags.drainage_type_flag == 1:
        # Exponential drainage based on time scale tau_road_drainage
        g_road_water_drainable = max(
            0,
            g_road_0_data[constants.water_index]
            - model_parameters.g_road_drainable_min,
        )
        g_road_drainable_withrain = max(
            0,
            converted_data.meteo_data[constants.Rain_precip_index, ti, ro]
            + g_road_0_data[constants.water_index]
            - model_parameters.g_road_drainable_min,
        )

        if g_road_drainable_withrain > 0:
            R_drainage[constants.water_index] = 1 / model_parameters.tau_road_drainage
        else:
            R_drainage[constants.water_index] = 0

        # Diagnostic only
        model_variables.g_road_balance_data[
            constants.water_index, constants.S_drainage_tau_index, ti, tr, ro
        ] = g_road_drainable_withrain * R_drainage[constants.water_index]

    if model_flags.drainage_type_flag == 2:
        R_drainage[constants.water_index] = 0
        model_variables.g_road_balance_data[
            constants.water_index, constants.S_drainage_tau_index, ti, tr, ro
        ] = 0

    if model_flags.drainage_type_flag == 3:
        # Combined drainage
        g_road_water_drainable = max(
            0,
            g_road_0_data[constants.water_index]
            - model_parameters.g_road_drainable_thresh,
        )
        g_road_drainable_withrain = max(
            0,
            converted_data.meteo_data[constants.Rain_precip_index, ti, ro]
            + g_road_0_data[constants.water_index]
            - model_parameters.g_road_drainable_min,
        )

        if g_road_drainable_withrain == 0 and g_road_water_drainable > 0:
            R_drainage[constants.water_index] = 1 / model_parameters.tau_road_drainage
        else:
            R_drainage[constants.water_index] = 0

        # Diagnostic only
        model_variables.g_road_balance_data[
            constants.water_index, constants.S_drainage_tau_index, ti, tr, ro
        ] = g_road_water_drainable * R_drainage[constants.water_index]

    # Set the drainage rate for dust/salt module
    model_variables.g_road_balance_data[
        constants.water_index, constants.R_drainage_index, ti, tr, ro
    ] = R_drainage[constants.water_index]

    # Splash and spray sinks and production
    R_spray[: constants.num_moisture] = 0
    for m in range(constants.num_moisture):
        g_road_sprayable = max(
            0, g_road_0_data[m] - model_parameters.g_road_sprayable_min[m]
        )
        if g_road_sprayable > 0 and model_flags.water_spray_flag:
            for v in range(constants.num_veh):
                R_spray[m] += (
                    converted_data.traffic_data[constants.N_v_index[v], ti, ro]
                    / metadata.n_lanes
                    * model_parameters.veh_track[tr]
                    * f_spray_func(
                        model_parameters.R_0_spray[v, m],
                        converted_data.traffic_data[constants.V_veh_index[v], ti, ro],
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

        model_variables.g_road_balance_data[m, constants.S_spray_index, ti, tr, ro] = (
            R_spray[m] * g_road_0_data[m]
        )
        model_variables.g_road_balance_data[m, constants.R_spray_index, ti, tr, ro] = (
            R_spray[m]
        )

    # Add production terms
    model_variables.g_road_balance_data[
        : constants.num_moisture, constants.P_total_index, ti, tr, ro
    ] = (
        model_variables.g_road_balance_data[
            : constants.num_moisture, constants.P_precip_index, ti, tr, ro
        ]
        + model_variables.g_road_balance_data[
            : constants.num_moisture, constants.P_evap_index, ti, tr, ro
        ]
        + model_variables.g_road_balance_data[
            : constants.num_moisture, constants.P_roadwetting_index, ti, tr, ro
        ]
    )

    # Add sink rate terms
    R_total = R_evaporation + R_drainage + R_spray + R_ploughing

    # Calculate change in water and snow using mass balance
    for m in range(constants.num_moisture):
        model_variables.g_road_data[m, ti, tr, ro] = mass_balance_func(
            g_road_0_data[m],
            model_variables.g_road_balance_data[m, constants.P_total_index, ti, tr, ro],
            R_total[m],
            dt,
        )

    # Recalculate spray and evaporation diagnostics based on average moisture
    for m in range(constants.num_moisture):
        model_variables.g_road_balance_data[m, constants.S_spray_index, ti, tr, ro] = (
            R_spray[m]
            * (model_variables.g_road_data[m, ti, tr, ro] + g_road_0_data[m])
            / 2
        )
        model_variables.g_road_balance_data[m, constants.S_evap_index, ti, tr, ro] = (
            R_evaporation[m]
            * (model_variables.g_road_data[m, ti, tr, ro] + g_road_0_data[m])
            / 2
        )

    # Remove and add snow melt after the rest of the calculations
    for m in constants.snow_ice_index:
        model_variables.g_road_balance_data[m, constants.S_melt_index, ti, tr, ro] = (
            min(
                model_variables.g_road_data[m, ti, tr, ro] / dt,
                model_variables.g_road_balance_data[
                    m, constants.S_melt_index, ti, tr, ro
                ],
            )
        )

    # Sink of melt is the same as production of water
    model_variables.g_road_balance_data[
        constants.water_index, constants.P_melt_index, ti, tr, ro
    ] = np.sum(
        model_variables.g_road_balance_data[
            constants.snow_ice_index, constants.S_melt_index, ti, tr, ro
        ]
    )

    for m in range(constants.num_moisture):
        model_variables.g_road_data[m, ti, tr, ro] = max(
            0,
            model_variables.g_road_data[m, ti, tr, ro]
            - model_variables.g_road_balance_data[m, constants.S_melt_index, ti, tr, ro]
            * dt,
        )
        model_variables.g_road_data[m, ti, tr, ro] = (
            model_variables.g_road_data[m, ti, tr, ro]
            + model_variables.g_road_balance_data[m, constants.P_melt_index, ti, tr, ro]
            * dt
        )

    # Remove water through drainage for drainage_type_flag=2 or 3
    g_road_water_drainable = 0
    if model_flags.drainage_type_flag == 2 or model_flags.drainage_type_flag == 3:
        if retain_water_by_snow:
            g_road_drainable_min_temp = max(
                model_parameters.g_road_drainable_min,
                model_variables.g_road_data[constants.snow_index, ti, tr, ro],
            )
        else:
            g_road_drainable_min_temp = model_parameters.g_road_drainable_min

        g_road_water_drainable = max(
            0,
            model_variables.g_road_data[constants.water_index, ti, tr, ro]
            - g_road_drainable_min_temp,
        )
        model_variables.g_road_data[constants.water_index, ti, tr, ro] = min(
            model_variables.g_road_data[constants.water_index, ti, tr, ro],
            g_road_drainable_min_temp,
        )
        model_variables.g_road_balance_data[
            constants.water_index, constants.S_drainage_index, ti, tr, ro
        ] = g_road_water_drainable / dt

    # Freeze after the rest of the calculations
    # Limit the amount of freezing to the amount of available water
    model_variables.g_road_balance_data[
        constants.ice_index, constants.P_freeze_index, ti, tr, ro
    ] = (
        min(
            model_variables.g_road_data[constants.water_index, ti, tr, ro],
            model_variables.g_road_balance_data[
                constants.ice_index, constants.P_freeze_index, ti, tr, ro
            ]
            * dt,
        )
        / dt
    )

    model_variables.g_road_balance_data[
        constants.water_index, constants.S_freeze_index, ti, tr, ro
    ] = model_variables.g_road_balance_data[
        constants.ice_index, constants.P_freeze_index, ti, tr, ro
    ]

    model_variables.g_road_data[constants.water_index, ti, tr, ro] = (
        model_variables.g_road_data[constants.water_index, ti, tr, ro]
        - model_variables.g_road_balance_data[
            constants.water_index, constants.S_freeze_index, ti, tr, ro
        ]
    )
    model_variables.g_road_data[constants.ice_index, ti, tr, ro] = (
        model_variables.g_road_data[constants.ice_index, ti, tr, ro]
        + model_variables.g_road_balance_data[
            constants.ice_index, constants.P_freeze_index, ti, tr, ro
        ]
    )

    # Set moisture content to be always >= 0 (avoiding round off errors)
    for m in range(constants.num_moisture):
        model_variables.g_road_data[m, ti, tr, ro] = max(
            0, model_variables.g_road_data[m, ti, tr, ro]
        )

    # Calculate inhibition/retention factors
    g_road_total = np.sum(
        model_variables.g_road_data[: constants.num_moisture, ti, tr, ro]
    )
    g_ratio_road = (
        g_road_total - model_parameters.g_retention_min[constants.road_index]
    ) / (
        model_parameters.g_retention_thresh[constants.road_index]
        - model_parameters.g_retention_min[constants.road_index]
    )
    g_ratio_brake = (
        model_variables.g_road_data[constants.water_index, ti, tr, ro]
        - model_parameters.g_retention_min[constants.brake_index]
    ) / (
        model_parameters.g_retention_thresh[constants.brake_index]
        - model_parameters.g_retention_min[constants.brake_index]
    )
    g_ratio_binder = (
        M2_road_salt_0[1] - model_parameters.g_retention_min[constants.salt_index[1]]
    ) / (
        model_parameters.g_retention_thresh[constants.salt_index[1]]
        - model_parameters.g_retention_min[constants.salt_index[1]]
    )

    if model_flags.retention_flag == 1:
        model_variables.f_q[: constants.num_source, ti, tr, ro] = np.maximum(
            0, np.minimum(1, 1 - g_ratio_road)
        )
        model_variables.f_q[: constants.num_source, ti, tr, ro] = (
            np.maximum(0, np.minimum(1, 1 - g_ratio_binder))
            * model_variables.f_q[: constants.num_source, ti, tr, ro]
        )
        model_variables.f_q[constants.brake_index, ti, tr, ro] = max(
            0, min(1, 1 - g_ratio_brake)
        )
    elif model_flags.retention_flag == 2:
        model_variables.f_q[: constants.num_source, ti, tr, ro] = np.exp(
            -2 * np.maximum(0, g_ratio_road)
        )
        model_variables.f_q[: constants.num_source, ti, tr, ro] = (
            np.exp(-2 * np.maximum(0, g_ratio_binder))
            * model_variables.f_q[: constants.num_source, ti, tr, ro]
        )
        model_variables.f_q[constants.brake_index, ti, tr, ro] = np.exp(
            -2 * max(0, g_ratio_brake)
        )
    else:
        model_variables.f_q[: constants.num_source, ti, tr, ro] = 1.0

    # Set observed retention parameter if available
    if (
        meteorology_input
        and meteorology_input.road_wetness_obs_available
        and meteorology_input.road_wetness_obs_in_mm
    ):
        g_ratio_obs = (
            converted_data.meteo_data[constants.road_wetness_obs_input_index, ti, ro]
            - model_parameters.g_retention_min[constants.road_index]
        ) / (
            model_parameters.g_retention_thresh[constants.road_index]
            - model_parameters.g_retention_min[constants.road_index]
        )

        if (
            converted_data.meteo_data[constants.road_wetness_obs_input_index, ti, ro]
            == metadata.nodata
        ):
            model_variables.f_q_obs[ti, tr, ro] = 1  # No data then the road is dry
        elif model_flags.retention_flag == 1:
            model_variables.f_q_obs[ti, tr, ro] = max(0, min(1, 1 - g_ratio_obs))
        elif model_flags.retention_flag == 2:
            model_variables.f_q_obs[ti, tr, ro] = np.exp(-2 * max(0, g_ratio_obs))
        else:
            model_variables.f_q_obs[ti, tr, ro] = 1

    if (
        meteorology_input
        and meteorology_input.road_wetness_obs_available
        and meteorology_input.road_wetness_obs_in_mm == 0
    ):
        # Handle observed moisture cutoff value
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
            model_variables.road_meteo_data[
                constants.road_wetness_obs_index, ti, tr, ro
            ]
            == metadata.nodata
        ):
            model_variables.f_q_obs[ti, tr, ro] = 1  # No data then dry road
        elif (
            model_variables.road_meteo_data[
                constants.road_wetness_obs_index, ti, tr, ro
            ]
            < observed_moisture_cutoff_value_temp
        ):
            model_variables.f_q_obs[ti, tr, ro] = 1
        else:
            model_variables.f_q_obs[ti, tr, ro] = 0

    # Set retention based on observed wetness if required
    if (
        model_flags.use_obs_retention_flag
        and meteorology_input
        and meteorology_input.road_wetness_obs_available
        and model_flags.retention_flag != 0
    ):
        model_variables.f_q[: constants.num_source, ti, tr, ro] = (
            model_variables.f_q_obs[ti, tr, ro]
        )
        model_variables.f_q[constants.brake_index, ti, tr, ro] = (
            1  # No retention for brakes
        )
        model_variables.f_q[constants.exhaust_index, ti, tr, ro] = (
            1  # No retention for exhaust
        )
