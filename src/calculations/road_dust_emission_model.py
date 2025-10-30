"""
NORTRIP model road dust emission model
"""

import numpy as np
import constants
import logging

from functions import (
    w_func,
    f_abrasion_func,
    f_crushing_func,
    f_susroad_func,
    r_0_wind_func,
    mass_balance_func,
)
from initialise import time_config
from input_classes import (
    converted_data,
    input_metadata,
    model_variables,
    input_initial,
    input_airquality,
)
from config_classes import model_parameters, model_flags

logger = logging.getLogger(__name__)


def road_dust_emission_model(
    ti: int,
    tr: int,
    ro: int,
    time_config: time_config,
    converted_data: converted_data,
    model_variables: model_variables,
    model_parameters: model_parameters,
    model_flags: model_flags,
    metadata: input_metadata,
    initial_data: input_initial,
    airquality_data: input_airquality,
):
    """
    Calculate road dust emissions and mass loading.
    This function populates the model_variables class with emission and mass loading data.
    """
    # Allows salt that is dissolved to be suspended and sprayed
    use_dissolved_ratio = 1

    # Conversion factor from g/km to g/m^2
    b_factor = 1 / (1000 * metadata.b_road_lanes * model_parameters.f_track[tr])

    # Set initial mass loading prior to the time step
    M_road_bin_0_data = np.zeros((constants.num_source_all, constants.num_size))
    prev_time = max(time_config.min_time, ti - 1)
    M_road_bin_0_data[:, :] = model_variables.M_road_bin_data[:, :, prev_time, tr, ro]

    # Initialize arrays
    P_wear = np.zeros(constants.num_size)
    E_wear = np.zeros(constants.num_size)
    P_abrasion = np.zeros(constants.num_size)
    E_abrasion = np.zeros(constants.num_size)
    f_abrasion_temp = np.zeros(constants.num_size)
    abrasion_temp = np.zeros(constants.num_size)
    P_crushing = np.zeros(constants.num_size)
    E_crushing = np.zeros(constants.num_size)
    f_crushing_temp = 0.0
    crushing_temp = np.zeros(constants.num_size)

    # ==========================================================================
    # Calculate road production of dust and salt for each track and each road
    # ==========================================================================

    # --------------------------------------------------------------------------
    # Calculate the direct source wear rates for each s, t and v (WR)
    # --------------------------------------------------------------------------
    WR_array = np.zeros((constants.num_wear, constants.num_tyre, constants.num_veh))
    WR_temp = 0.0

    for s in range(constants.num_wear):
        WR_temp = 0.0
        for t in range(constants.num_tyre):
            for v in range(constants.num_veh):
                # Get wear flags
                wear_flags = [
                    model_flags.road_wear_flag,
                    model_flags.tyre_wear_flag,
                    model_flags.brake_wear_flag,
                ]

                # Calculate snow/ice sum for wear threshold
                snow_ice_sum = np.sum(
                    model_variables.g_road_data[constants.snow_ice_index, ti, tr, ro]
                )

                wear_temp = w_func(
                    model_parameters.W_0[s, t, v],
                    model_parameters.h_pave[int(metadata.p_index) - 1],
                    model_parameters.h_drivingcycle[int(metadata.d_index) - 1],
                    converted_data.traffic_data[constants.V_veh_index[v], ti, ro],
                    model_parameters.a_wear[s, :],
                    snow_ice_sum,
                    model_parameters.s_roadwear_thresh,
                    s,
                    constants.road_index,
                    constants.tyre_index,
                    constants.brake_index,
                )

                WR_array[s, t, v] = (
                    converted_data.traffic_data[constants.N_t_v_index[(t, v)], ti, ro]
                    * model_parameters.veh_track[tr]
                    * wear_temp
                    * wear_flags[s]
                )

                # if ti == 2209 and s == 0 and t == 0 and v == 0:
                #     logger.info("--- DEBUGGING WR_time_data CALCULATION ---")
                #     logger.info(f"ti={ti}, s={s}, t={t}, v={v}, tr={tr}, ro={ro}")
                #     logger.info(f"--- Indices used ---")
                #     logger.info(f"metadata.p_index: {metadata.p_index}")
                #     logger.info(f"metadata.d_index: {metadata.d_index}")
                #     logger.info(f"V_veh_index for v={v}: {constants.V_veh_index[v]}")
                #     logger.info(f"N_t_v_index for (t={t}, v={v}): {constants.N_t_v_index[(t, v)]}")
                #     logger.info(f"--- Values ---")
                #     logger.info(f"1. model_parameters.W_0[s, t, v]: {model_parameters.W_0[s, t, v]}")
                #     logger.info(f"2. model_parameters.h_pave[int(metadata.p_index) - 1]: {model_parameters.h_pave[int(metadata.p_index) - 1]}")
                #     logger.info(f"3. model_parameters.h_drivingcycle[int(metadata.d_index) - 1]: {model_parameters.h_drivingcycle[int(metadata.d_index) - 1]}")
                #     logger.info(f"4. converted_data.traffic_data[constants.V_veh_index[v], ti, ro]: {converted_data.traffic_data[constants.V_veh_index[v], ti, ro]}")
                #     logger.info(f"5. model_parameters.a_wear[s, :]: {model_parameters.a_wear[s, :]}")
                #     logger.info(f"6. snow_ice_sum: {snow_ice_sum}")
                #     logger.info(f"7. model_parameters.s_roadwear_thresh: {model_parameters.s_roadwear_thresh}")
                #     logger.info(f"8. wear_temp (from w_func): {wear_temp}")
                #     logger.info(f"9. converted_data.traffic_data[N_t_v_index]: {converted_data.traffic_data[constants.N_t_v_index[(t, v)], ti, ro]}")
                #     logger.info(f"10. model_parameters.veh_track[tr]: {model_parameters.veh_track[tr]}")
                #     logger.info(f"11. wear_flags[s]: {wear_flags[s]}")
                #     logger.info(f"12. WR_array[s, t, v]: {WR_array[s, t, v]}")
                #     logger.info("--- END DEBUGGING ---")

                WR_temp += WR_array[s, t, v]

        model_variables.WR_time_data[s, ti, tr, ro] = WR_temp

    # --------------------------------------------------------------------------
    # Calculate PM fraction speed dependence correction for road wear and PM only
    # --------------------------------------------------------------------------
    f_PM_adjust = np.ones(
        (
            constants.num_source,
            constants.num_size,
            constants.num_tyre,
            constants.num_veh,
        )
    )
    s = constants.road_index
    x_indices = [constants.pm_10, constants.pm_25]

    for v in range(constants.num_veh):
        # Only allow the parameterisation between 20 and 60 km/hr
        V_temp = min(
            60, max(20, converted_data.traffic_data[constants.V_veh_index[v], ti, ro])
        )
        for x in x_indices:
            f_PM_adjust[s, x, :, v] = (1 + model_parameters.c_pm_fraction * V_temp) / (
                1 + model_parameters.c_pm_fraction * model_parameters.V_ref_pm_fraction
            )

    # --------------------------------------------------------------------------
    # Calculate surface mass production due to retention of wear (P_wear)
    # --------------------------------------------------------------------------
    for s in range(constants.num_wear):
        P_wear[:] = 0.0
        E_wear[:] = 0.0

        for t in range(constants.num_tyre):
            for v in range(constants.num_veh):
                P_wear[:] += (
                    WR_array[s, t, v]
                    * (
                        1
                        - model_parameters.f_0_dir[s]
                        * model_variables.f_q[s, ti, tr, ro]
                    )
                    * model_parameters.f_PM_bin[s, :, t]
                    * f_PM_adjust[s, :, t, v]
                )
                E_wear[:] += (
                    WR_array[s, t, v]
                    * model_parameters.f_0_dir[s]
                    * model_parameters.f_PM_bin[s, :, t]
                    * f_PM_adjust[s, :, t, v]
                    * model_variables.f_q[s, ti, tr, ro]
                )

        # Total suspended wear
        model_variables.E_road_bin_data[s, :, constants.E_direct_index, ti, tr, ro] = (
            E_wear[:]
        )

        # Total retained wear distributed over all tracks according to area
        for tr2 in range(model_parameters.num_track):
            model_variables.M_road_bin_balance_data[
                s, :, constants.P_wear_index, ti, tr2, ro
            ] = P_wear[:] * model_parameters.f_track[tr2]

    # --------------------------------------------------------------------------
    # Calculate road production and emission rate due to abrasion (P_abrasion)
    # --------------------------------------------------------------------------
    if model_flags.abrasion_flag:
        P_abrasion[:] = 0.0
        E_abrasion[:] = 0.0
        f_abrasion_temp[:] = 0.0
        WR_temp = 0.0

        for s in range(constants.num_source):
            if model_parameters.p_0_abrasion[s] > 0:
                for t in range(constants.num_tyre):
                    for v in range(constants.num_veh):
                        snow_ice_sum = np.sum(
                            model_variables.g_road_data[
                                constants.snow_ice_index, ti, tr, ro
                            ]
                        )

                        f_abrasion_temp[:] = (
                            f_abrasion_func(
                                model_parameters.f_0_abrasion[t, v],
                                model_parameters.h_pave[int(metadata.p_index) - 1],
                                converted_data.traffic_data[
                                    constants.V_veh_index[v], ti, ro
                                ],
                                snow_ice_sum,
                                model_parameters.V_ref_abrasion,
                                model_parameters.s_roadwear_thresh,
                            )
                            * model_parameters.h_0_abrasion[:]
                        )

                        abrasion_temp[:] = (
                            converted_data.traffic_data[
                                constants.N_t_v_index[(t, v)], ti, ro
                            ]
                            / metadata.n_lanes
                            * model_parameters.veh_track[tr]
                            * f_abrasion_temp[:]
                            * M_road_bin_0_data[s, :]
                        )

                        P_abrasion[:] += abrasion_temp[:] * (
                            1 - model_parameters.f_0_dir[constants.abrasion_index]
                        )
                        E_abrasion[:] += (
                            abrasion_temp[:]
                            * model_parameters.f_0_dir[constants.abrasion_index]
                            * model_variables.f_q[constants.road_index, ti, tr, ro]
                        )
                        WR_temp += np.sum(abrasion_temp[:])

        # Distribute the abrasion to the size bins and tracks
        s = constants.road_index
        for x in range(constants.num_size):
            for tr2 in range(model_parameters.num_track):
                model_variables.M_road_bin_balance_data[
                    s, :, constants.P_abrasion_index, ti, tr2, ro
                ] += (
                    P_abrasion[x]
                    * model_parameters.f_PM_bin[constants.abrasion_index, :, 0]
                    * model_parameters.f_track[tr2]
                )

            model_variables.E_road_bin_data[
                s, :, constants.E_direct_index, ti, tr, ro
            ] += (
                E_abrasion[x]
                * model_parameters.f_PM_bin[constants.abrasion_index, :, 0]
            )

        model_variables.WR_time_data[s, ti, tr, ro] += WR_temp

    # --------------------------------------------------------------------------
    # Calculate production and emission due to crushing (P_crushing)
    # --------------------------------------------------------------------------
    R_crushing = np.zeros((constants.num_source_all, constants.num_size))
    f_crushing_temp = 0.0

    # Initialize crushing balance data
    model_variables.M_road_bin_balance_data[
        :, :, constants.P_crushing_index, ti, :, ro
    ] = 0.0

    if model_flags.crushing_flag:
        for s in range(constants.num_source):
            if model_parameters.p_0_crushing[s] > 0:
                for t in range(constants.num_tyre):
                    for v in range(constants.num_veh):
                        snow_ice_sum = np.sum(
                            model_variables.g_road_data[
                                constants.snow_ice_index, ti, tr, ro
                            ]
                        )

                        f_crushing_temp = (
                            f_crushing_func(
                                model_parameters.f_0_crushing[t, v],
                                converted_data.traffic_data[
                                    constants.V_veh_index[v], ti, ro
                                ],
                                snow_ice_sum,
                                model_parameters.V_ref_crushing,
                                model_parameters.s_roadwear_thresh,
                            )
                            * model_parameters.h_0_crushing[:]
                        )

                        R_crushing[s, :] += (
                            converted_data.traffic_data[
                                constants.N_t_v_index[(t, v)], ti, ro
                            ]
                            / metadata.n_lanes
                            * model_parameters.veh_track[tr]
                            * f_crushing_temp
                        )

                        model_variables.M_road_bin_balance_data[
                            s, :, constants.S_crushing_index, ti, tr, ro
                        ] = R_crushing[s, :] * M_road_bin_0_data[s, :]

                # Distribute the crushing sink to the product in the smaller sizes
                for x in range(constants.num_size - 1):
                    for x2 in range(x + 1, constants.num_size):
                        for tr2 in range(model_parameters.num_track):
                            model_variables.M_road_bin_balance_data[
                                s, x2, constants.P_crushing_index, ti, tr2, ro
                            ] += (
                                model_variables.M_road_bin_balance_data[
                                    s, x, constants.S_crushing_index, ti, tr, ro
                                ]
                                * (
                                    1
                                    - model_parameters.f_0_dir[constants.crushing_index]
                                    * model_variables.f_q[s, ti, tr, ro]
                                )
                                * model_parameters.f_PM_bin[
                                    constants.crushing_index, x2, 0
                                ]
                                / np.sum(
                                    model_parameters.f_PM_bin[
                                        constants.crushing_index, x + 1 :, 0
                                    ]
                                )
                            )

                        model_variables.E_road_bin_data[
                            s, x2, constants.E_direct_index, ti, tr, ro
                        ] += (
                            model_variables.M_road_bin_balance_data[
                                s, x, constants.S_crushing_index, ti, tr, ro
                            ]
                            * model_parameters.f_0_dir[constants.crushing_index]
                            * model_variables.f_q[s, ti, tr, ro]
                            * model_parameters.f_PM_bin[constants.crushing_index, x2, 0]
                            / np.sum(
                                model_parameters.f_PM_bin[
                                    constants.crushing_index, x + 1 :, 0
                                ]
                            )
                        )

    # --------------------------------------------------------------------------
    # Calculate road production flux due to deposition (F_deposition g/(km.m)/hr)
    # --------------------------------------------------------------------------
    if model_flags.dust_deposition_flag:
        if (
            len(airquality_data.PM_background) > ti
            and airquality_data.PM_background[constants.pm_10, ti] != metadata.nodata
        ):
            pm_bg_value = max(0, airquality_data.PM_background[constants.pm_10, ti])
        else:
            pm_bg_value = max(0, 20)  # Default value as in MATLAB

        model_variables.M_road_bin_balance_data[
            constants.depo_index, 1:, constants.P_depo_index, ti, tr, ro
        ] = (
            model_parameters.w_dep[: constants.num_size - 1]
            * model_parameters.f_PM_bin[constants.depo_index, 1:, 0]
            / model_parameters.f_PM_bin[constants.depo_index, constants.pm_10, 0]
            * pm_bg_value
            * 3.6
            * metadata.b_road_lanes
            * model_parameters.f_track[tr]
        )
    else:
        model_variables.M_road_bin_balance_data[
            constants.depo_index, :, constants.P_depo_index, ti, tr, ro
        ] = 0.0

    # --------------------------------------------------------------------------
    # Calculate road production due to sanding (P_sanding)
    # --------------------------------------------------------------------------
    model_variables.M_road_bin_balance_data[
        constants.sand_index, :, constants.P_depo_index, ti, tr, ro
    ] = (
        converted_data.activity_data[constants.M_sanding_index, ti, ro]
        / time_config.dt
        * model_parameters.f_PM_bin[constants.sand_index, :, 0]
        * 1000
        * metadata.b_road_lanes
        * model_parameters.f_track[tr]
        * model_flags.use_sanding_data_flag
    )

    # --------------------------------------------------------------------------
    # Calculate road production due to exhaust deposition (P_exhaust)
    # --------------------------------------------------------------------------
    if (
        airquality_data.EP_emis_available or metadata.exhaust_EF_available
    ) and model_flags.exhaust_flag:
        model_variables.M_road_bin_balance_data[
            constants.exhaust_index, :, constants.P_depo_index, ti, tr, ro
        ] = (
            model_variables.E_road_bin_data[
                constants.exhaust_index, :, constants.E_total_index, ti, tr, ro
            ]
            * model_parameters.f_PM_bin[constants.exhaust_index, :, 0]
            * model_parameters.f_track[tr]
            * (1 - model_parameters.f_0_dir[constants.exhaust_index])
        )
        model_variables.E_road_bin_data[
            constants.exhaust_index, :, constants.E_direct_index, ti, tr, ro
        ] = (
            model_variables.E_road_bin_data[
                constants.exhaust_index, :, constants.E_total_index, ti, tr, ro
            ]
            * model_parameters.f_PM_bin[constants.exhaust_index, :, 0]
            * model_parameters.f_track[tr]
            * model_parameters.f_0_dir[constants.exhaust_index]
        )
    else:
        model_variables.M_road_bin_balance_data[
            constants.exhaust_index, :, constants.P_depo_index, ti, tr, ro
        ] = 0.0
        model_variables.E_road_bin_data[
            constants.exhaust_index, :, constants.E_direct_index, ti, tr, ro
        ] = 0.0

    # --------------------------------------------------------------------------
    # Calculate road production due to fugitive deposition (P_fugitive g/km)
    # --------------------------------------------------------------------------
    model_variables.M_road_bin_balance_data[
        constants.fugitive_index, :, constants.P_depo_index, ti, tr, ro
    ] = (
        (
            initial_data.P_fugitive
            + converted_data.activity_data[constants.M_fugitive_index, ti, ro]
        )
        / time_config.dt
        * model_parameters.f_PM_bin[constants.fugitive_index, :, 0]
        * model_parameters.f_track[tr]
    )

    # --------------------------------------------------------------------------
    # Calculate production of salt (P_salt)
    # --------------------------------------------------------------------------
    for i in range(constants.num_salt):
        salt_flags = [
            model_flags.use_salting_data_1_flag,
            model_flags.use_salting_data_2_flag,
        ]
        model_variables.M_road_bin_balance_data[
            constants.salt_index[i], :, constants.P_depo_index, ti, tr, ro
        ] = (
            converted_data.activity_data[constants.M_salting_index[i], ti, ro]
            / time_config.dt
            * model_parameters.f_PM_bin[constants.salt_index[i], :, 0]
            * 1000
            * metadata.b_road_lanes
            * model_parameters.f_track[tr]
            * salt_flags[i]
        )

    # ==========================================================================
    # Calculate road sinks
    # ==========================================================================

    # --------------------------------------------------------------------------
    # Calculate the suspension emission sink rate from the road (R_suspension)
    # --------------------------------------------------------------------------
    R_suspension = np.zeros((constants.num_source_all, constants.num_size))
    R_suspension_array = np.zeros(constants.num_size)

    for s in range(constants.num_source):
        R_suspension[s, :] = 0.0

        # Check if dissolved ratio should be used
        not_dissolved_ratio_temp = 1.0
        if s == constants.salt_index[0] and use_dissolved_ratio:
            not_dissolved_ratio_temp = (
                1.0
                - model_variables.road_salt_data[
                    constants.dissolved_ratio_index, 0, ti, tr, ro
                ]
            )
        elif s == constants.salt_index[1] and use_dissolved_ratio:
            not_dissolved_ratio_temp = (
                1.0
                - model_variables.road_salt_data[
                    constants.dissolved_ratio_index, 1, ti, tr, ro
                ]
            )

        for t in range(constants.num_tyre):
            for v in range(constants.num_veh):
                # f_0_suspension_temp = model_parameters.h_0_sus[s, :] * f_susroad_func(
                f_0_suspension_temp = metadata.h_sus * f_susroad_func(
                    model_parameters.f_0_suspension[s, :, t, v],
                    converted_data.traffic_data[constants.V_veh_index[v], ti, ro],
                    model_parameters.a_sus,
                )
                R_suspension_array[:] = (
                    converted_data.traffic_data[constants.N_t_v_index[(t, v)], ti, ro]
                    / metadata.n_lanes
                    * model_parameters.veh_track[tr]
                    * f_0_suspension_temp
                    * (
                        model_variables.f_q[s, ti, tr, ro]
                        * model_parameters.h_0_q_road[:]
                        + (1 - model_parameters.h_0_q_road[:])
                    )
                    * not_dissolved_ratio_temp
                    * model_flags.road_suspension_flag
                )
                R_suspension[s, :] += R_suspension_array[:]

        # Diagnose the suspension sink
        model_variables.M_road_bin_balance_data[
            s, :, constants.S_suspension_index, ti, tr, ro
        ] = R_suspension[s, :] * M_road_bin_0_data[s, :]
        # Calculate the emissions. The same as the suspension sink
        model_variables.E_road_bin_data[
            s, :, constants.E_suspension_index, ti, tr, ro
        ] = model_variables.M_road_bin_balance_data[
            s, :, constants.S_suspension_index, ti, tr, ro
        ]

    # --------------------------------------------------------------------------
    # Wind blown dust road sink and emission rate (R_windblown)
    # --------------------------------------------------------------------------
    R_windblown = np.zeros((constants.num_source_all, constants.num_size))

    for s in range(constants.num_source):
        R_windblown[s, constants.pm_sus] = (
            r_0_wind_func(
                converted_data.meteo_data[constants.FF_index, ti, ro],
                model_parameters.tau_wind,
                model_parameters.FF_thresh,
            )
            * model_variables.f_q[s, ti, tr, ro]
            * model_flags.wind_suspension_flag
        )

        model_variables.M_road_bin_balance_data[
            s, constants.pm_sus, constants.S_windblown_index, ti, tr, ro
        ] = R_windblown[s, constants.pm_sus] * M_road_bin_0_data[s, constants.pm_sus]
        model_variables.E_road_bin_data[
            s, :, constants.E_windblown_index, ti, tr, ro
        ] = model_variables.M_road_bin_balance_data[
            s, :, constants.S_windblown_index, ti, tr, ro
        ]

    # --------------------------------------------------------------------------
    # Spray and splash road sink (R_spray)
    # --------------------------------------------------------------------------
    R_spray = np.zeros((constants.num_source_all, constants.num_size))
    dissolved_ratio_temp = 1.0
    h_eff_temp = np.zeros((1, constants.num_size))

    if (
        np.sum(model_variables.g_road_data[: constants.num_moisture, ti, tr, ro]) > 0
        and model_flags.dust_spray_flag
    ):
        for s in range(constants.num_source):
            if s == constants.salt_index[0] and use_dissolved_ratio:
                dissolved_ratio_temp = model_variables.road_salt_data[
                    constants.dissolved_ratio_index, 0, ti, tr, ro
                ]
            elif s == constants.salt_index[1] and use_dissolved_ratio:
                dissolved_ratio_temp = model_variables.road_salt_data[
                    constants.dissolved_ratio_index, 1, ti, tr, ro
                ]
            else:
                dissolved_ratio_temp = 1.0

            h_eff_temp[0, :] = model_parameters.h_eff[
                constants.spraying_eff_index, s, :
            ]
            R_spray[s, :] = (
                np.sum(
                    model_variables.g_road_balance_data[
                        : constants.num_moisture, constants.R_spray_index, ti, tr, ro
                    ]
                )
                * h_eff_temp[0, :]
                * dissolved_ratio_temp
            )

            model_variables.M_road_bin_balance_data[
                s, :, constants.S_dustspray_index, ti, tr, ro
            ] = R_spray[s, :] * M_road_bin_0_data[s, :]

            # Production due to spray for multitracks
            model_variables.M_road_bin_balance_data[
                s, :, constants.P_dustspray_index, ti, tr, ro
            ] = (
                np.sum(
                    model_variables.g_road_balance_data[
                        : constants.num_moisture, constants.P_spray_index, ti, tr, ro
                    ]
                )
                * h_eff_temp[0, :]
                * dissolved_ratio_temp
                * M_road_bin_0_data[s, :]
            )
    else:
        model_variables.M_road_bin_balance_data[
            :, :, constants.S_dustspray_index, ti, tr, ro
        ] = 0.0

    # --------------------------------------------------------------------------
    # Drainage road sink rate (R_drainage)
    # --------------------------------------------------------------------------
    R_drainage = np.zeros((constants.num_source_all, constants.num_size))
    dissolved_ratio_temp = 1.0

    if model_flags.drainage_type_flag == 1 or model_flags.drainage_type_flag == 3:
        snow_limit = model_parameters.snow_dust_drainage_retainment_limit
        if (
            model_variables.g_road_data[constants.snow_index, ti, tr, ro] < snow_limit
            and model_flags.dust_drainage_flag > 0
        ):
            for s in range(constants.num_source):
                if s == constants.salt_index[0] and use_dissolved_ratio:
                    dissolved_ratio_temp = model_variables.road_salt_data[
                        constants.dissolved_ratio_index, 0, ti, tr, ro
                    ]
                elif s == constants.salt_index[1] and use_dissolved_ratio:
                    dissolved_ratio_temp = model_variables.road_salt_data[
                        constants.dissolved_ratio_index, 1, ti, tr, ro
                    ]
                else:
                    dissolved_ratio_temp = 1.0

                R_drainage[s, :] = (
                    model_variables.g_road_balance_data[
                        constants.water_index, constants.R_drainage_index, ti, tr, ro
                    ]
                    * model_parameters.h_eff[constants.drainage_eff_index, s, :]
                    * dissolved_ratio_temp
                )

                model_variables.M_road_bin_balance_data[
                    s, :, constants.S_dustdrainage_index, ti, tr, ro
                ] = R_drainage[s, :] * M_road_bin_0_data[s, :]
        else:
            model_variables.M_road_bin_balance_data[
                :, :, constants.S_dustdrainage_index, ti, tr, ro
            ] = 0.0

    # --------------------------------------------------------------------------
    # Cleaning road sink rate (R_cleaning)
    # --------------------------------------------------------------------------
    R_cleaning = np.zeros((constants.num_source_all, constants.num_size))

    for s in range(constants.num_source):
        R_cleaning[s, :] = (
            -np.log(
                1
                - np.minimum(
                    0.99999,
                    model_parameters.h_eff[constants.cleaning_eff_index, s, :]
                    * converted_data.activity_data[constants.t_cleaning_index, ti, ro],
                )
            )
            / time_config.dt
            * model_flags.use_cleaning_data_flag
        )

        model_variables.M_road_bin_balance_data[
            s, :, constants.S_cleaning_index, ti, tr, ro
        ] = R_cleaning[s, :] * M_road_bin_0_data[s, :]

    # --------------------------------------------------------------------------
    # Ploughing road sink (R_ploughing)
    # --------------------------------------------------------------------------
    R_ploughing = np.zeros((constants.num_source_all, constants.num_size))

    for s in range(constants.num_source):
        R_ploughing[s, :] = (
            -np.log(
                1
                - np.minimum(
                    0.99999,
                    model_parameters.h_eff[constants.ploughing_eff_index, s, :]
                    * converted_data.activity_data[constants.t_ploughing_index, ti, ro],
                )
            )
            / time_config.dt
            * model_flags.use_ploughing_data_flag
            * model_flags.dust_ploughing_flag
        )

        model_variables.M_road_bin_balance_data[
            s, :, constants.S_dustploughing_index, ti, tr, ro
        ] = R_ploughing[s, :] * M_road_bin_0_data[s, :]

    # --------------------------------------------------------------------------
    # Add up the contributions for the road mass and salt production (P_road)
    # --------------------------------------------------------------------------
    model_variables.M_road_bin_balance_data[
        :, :, constants.P_dusttotal_index, ti, tr, ro
    ] = (
        model_variables.M_road_bin_balance_data[
            :, :, constants.P_wear_index, ti, tr, ro
        ]
        + model_variables.M_road_bin_balance_data[
            :, :, constants.P_abrasion_index, ti, tr, ro
        ]
        + model_variables.M_road_bin_balance_data[
            :, :, constants.P_crushing_index, ti, tr, ro
        ]
        + model_variables.M_road_bin_balance_data[
            :, :, constants.P_depo_index, ti, tr, ro
        ]
    )

    # --------------------------------------------------------------------------
    # Add up all the road sink rates (R_total)
    # --------------------------------------------------------------------------
    R_total = (
        R_drainage
        + R_cleaning
        + R_ploughing
        + R_spray
        + R_crushing
        + R_suspension
        + R_windblown
    )

    # --------------------------------------------------------------------------
    # Calculate mass balance for the road
    # --------------------------------------------------------------------------
    for s in range(constants.num_source):
        for x in range(constants.num_size):
            model_variables.M_road_bin_data[s, x, ti, tr, ro] = mass_balance_func(
                M_road_bin_0_data[s, x],
                model_variables.M_road_bin_balance_data[
                    s, x, constants.P_dusttotal_index, ti, tr, ro
                ],
                R_total[s, x],
                time_config.dt,
            )

    # Diagnose sinks
    model_variables.M_road_bin_balance_data[
        :, :, constants.S_dusttotal_index, ti, tr, ro
    ] = R_total * M_road_bin_0_data

    # --------------------------------------------------------------------------
    # Remove mass through drainage using drainage type = 2 or 3
    # --------------------------------------------------------------------------
    drain_factor = 1.0
    dissolved_ratio_temp = 1.0
    h_eff_temp = np.zeros((1, constants.num_size))

    if model_flags.drainage_type_flag == 2 or model_flags.drainage_type_flag == 3:
        g_road_drainable_min = 0.1  # g_road_drainable_min equivalent
        g_road_drainable_min = model_parameters.g_road_drainable_min
        drain_factor = (
            model_variables.g_road_balance_data[
                constants.water_index, constants.S_drainage_index, ti, tr, ro
            ]
            * time_config.dt
            / (
                g_road_drainable_min
                + model_variables.g_road_balance_data[
                    constants.water_index, constants.S_drainage_index, ti, tr, ro
                ]
                * time_config.dt
            )
        )

        for s in range(constants.num_source):
            if s == constants.salt_index[0] and use_dissolved_ratio:
                dissolved_ratio_temp = model_variables.road_salt_data[
                    constants.dissolved_ratio_index, 0, ti, tr, ro
                ]
            elif s == constants.salt_index[1] and use_dissolved_ratio:
                dissolved_ratio_temp = model_variables.road_salt_data[
                    constants.dissolved_ratio_index, 1, ti, tr, ro
                ]
            else:
                dissolved_ratio_temp = 1.0

            model_variables.M_road_bin_balance_data[
                s, :, constants.S_dustdrainage_index, ti, tr, ro
            ] = 0.0
            h_eff_temp[0, :] = model_parameters.h_eff[
                constants.drainage_eff_index, s, :
            ]

            if model_flags.dust_drainage_flag == 1:
                model_variables.M_road_bin_balance_data[
                    s, :, constants.S_dustdrainage_index, ti, tr, ro
                ] = (
                    model_variables.M_road_bin_data[s, :, ti, tr, ro]
                    * dissolved_ratio_temp
                    * h_eff_temp[0, :]
                    * drain_factor
                    / time_config.dt
                )
            elif model_flags.dust_drainage_flag == 2:
                model_variables.M_road_bin_balance_data[
                    s, :, constants.S_dustdrainage_index, ti, tr, ro
                ] = (
                    model_variables.M_road_bin_data[s, :, ti, tr, ro]
                    * dissolved_ratio_temp
                    * (1 - np.exp(-h_eff_temp[0, :] * drain_factor))
                    / time_config.dt
                )

            model_variables.M_road_bin_data[s, :, ti, tr, ro] -= (
                model_variables.M_road_bin_balance_data[
                    s, :, constants.S_dustdrainage_index, ti, tr, ro
                ]
                * time_config.dt
            )

            model_variables.M_road_bin_balance_data[
                s, :, constants.S_dusttotal_index, ti, tr, ro
            ] += model_variables.M_road_bin_balance_data[
                s, :, constants.S_dustdrainage_index, ti, tr, ro
            ]

    # --------------------------------------------------------------------------
    # Remove any negative values in mass (round off errors)
    # --------------------------------------------------------------------------
    for s in range(constants.num_source):
        for x in range(constants.num_size):
            model_variables.M_road_bin_data[s, x, ti, tr, ro] = max(
                0.0, model_variables.M_road_bin_data[s, x, ti, tr, ro]
            )

    # --------------------------------------------------------------------------
    # Calculate the final total road dust loadings. Not including salt
    # --------------------------------------------------------------------------
    model_variables.M_road_bin_data[constants.total_dust_index, :, ti, tr, ro] = np.sum(
        model_variables.M_road_bin_data[constants.dust_index, :, ti, tr, ro], axis=0
    )
    model_variables.M_road_bin_balance_data[
        constants.total_dust_index, :, :, ti, tr, ro
    ] = np.sum(
        model_variables.M_road_bin_balance_data[constants.dust_index, :, :, ti, tr, ro],
        axis=0,
    )

    # ==========================================================================
    # Calculate binned emissions
    # ==========================================================================

    # --------------------------------------------------------------------------
    # Total emissions for each source
    # --------------------------------------------------------------------------
    model_variables.E_road_bin_data[:, :, constants.E_total_index, ti, tr, ro] = (
        model_variables.E_road_bin_data[:, :, constants.E_direct_index, ti, tr, ro]
        + model_variables.E_road_bin_data[
            :, :, constants.E_suspension_index, ti, tr, ro
        ]
        + model_variables.E_road_bin_data[:, :, constants.E_windblown_index, ti, tr, ro]
    )

    # --------------------------------------------------------------------------
    # Total dust emissions (i.e. including salt)
    # --------------------------------------------------------------------------
    model_variables.E_road_bin_data[constants.total_dust_index, :, :, ti, tr, ro] = (
        np.sum(
            model_variables.E_road_bin_data[: constants.num_source, :, :, ti, tr, ro],
            axis=0,
        )
    )
