"""
NORTRIP model road dust emission model
"""

import numpy as np
import constants

from functions import (
    w_func,
    f_abrasion_func,
    f_crushing_func,
    f_susroad_func,
    r_0_wind_func,
    mass_balance_func,
)
from initialise import time_config, model_variables
from input_classes import (
    converted_data,
    input_metadata,
    input_initial,
    input_airquality,
)
from config_classes import model_parameters, model_flags


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
    The following model_variables are modified:
    - M_road_bin_data
    - M_road_bin_balance_data
    - E_road_bin_data
    - WR_time_data
    """
    # Allows salt that is dissolved to be suspended and sprayed
    use_dissolved_ratio = 1

    # Set the mass loading prior to the time step
    prev_ti = max(time_config.min_time, ti - 1)
    M_road_bin_0_data = model_variables.M_road_bin_data[:, :, prev_ti, tr, ro].copy()

    # --- Calculate road production of dust and salt ---

    # Calculate direct source wear rates (WR)
    WR_array = np.zeros((constants.num_wear, constants.num_tyre, constants.num_veh))
    wear_flag = [
        model_flags.road_wear_flag,
        model_flags.tyre_wear_flag,
        model_flags.brake_wear_flag,
    ]
    for s in range(constants.num_wear):
        WR_temp = 0
        for t in range(constants.num_tyre):
            for v in range(constants.num_veh):
                snow_ice_sum = np.sum(
                    model_variables.g_road_data[
                        constants.snow_ice_index, ti, tr, ro
                    ]
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

                # Get traffic data for a specific tyre and vehicle
                traffic_key = (t, v)
                traffic_idx = constants.N_t_v_index.get(traffic_key)

                if ti == 2210 and s == 0 and t == 0 and v == 0:
                    print("--- DEBUGGING WR_time_data CALCULATION ---")
                    print(f"ti={ti}, s={s}, t={t}, v={v}, tr={tr}, ro={ro}")
                    print("--- Indices used ---")
                    print(f"p_index: {metadata.p_index}")
                    print(f"d_index: {metadata.d_index}")
                    print(f"V_veh_index for v={v}: {constants.V_veh_index[v]}")
                    print(f"N_t_v_index for (t={t}, v={v}): {traffic_idx}")
                    print("--- Values ---")
                    print(f"1. W_0(s,t,v): {model_parameters.W_0[s, t, v]}")
                    print(
                        "2. h_pave(p_index): "
                        f"{model_parameters.h_pave[int(metadata.p_index) - 1]}"
                    )
                    print(
                        "3. h_drivingcycle(d_index): "
                        f"{model_parameters.h_drivingcycle[int(metadata.d_index) - 1]}"
                    )
                    print(
                        "4. traffic_data(V_veh_index(v),ti,ro): "
                        f"{converted_data.traffic_data[constants.V_veh_index[v], ti, ro]}"
                    )
                    print(f"5. a_wear(s,:): {model_parameters.a_wear[s, :]}")
                    print(f"6. snow_ice_sum: {snow_ice_sum}")
                    print(
                        "7. s_roadwear_thresh: "
                        f"{model_parameters.s_roadwear_thresh}"
                    )
                    print(f"8. wear_temp (from W_func): {wear_temp}")
                    print(
                        f"9. traffic_data(N_t_v_index(t,v),ti,ro): "
                        f"{converted_data.traffic_data[traffic_idx, ti, ro]}"
                    )
                    print(f"10. veh_track(tr): {model_parameters.veh_track[tr]}")
                    print(f"11. wear_flag(s): {wear_flag[s]}")

                WR_array[s, t, v] = (
                    converted_data.traffic_data[traffic_idx, ti, ro]
                    * model_parameters.veh_track[tr]
                    * wear_temp
                    * wear_flag[s]
                )
                if ti == 2210 and s == 0 and t == 0 and v == 0:
                    print(f"12. WR_array(s,t,v): {WR_array[s, t, v]}")
                    print("--- END DEBUGGING ---")
                WR_temp += WR_array[s, t, v]
        model_variables.WR_time_data[s, ti, tr, ro] = WR_temp

    # Calculate PM fraction speed dependence correction
    f_PM_adjust = np.ones(
        (
            constants.num_source,
            constants.num_size,
            constants.num_tyre,
            constants.num_veh,
        )
    )
    s = constants.road_index
    x = [constants.pm_10, constants.pm_25]
    for v in range(constants.num_veh):
        V_temp = min(
            60,
            max(20, converted_data.traffic_data[constants.V_veh_index[v], ti, ro]),
        )
        f_PM_adjust[s, x, :, v] = (1 + model_parameters.c_pm_fraction * V_temp) / (
            1 + model_parameters.c_pm_fraction * model_parameters.V_ref_pm_fraction
        )

    # Calculate surface mass production due to retention of wear (P_wear)
    for s in range(constants.num_wear):
        P_wear = np.zeros(constants.num_size)
        E_wear = np.zeros(constants.num_size)
        for t in range(constants.num_tyre):
            for v in range(constants.num_veh):
                P_wear += (
                    WR_array[s, t, v]
                    * (
                        1
                        - model_parameters.f_0_dir[s]
                        * model_variables.f_q[s, ti, tr, ro]
                    )
                    * model_parameters.f_PM_bin[s, :, t]
                    * f_PM_adjust[s, :, t, v]
                )
                E_wear += (
                    WR_array[s, t, v]
                    * model_parameters.f_0_dir[s]
                    * model_parameters.f_PM_bin[s, :, t]
                    * f_PM_adjust[s, :, t, v]
                    * model_variables.f_q[s, ti, tr, ro]
                )

        model_variables.E_road_bin_data[s, :, constants.E_direct_index, ti, tr, ro] = (
            E_wear
        )
        for tr2 in range(model_parameters.num_track):
            model_variables.M_road_bin_balance_data[
                s, :, constants.P_wear_index, ti, tr2, ro
            ] = P_wear * model_parameters.f_track[tr2]

    # Abrasion
    if model_flags.abrasion_flag:
        P_abrasion = np.zeros(constants.num_size)
        E_abrasion = np.zeros(constants.num_size)
        WR_temp = 0
        for s in range(constants.num_source):
            if model_parameters.p_0_abrasion[s] > 0:
                for t in range(constants.num_tyre):
                    for v in range(constants.num_veh):
                        f_abrasion_temp = (
                            f_abrasion_func(
                                model_parameters.f_0_abrasion[t, v],
                                model_parameters.h_pave[metadata.p_index - 1],
                                converted_data.traffic_data[
                                    constants.V_veh_index[v], ti, ro
                                ],
                                np.sum(
                                    model_variables.g_road_data[
                                        constants.snow_ice_index, ti, tr, ro
                                    ]
                                ),
                                model_parameters.V_ref_abrasion,
                                model_parameters.s_roadwear_thresh,
                            )
                            * model_parameters.h_0_abrasion
                        )

                        traffic_key = (t, v)
                        traffic_idx = constants.N_t_v_index.get(traffic_key)

                        abrasion_temp = (
                            converted_data.traffic_data[traffic_idx, ti, ro]
                            / metadata.n_lanes
                            * model_parameters.veh_track[tr]
                            * f_abrasion_temp
                            * M_road_bin_0_data[s, :]
                        )

                        P_abrasion += abrasion_temp * (
                            1 - model_parameters.f_0_dir[constants.abrasion_index]
                        )
                        E_abrasion += (
                            abrasion_temp
                            * model_parameters.f_0_dir[constants.abrasion_index]
                            * model_variables.f_q[constants.road_index, ti, tr, ro]
                        )
                        WR_temp += np.sum(abrasion_temp)

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

    # Crushing
    R_crushing = np.zeros((constants.num_source, constants.num_size))
    if model_flags.crushing_flag:
        model_variables.M_road_bin_balance_data[
            :, :, constants.P_crushing_index, ti, :, ro
        ] = 0
        for s in range(constants.num_source):
            if model_parameters.p_0_crushing[s] > 0:
                for t in range(constants.num_tyre):
                    for v in range(constants.num_veh):
                        f_crushing_temp = (
                            f_crushing_func(
                                model_parameters.f_0_crushing[t, v],
                                converted_data.traffic_data[
                                    constants.V_veh_index[v], ti, ro
                                ],
                                np.sum(
                                    model_variables.g_road_data[
                                        constants.snow_ice_index, ti, tr, ro
                                    ]
                                ),
                                model_parameters.V_ref_crushing,
                                model_parameters.s_roadwear_thresh,
                            )
                            * model_parameters.h_0_crushing
                        )

                        traffic_key = (t, v)
                        traffic_idx = constants.N_t_v_index.get(traffic_key)

                        R_crushing[s, :] += (
                            converted_data.traffic_data[traffic_idx, ti, ro]
                            / metadata.n_lanes
                            * model_parameters.veh_track[tr]
                            * f_crushing_temp
                        )

                model_variables.M_road_bin_balance_data[
                    s, :, constants.S_crushing_index, ti, tr, ro
                ] = R_crushing[s, :] * M_road_bin_0_data[s, :]

                for x in range(constants.num_size - 1):
                    sum_f_PM_bin = np.sum(
                        model_parameters.f_PM_bin[constants.crushing_index, x + 1 :, 0]
                    )
                    if sum_f_PM_bin > 0:
                        for x2 in range(x + 1, constants.num_size):
                            crushing_sink = model_variables.M_road_bin_balance_data[
                                s, x, constants.S_crushing_index, ti, tr, ro
                            ]
                            crushing_retention = (
                                1
                                - model_parameters.f_0_dir[constants.crushing_index]
                                * model_variables.f_q[s, ti, tr, ro]
                            )
                            f_PM_ratio = (
                                model_parameters.f_PM_bin[
                                    constants.crushing_index, x2, 0
                                ]
                                / sum_f_PM_bin
                            )

                            for tr2 in range(model_parameters.num_track):
                                model_variables.M_road_bin_balance_data[
                                    s, x2, constants.P_crushing_index, ti, tr2, ro
                                ] += crushing_sink * crushing_retention * f_PM_ratio

                            model_variables.E_road_bin_data[
                                s, x2, constants.E_direct_index, ti, tr, ro
                            ] += (
                                crushing_sink
                                * model_parameters.f_0_dir[constants.crushing_index]
                                * model_variables.f_q[s, ti, tr, ro]
                                * f_PM_ratio
                            )

        model_variables.WR_time_data[s, ti, tr, ro] += np.sum(R_crushing)

    # Deposition
    if (
        airquality_data.PM_background.shape[1] > ti
        and airquality_data.PM_background[constants.pm_10, ti] != metadata.nodata
        and model_flags.dust_deposition_flag
    ):
        pm10_bg = max(0, airquality_data.PM_background[constants.pm_10, ti])
        f_pm_ratio = model_parameters.f_PM_bin[constants.depo_index, constants.pm_10, 0]
        if f_pm_ratio > 0:
            model_variables.M_road_bin_balance_data[
                constants.depo_index,
                1 : constants.num_size,
                constants.P_depo_index,
                ti,
                tr,
                ro,
            ] = (
                model_parameters.w_dep[0 : constants.num_size - 1]
                * model_parameters.f_PM_bin[
                    constants.depo_index, 1 : constants.num_size, 0
                ]
                / f_pm_ratio
                * pm10_bg
                * 3.6
                * metadata.b_road_lanes
                * model_parameters.f_track[tr]
            )
    else:
        model_variables.M_road_bin_balance_data[
            constants.depo_index, :, constants.P_depo_index, ti, tr, ro
        ] = 0

    # Sanding production
    if model_flags.use_sanding_data_flag:
        model_variables.M_road_bin_balance_data[
            constants.sand_index, :, constants.P_depo_index, ti, tr, ro
        ] = (
            converted_data.activity_data[constants.M_sanding_index, ti, ro]
            / time_config.dt
            * model_parameters.f_PM_bin[constants.sand_index, :, 0]
            * 1000
            * metadata.b_road_lanes
            * model_parameters.f_track[tr]
        )

    # Exhaust deposition
    if (
        airquality_data.EP_emis_available or metadata.exhaust_EF_available
    ) and model_flags.exhaust_flag:
        total_exhaust_emission = model_variables.E_road_bin_data[
            constants.exhaust_index, :, constants.E_total_index, ti, tr, ro
        ]
        model_variables.M_road_bin_balance_data[
            constants.exhaust_index, :, constants.P_depo_index, ti, tr, ro
        ] = (
            total_exhaust_emission
            * model_parameters.f_PM_bin[constants.exhaust_index, :, 0]
            * model_parameters.f_track[tr]
            * (1 - model_parameters.f_0_dir[constants.exhaust_index])
        )
        model_variables.E_road_bin_data[
            constants.exhaust_index, :, constants.E_direct_index, ti, tr, ro
        ] = (
            total_exhaust_emission
            * model_parameters.f_PM_bin[constants.exhaust_index, :, 0]
            * model_parameters.f_track[tr]
            * model_parameters.f_0_dir[constants.exhaust_index]
        )
    else:
        model_variables.M_road_bin_balance_data[
            constants.exhaust_index, :, constants.P_depo_index, ti, tr, ro
        ] = 0
        model_variables.E_road_bin_data[
            constants.exhaust_index, :, constants.E_direct_index, ti, tr, ro
        ] = 0

    # Fugitive deposition
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

    # Salt production
    use_salting_data_flag = [
        model_flags.use_salting_data_1_flag,
        model_flags.use_salting_data_2_flag,
    ]
    for i in range(constants.num_salt):
        if use_salting_data_flag[i]:
            model_variables.M_road_bin_balance_data[
                constants.salt_index[i], :, constants.P_depo_index, ti, tr, ro
            ] = (
                converted_data.activity_data[constants.M_salting_index[i], ti, ro]
                / time_config.dt
                * model_parameters.f_PM_bin[constants.salt_index[i], :, 0]
                * 1000
                * metadata.b_road_lanes
                * model_parameters.f_track[tr]
            )

    # --- Calculate road sinks ---
    R_suspension = np.zeros((constants.num_source, constants.num_size))
    for s in range(constants.num_source):
        not_dissolved_ratio_temp = 1.0
        if use_dissolved_ratio:
            if s == constants.salt_index[0]:
                not_dissolved_ratio_temp = (
                    1.0
                    - model_variables.road_salt_data[
                        constants.dissolved_ratio_index, 0, ti, tr, ro
                    ]
                )
            elif s == constants.salt_index[1]:
                not_dissolved_ratio_temp = (
                    1.0
                    - model_variables.road_salt_data[
                        constants.dissolved_ratio_index, 1, ti, tr, ro
                    ]
                )

        for t in range(constants.num_tyre):
            for v in range(constants.num_veh):
                f_0_suspension_temp = metadata.h_sus * f_susroad_func(
                    model_parameters.f_0_suspension[s, :, t, v],
                    converted_data.traffic_data[constants.V_veh_index[v], ti, ro],
                    model_parameters.a_sus,
                )

                traffic_key = (t, v)
                traffic_idx = constants.N_t_v_index.get(traffic_key)

                R_suspension_array = (
                    converted_data.traffic_data[traffic_idx, ti, ro]
                    / metadata.n_lanes
                    * model_parameters.veh_track[tr]
                    * f_0_suspension_temp
                    * (
                        model_variables.f_q[s, ti, tr, ro] * model_parameters.h_0_q_road
                        + (1 - model_parameters.h_0_q_road)
                    )
                    * not_dissolved_ratio_temp
                    * model_flags.road_suspension_flag
                )
                R_suspension[s, :] += R_suspension_array

        model_variables.M_road_bin_balance_data[
            s, :, constants.S_suspension_index, ti, tr, ro
        ] = R_suspension[s, :] * M_road_bin_0_data[s, :]
        model_variables.E_road_bin_data[
            s, :, constants.E_suspension_index, ti, tr, ro
        ] = model_variables.M_road_bin_balance_data[
            s, :, constants.S_suspension_index, ti, tr, ro
        ]

    # Wind blown dust
    R_windblown = np.zeros((constants.num_source, constants.num_size))
    if model_flags.wind_suspension_flag:
        for s in range(constants.num_source):
            R_windblown[s, constants.pm_sus] = (
                r_0_wind_func(
                    converted_data.meteo_data[constants.FF_index, ti, ro],
                    model_parameters.tau_wind,
                    model_parameters.FF_thresh,
                )
                * model_variables.f_q[s, ti, tr, ro]
            )
            model_variables.M_road_bin_balance_data[
                s, constants.pm_sus, constants.S_windblown_index, ti, tr, ro
            ] = (
                R_windblown[s, constants.pm_sus]
                * M_road_bin_0_data[s, constants.pm_sus]
            )
            model_variables.E_road_bin_data[
                s, :, constants.E_windblown_index, ti, tr, ro
            ] = model_variables.M_road_bin_balance_data[
                s, :, constants.S_windblown_index, ti, tr, ro
            ]

    # Spray and splash
    R_spray = np.zeros((constants.num_source, constants.num_size))
    if (
        np.sum(model_variables.g_road_data[:, ti, tr, ro]) > 0
        and model_flags.dust_spray_flag
    ):
        for s in range(constants.num_source):
            dissolved_ratio_temp = 1.0
            if use_dissolved_ratio:
                if s == constants.salt_index[0]:
                    dissolved_ratio_temp = model_variables.road_salt_data[
                        constants.dissolved_ratio_index, 0, ti, tr, ro
                    ]
                elif s == constants.salt_index[1]:
                    dissolved_ratio_temp = model_variables.road_salt_data[
                        constants.dissolved_ratio_index, 1, ti, tr, ro
                    ]

            h_eff_temp = model_parameters.h_eff[constants.spraying_eff_index, s, :]
            R_spray[s, :] = (
                np.sum(
                    model_variables.g_road_balance_data[
                        :, constants.R_spray_index, ti, tr, ro
                    ]
                )
                * h_eff_temp
                * dissolved_ratio_temp
            )
            model_variables.M_road_bin_balance_data[
                s, :, constants.S_dustspray_index, ti, tr, ro
            ] = R_spray[s, :] * M_road_bin_0_data[s, :]
            model_variables.M_road_bin_balance_data[
                s, :, constants.P_dustspray_index, ti, tr, ro
            ] = (
                np.sum(
                    model_variables.g_road_balance_data[
                        :, constants.P_spray_index, ti, tr, ro
                    ]
                )
                * h_eff_temp
                * dissolved_ratio_temp
                * M_road_bin_0_data[s, :]
            )
    else:
        model_variables.M_road_bin_balance_data[
            :, :, constants.S_dustspray_index, ti, tr, ro
        ] = 0

    # Drainage
    R_drainage = np.zeros((constants.num_source, constants.num_size))
    if (
        (model_flags.drainage_type_flag == 1 or model_flags.drainage_type_flag == 3)
        and model_flags.dust_drainage_flag > 0
        and model_variables.g_road_data[constants.snow_index, ti, tr, ro]
        < model_parameters.snow_dust_drainage_retainment_limit
    ):
        for s in range(constants.num_source):
            dissolved_ratio_temp = 1.0
            if use_dissolved_ratio:
                if s == constants.salt_index[0]:
                    dissolved_ratio_temp = model_variables.road_salt_data[
                        constants.dissolved_ratio_index, 0, ti, tr, ro
                    ]
                elif s == constants.salt_index[1]:
                    dissolved_ratio_temp = model_variables.road_salt_data[
                        constants.dissolved_ratio_index, 1, ti, tr, ro
                    ]

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
        ] = 0

    # Cleaning
    R_cleaning = np.zeros((constants.num_source, constants.num_size))
    if model_flags.use_cleaning_data_flag:
        for s in range(constants.num_source):
            eff = np.minimum(
                0.99999,
                model_parameters.h_eff[constants.cleaning_eff_index, s, :]
                * converted_data.activity_data[constants.t_cleaning_index, ti, ro],
            )
            R_cleaning[s, :] = -np.log(1 - eff) / time_config.dt
            model_variables.M_road_bin_balance_data[
                s, :, constants.S_cleaning_index, ti, tr, ro
            ] = R_cleaning[s, :] * M_road_bin_0_data[s, :]

    # Ploughing
    R_ploughing = np.zeros((constants.num_source, constants.num_size))
    if model_flags.use_ploughing_data_flag and model_flags.dust_ploughing_flag:
        for s in range(constants.num_source):
            eff = np.minimum(
                0.99999,
                model_parameters.h_eff[constants.ploughing_eff_index, s, :]
                * converted_data.activity_data[constants.t_ploughing_index, ti, ro],
            )
            R_ploughing[s, :] = -np.log(1 - eff) / time_config.dt
            model_variables.M_road_bin_balance_data[
                s, :, constants.S_dustploughing_index, ti, tr, ro
            ] = R_ploughing[s, :] * M_road_bin_0_data[s, :]

    # Sum up production
    model_variables.M_road_bin_balance_data[
        :, :, constants.P_dusttotal_index, ti, tr, ro
    ] = np.sum(
        model_variables.M_road_bin_balance_data[
            :,
            :,
            [
                constants.P_wear_index,
                constants.P_abrasion_index,
                constants.P_crushing_index,
                constants.P_depo_index,
            ],
            ti,
            tr,
            ro,
        ],
        axis=2,
    )

    # Sum up sink rates
    R_total = (
        R_drainage
        + R_cleaning
        + R_ploughing
        + R_spray
        + R_crushing
        + R_suspension
        + R_windblown
    )

    # Mass balance for the road
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

    model_variables.M_road_bin_balance_data[
        0 : constants.num_source, :, constants.S_dusttotal_index, ti, tr, ro
    ] = R_total * M_road_bin_0_data[0 : constants.num_source, :]

    # Drainage for type 2/3
    if model_flags.drainage_type_flag == 2 or model_flags.drainage_type_flag == 3:
        drainage_sink = (
            model_variables.g_road_balance_data[
                constants.water_index, constants.S_drainage_index, ti, tr, ro
            ]
            * time_config.dt
        )
        drain_factor = drainage_sink / (
            model_parameters.g_road_drainable_min + drainage_sink
        )

        for s in range(constants.num_source):
            dissolved_ratio_temp = 1.0
            if use_dissolved_ratio:
                if s == constants.salt_index[0]:
                    dissolved_ratio_temp = model_variables.road_salt_data[
                        constants.dissolved_ratio_index, 0, ti, tr, ro
                    ]
                elif s == constants.salt_index[1]:
                    dissolved_ratio_temp = model_variables.road_salt_data[
                        constants.dissolved_ratio_index, 1, ti, tr, ro
                    ]

            h_eff_temp = model_parameters.h_eff[constants.drainage_eff_index, s, :]
            drainage_sink_mass = 0
            if model_flags.dust_drainage_flag == 1:
                drainage_sink_mass = (
                    model_variables.M_road_bin_data[s, :, ti, tr, ro]
                    * dissolved_ratio_temp
                    * h_eff_temp
                    * drain_factor
                )
            elif model_flags.dust_drainage_flag == 2:
                drainage_sink_mass = (
                    model_variables.M_road_bin_data[s, :, ti, tr, ro]
                    * dissolved_ratio_temp
                    * (1 - np.exp(-h_eff_temp * drain_factor))
                )

            model_variables.M_road_bin_balance_data[
                s, :, constants.S_dustdrainage_index, ti, tr, ro
            ] = drainage_sink_mass / time_config.dt
            model_variables.M_road_bin_data[s, :, ti, tr, ro] -= drainage_sink_mass
            model_variables.M_road_bin_balance_data[
                s, :, constants.S_dusttotal_index, ti, tr, ro
            ] += model_variables.M_road_bin_balance_data[
                s, :, constants.S_dustdrainage_index, ti, tr, ro
            ]

    # Remove negative mass
    model_variables.M_road_bin_data[:, :, ti, tr, ro] = np.maximum(
        0, model_variables.M_road_bin_data[:, :, ti, tr, ro]
    )

    # Total road dust loadings
    model_variables.M_road_bin_data[constants.total_dust_index, :, ti, tr, ro] = np.sum(
        model_variables.M_road_bin_data[constants.dust_index, :, ti, tr, ro], axis=0
    )
    model_variables.M_road_bin_balance_data[
        constants.total_dust_index, :, :, ti, tr, ro
    ] = np.sum(
        model_variables.M_road_bin_balance_data[constants.dust_index, :, :, ti, tr, ro],
        axis=0,
    )

    # --- Calculate binned emissions ---

    # Total emissions for each source
    model_variables.E_road_bin_data[:, :, constants.E_total_index, ti, tr, ro] = np.sum(
        model_variables.E_road_bin_data[
            :,
            :,
            [
                constants.E_direct_index,
                constants.E_suspension_index,
                constants.E_windblown_index,
            ],
            ti,
            tr,
            ro,
        ],
        axis=2,
    )

    # Total dust emissions (including salt)
    model_variables.E_road_bin_data[constants.total_dust_index, :, :, ti, tr, ro] = (
        np.sum(
            model_variables.E_road_bin_data[0 : constants.num_source, :, :, ti, tr, ro],
            axis=0,
        )
    )
