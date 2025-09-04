from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

import constants
from config_classes import model_file_paths
from functions import average_data_func
from .shared_plot_data import shared_plot_data
from .helpers import matlab_datenum_to_datetime_array, format_time_axis, mask_nodata


def plot_emissions_mass(shared: shared_plot_data, paths: model_file_paths) -> None:
    """
    Plot figure 3: Emissions and mass balance (3 stacked panels).

    Panels:
    1) Emissions (g/km/hr): Total, Direct dust, Suspended dust, Exhaust
    2) Road mass loading (g/m²): dust, salt (na/mg), sand, non-suspendable sand (/10), optional cleaning stairs
    3) Road dust production and sink rates (g/m²/hr): wear retention, other production, and sinks

    Computes the underlying balance sums that MATLAB prints rely on, without printing.
    """

    # Shorthands
    date_num = shared.date_num
    n_date = shared.date_num.shape[0]
    nodata = shared.nodata
    i_min, i_max = shared.i_min, shared.i_max
    av = list(shared.av)
    x_size = shared.plot_size_fraction
    x_load = constants.pm_200
    b_factor = shared.b_factor

    # Masked local copies
    E_sum = mask_nodata(shared.E_road_data_sum_tracks.copy(), nodata)
    M_sum = mask_nodata(shared.M_road_data_sum_tracks.copy(), nodata)
    MB_sum = mask_nodata(shared.M_road_balance_data_sum_tracks.copy(), nodata)
    activity = mask_nodata(shared.activity_data_ro.copy(), nodata)
    roadm = mask_nodata(shared.road_meteo_weighted.copy(), nodata)

    # ---------------- Panel 1: Emissions ----------------
    y_total = np.maximum(
        E_sum[constants.total_dust_index, x_size, constants.E_total_index, :n_date], 0
    )
    y_direct = np.maximum(
        np.nansum(
            E_sum[
                constants.dust_noexhaust_index,
                x_size,
                constants.E_direct_index,
                :n_date,
            ],
            axis=0,
        ),
        0,
    )
    y_susp = np.maximum(
        E_sum[
            constants.total_dust_index, x_size, constants.E_suspension_index, :n_date
        ],
        0,
    )
    y_exhaust = np.maximum(
        E_sum[constants.exhaust_index, x_size, constants.E_total_index, :n_date], 0
    )

    x_str, xplot, y_total_av = average_data_func(date_num, y_total, i_min, i_max, av)
    _, _, y_direct_av = average_data_func(date_num, y_direct, i_min, i_max, av)
    _, _, y_susp_av = average_data_func(date_num, y_susp, i_min, i_max, av)
    _, _, y_exhaust_av = average_data_func(date_num, y_exhaust, i_min, i_max, av)
    dt_x = matlab_datenum_to_datetime_array(xplot)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 6), sharex=False)
    fig.subplots_adjust(hspace=0.4)
    try:
        fig.canvas.manager.set_window_title("Figure 3: Emissions and mass")  # type: ignore
    except Exception:
        pass

    # Title on first panel
    if hasattr(paths, "title_str") and paths.title_str:
        pm_text = (
            "PM10"
            if x_size == constants.pm_10
            else ("PM2.5" if x_size == constants.pm_25 else "PM")
        )
        ax1.set_title(f"{paths.title_str}: Emissions ({pm_text}) and mass balance")
    else:
        ax1.set_title("Emissions and mass balance")

    ax1.plot(dt_x, y_total_av.squeeze(), "k-", linewidth=1, label="Total")
    ax1.plot(dt_x, y_direct_av.squeeze(), "b-", linewidth=1, label="Direct dust")
    ax1.plot(dt_x, y_susp_av.squeeze(), "r-", linewidth=1, label="Suspended dust")
    ax1.plot(dt_x, y_exhaust_av.squeeze(), "m--", linewidth=1, label="Exhaust")
    ax1.set_ylabel("Emission (g/km/hr)")
    ax1.legend(loc="upper left")
    format_time_axis(ax1, dt_x, shared.av[0], day_tick_limit=150)

    # ---------------- Panel 2: Mass loading ----------------
    y_mass_dust = np.maximum(
        M_sum[constants.total_dust_index, x_load, :n_date] * b_factor, 0
    )
    y_mass_salt_na = np.maximum(
        M_sum[constants.salt_index[0], x_load, :n_date] * b_factor, 0
    )
    y_mass_salt_2 = np.maximum(
        M_sum[constants.salt_index[1], x_load, :n_date] * b_factor, 0
    )
    y_mass_sand = np.maximum(M_sum[constants.sand_index, x_load, :n_date] * b_factor, 0)
    y_mass_sand_nonsusp = np.maximum(
        (
            M_sum[constants.sand_index, constants.pm_all, :n_date]
            - M_sum[constants.sand_index, constants.pm_200, :n_date]
        )
        * b_factor,
        0,
    )

    _, _, y_mass_dust_av = average_data_func(date_num, y_mass_dust, i_min, i_max, av)
    _, _, y_mass_salt_na_av = average_data_func(
        date_num, y_mass_salt_na, i_min, i_max, av
    )
    _, _, y_mass_salt_2_av = average_data_func(
        date_num, y_mass_salt_2, i_min, i_max, av
    )
    _, _, y_mass_sand_av = average_data_func(date_num, y_mass_sand, i_min, i_max, av)
    _, _, y_mass_sand_nonsusp_av = average_data_func(
        date_num, y_mass_sand_nonsusp, i_min, i_max, av
    )

    # Optional cleaning indicator normalized to main series max
    _, _, y_clean_av = average_data_func(
        date_num, activity[constants.t_cleaning_index, :n_date], i_min, i_max, av
    )
    max_plot = np.nanmax(
        np.vstack(
            [
                y_mass_dust_av.squeeze(),
                y_mass_salt_na_av.squeeze(),
                y_mass_sand_av.squeeze(),
            ]
        )
    )
    has_Ts = np.nanmax(np.abs(roadm[constants.T_s_index, :n_date])) != 0

    if np.nanmax(y_clean_av) > 0 and np.isfinite(max_plot) and max_plot > 0 and has_Ts:
        y_clean_norm = y_clean_av.squeeze() / np.nanmax(y_clean_av) * max_plot
        ax2 = ax2  # satisfy linter; ax2 already defined
        ax2.step(
            dt_x, y_clean_norm, where="post", color="b", linewidth=1, label="Cleaning"
        )

    ax2.plot(
        dt_x, y_mass_dust_av.squeeze(), "k-", linewidth=1, label="Suspendable dust"
    )
    ax2.plot(dt_x, y_mass_salt_na_av.squeeze(), "g-", linewidth=1, label="Salt(na)")
    ax2.plot(dt_x, y_mass_salt_2_av.squeeze(), "g--", linewidth=1, label="Salt(mg)")
    ax2.plot(
        dt_x, y_mass_sand_av.squeeze(), "r--", linewidth=1, label="Suspendable sand"
    )
    ax2.plot(
        dt_x,
        (y_mass_sand_nonsusp_av / 10.0).squeeze(),
        "k:",
        linewidth=1,
        label="Non-suspendable sand (/10)",
    )
    ax2.set_ylabel("Mass loading (g/m²)")
    ax2.legend(loc="upper left")
    format_time_axis(ax2, dt_x, shared.av[0], day_tick_limit=150)

    # ---------------- Panel 3: Production and sink rates ----------------
    y_wear_retention = np.nansum(
        MB_sum[constants.wear_index, x_load, constants.P_wear_index, :n_date], axis=0
    )
    y_other_prod = (
        np.nansum(
            MB_sum[
                [constants.fugitive_index, constants.sand_index, constants.depo_index],
                x_load,
                constants.P_depo_index,
                :n_date,
            ],
            axis=0,
        )
        + MB_sum[constants.road_index, x_load, constants.P_abrasion_index, :n_date]
        + np.nansum(
            MB_sum[
                constants.all_source_index, x_load, constants.P_crushing_index, :n_date
            ],
            axis=0,
        )
    )
    y_susp_sink = -MB_sum[
        constants.total_dust_index, x_load, constants.S_suspension_index, :n_date
    ]
    y_drain_sink = -MB_sum[
        constants.total_dust_index, x_load, constants.S_dustdrainage_index, :n_date
    ]
    y_spray_sink = -MB_sum[
        constants.total_dust_index, x_load, constants.S_dustspray_index, :n_date
    ]
    y_clean_sink = -MB_sum[
        constants.total_dust_index, x_load, constants.S_cleaning_index, :n_date
    ]
    y_plough_sink = -MB_sum[
        constants.total_dust_index, x_load, constants.S_dustploughing_index, :n_date
    ]
    y_wind_sink = -MB_sum[
        constants.total_dust_index, x_load, constants.S_windblown_index, :n_date
    ]

    # Convert to g/m²/hr using width factor
    def to_gpm2hr(arr: np.ndarray) -> np.ndarray:
        return np.asarray(arr) * b_factor

    y_wear_retention = to_gpm2hr(y_wear_retention)
    y_other_prod = to_gpm2hr(y_other_prod)
    y_susp_sink = to_gpm2hr(y_susp_sink)
    y_drain_sink = to_gpm2hr(y_drain_sink)
    y_spray_sink = to_gpm2hr(y_spray_sink)
    y_clean_sink = to_gpm2hr(y_clean_sink)
    y_plough_sink = to_gpm2hr(y_plough_sink)
    y_wind_sink = to_gpm2hr(y_wind_sink)

    # Average
    _, _, y_wear_retention_av = average_data_func(
        date_num, y_wear_retention, i_min, i_max, av
    )
    _, _, y_other_prod_av = average_data_func(date_num, y_other_prod, i_min, i_max, av)
    _, _, y_susp_sink_av = average_data_func(date_num, y_susp_sink, i_min, i_max, av)
    _, _, y_drain_sink_av = average_data_func(date_num, y_drain_sink, i_min, i_max, av)
    _, _, y_spray_sink_av = average_data_func(date_num, y_spray_sink, i_min, i_max, av)
    _, _, y_clean_sink_av = average_data_func(date_num, y_clean_sink, i_min, i_max, av)
    _, _, y_plough_sink_av = average_data_func(
        date_num, y_plough_sink, i_min, i_max, av
    )
    _, _, y_wind_sink_av = average_data_func(date_num, y_wind_sink, i_min, i_max, av)

    ax3.plot(
        dt_x, y_wear_retention_av.squeeze(), "k-", linewidth=0.8, label="Wear retention"
    )
    ax3.plot(
        dt_x, y_other_prod_av.squeeze(), "k:", linewidth=0.8, label="Other production"
    )
    ax3.plot(dt_x, y_susp_sink_av.squeeze(), "r-", linewidth=0.8, label="Suspension")
    ax3.plot(dt_x, y_drain_sink_av.squeeze(), "b-", linewidth=0.8, label="Drainage")
    ax3.plot(dt_x, y_spray_sink_av.squeeze(), "m-", linewidth=0.8, label="Spray")
    ax3.plot(dt_x, y_clean_sink_av.squeeze(), "g-", linewidth=0.8, label="Cleaning")
    ax3.plot(dt_x, y_plough_sink_av.squeeze(), "y-", linewidth=0.8, label="Ploughing")
    ax3.plot(dt_x, y_wind_sink_av.squeeze(), "g:", linewidth=0.8, label="Windblown")
    ax3.set_ylabel("Rates (g/m²/hr)")
    format_time_axis(ax3, dt_x, shared.av[0], day_tick_limit=150)

    # Compute sums for legend values (match MATLAB totals)
    mask_range = slice(i_min, i_max + 1)
    sum_P_road_wearsource = float(np.sum(y_wear_retention[mask_range]))
    sum_P_road_allother = float(np.sum(y_other_prod[mask_range]))
    sum_S_suspension = float(np.sum(y_susp_sink[mask_range]))
    sum_S_drainage = float(np.sum(y_drain_sink[mask_range]))
    sum_S_spray = float(np.sum(y_spray_sink[mask_range]))
    sum_S_cleaning = float(np.sum(y_clean_sink[mask_range]))
    sum_S_ploughing = float(np.sum(y_plough_sink[mask_range]))
    sum_S_windblown = float(np.sum(y_wind_sink[mask_range]))

    # Update legend with totals
    handles, _labels = ax3.get_legend_handles_labels()
    legend_text = [
        f"Wear retention = {sum_P_road_wearsource:4.1f} (g/m²)",
        f"Other production = {sum_P_road_allother:4.1f} (g/m²)",
        f"Suspension = {sum_S_suspension:4.1f} (g/m²)",
        f"Drainage = {sum_S_drainage:4.1f} (g/m²)",
        f"Spray = {sum_S_spray:4.1f} (g/m²)",
        f"Cleaning = {sum_S_cleaning:4.1f} (g/m²)",
        f"Ploughing = {sum_S_ploughing:4.1f} (g/m²)",
        f"Windblown = {sum_S_windblown:4.1f} (g/m²)",
    ]
    ax3.legend(handles, legend_text, loc="upper left")

    
    plt.tight_layout()

    # ---------------- Optional textual summaries (post-plot prints) ----------------
    if shared.print_result:
        pm_text_print = (
            "PM10" if x_size == constants.pm_10 else ("PM2.5" if x_size == constants.pm_25 else "PM")
        )
        title_str = paths.title_str
        
        av_label = constants.av_str[shared.av[0] - 1]

        print()
        print(f"{title_str} ({pm_text_print}) {av_label}")
        print("-----------------------------------------------------")

        # Total surface mass budget (g/m^2) over selected period
        # y_* arrays are in g/m^2/hr; integrate by multiplying with dt
        dt = float(shared.dt)
        sum_P_road_wearsource_print = float(np.nansum(y_wear_retention[mask_range])) * dt
        sum_P_road_allother_print = float(np.nansum(y_other_prod[mask_range])) * dt
        sum_S_suspension_print = float(np.nansum(y_susp_sink[mask_range])) * dt
        sum_S_drainage_print = float(np.nansum(y_drain_sink[mask_range])) * dt
        sum_S_spray_print = float(np.nansum(y_spray_sink[mask_range])) * dt
        sum_S_cleaning_print = float(np.nansum(y_clean_sink[mask_range])) * dt
        sum_S_ploughing_print = float(np.nansum(y_plough_sink[mask_range])) * dt
        sum_S_windblown_print = float(np.nansum(y_wind_sink[mask_range])) * dt

        print("Total surface mass budget (g/m^2)")
        print(
            "\t".join(
                [
                    f"{'Wear retention':<18}",
                    f"{'Other production':<18}",
                    f"{'Suspension':<18}",
                    f"{'Drainage':<18}",
                    f"{'Spray':<18}",
                    f"{'Cleaning':<18}",
                    f"{'Ploughing':<18}",
                    f"{'Windblown':<18}",
                ]
            )
        )
        print(
            "\t".join(
                [
                    f"{sum_P_road_wearsource_print:<18.2f}",
                    f"{sum_P_road_allother_print:<18.2f}",
                    f"{sum_S_suspension_print:<18.2f}",
                    f"{sum_S_drainage_print:<18.2f}",
                    f"{sum_S_spray_print:<18.2f}",
                    f"{sum_S_cleaning_print:<18.2f}",
                    f"{sum_S_ploughing_print:<18.2f}",
                    f"{sum_S_windblown_print:<18.2f}",
                ]
            )
        )

        # Total surface salt (NaCl) budget (g/m^2)
        # Work directly on masked raw arrays and scale using dt and b_factor as in MATLAB
        salt_species_index = constants.salt_index[0]
        b_factor_local = float(b_factor)

        sum_salt_application = float(
            np.nansum(
                MB_sum[
                    salt_species_index, constants.pm_all, constants.P_depo_index, mask_range
                ]
            )
        ) * dt * b_factor_local

        sum_S_suspension_salt = float(
            np.nansum(
                -MB_sum[
                    salt_species_index,
                    constants.pm_all,
                    constants.S_suspension_index,
                    mask_range,
                ]
            )
        ) * dt * b_factor_local

        sum_S_emission_salt = float(
            np.nansum(
                -E_sum[
                    salt_species_index, constants.pm_all, constants.E_suspension_index, mask_range
                ]
            )
        ) * dt * b_factor_local

        sum_S_drainage_salt = float(
            np.nansum(
                -MB_sum[
                    salt_species_index,
                    constants.pm_all,
                    constants.S_dustdrainage_index,
                    mask_range,
                ]
            )
        ) * dt * b_factor_local

        sum_S_spray_salt = float(
            np.nansum(
                -MB_sum[
                    salt_species_index,
                    constants.pm_all,
                    constants.S_dustspray_index,
                    mask_range,
                ]
            )
        ) * dt * b_factor_local

        sum_S_cleaning_salt = float(
            np.nansum(
                -MB_sum[
                    salt_species_index,
                    constants.pm_all,
                    constants.S_cleaning_index,
                    mask_range,
                ]
            )
        ) * dt * b_factor_local

        sum_S_ploughing_salt = float(
            np.nansum(
                -MB_sum[
                    salt_species_index,
                    constants.pm_all,
                    constants.S_dustploughing_index,
                    mask_range,
                ]
            )
        ) * dt * b_factor_local

        sum_S_windblown_salt = float(
            np.nansum(
                -MB_sum[
                    salt_species_index,
                    constants.pm_all,
                    constants.S_windblown_index,
                    mask_range,
                ]
            )
        ) * dt * b_factor_local

        print("Total surface salt (NaCl) budget (g/m^2)")
        print(
            "\t".join(
                [
                    f"{'Salt application':<18}",
                    f"{'Suspension':<18}",
                    f"{'Total emission':<18}",
                    f"{'Drainage':<18}",
                    f"{'Spray':<18}",
                    f"{'Cleaning':<18}",
                    f"{'Ploughing':<18}",
                    f"{'Windblown':<18}",
                ]
            )
        )
        print(
            "\t".join(
                [
                    f"{sum_salt_application:<18.2f}",
                    f"{sum_S_suspension_salt:<18.2f}",
                    f"{sum_S_emission_salt:<18.2f}",
                    f"{sum_S_drainage_salt:<18.2f}",
                    f"{sum_S_spray_salt:<18.2f}",
                    f"{sum_S_cleaning_salt:<18.2f}",
                    f"{sum_S_ploughing_salt:<18.2f}",
                    f"{sum_S_windblown_salt:<18.2f}",
                ]
            )
        )
        print()

