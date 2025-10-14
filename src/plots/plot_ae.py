from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import os

import constants
from config_classes import model_file_paths
from functions.average_data_func import average_data_func
from .shared_plot_data import shared_plot_data
from .helpers import (
    matlab_datenum_to_datetime_array,
    format_time_axis,
    mask_nodata,
    generate_plot_filename,
)


def plot_ae(shared: shared_plot_data, paths: model_file_paths) -> None:
    """
    Plot figure 8 (AE plot) translated from MATLAB.

    Panels:
      1) PM10 concentrations: observed vs modelled components and total
      2) Mass loading (g/m²): suspendable dust, salt, sand; optional cleaning stairs
      3) Emission factor (g/km/veh): modelled vs observed (derived)

    Notes on MATLAB parity:
    - We mask concentration series where either f_conc or PM_obs(x_size) is nodata
    - Mass loading values are converted to g/m² using the width factor b_factor
    - Emission factor uses total emissions and observed emissions (PM_obs/f_conc)
    - Time averaging uses the shared Average_data_func wrapper
    """

    # Shorthands and local masked copies
    date_num = shared.date_num
    n_date = shared.date_num.shape[0]
    nodata = shared.nodata
    i_min, i_max = shared.i_min, shared.i_max
    av = list(shared.av)
    x_size = shared.plot_size_fraction
    x_load = constants.pm_200
    b_factor = shared.b_factor
    day_tick_limit = 150

    # Concentration arrays (sum across tracks provided by shared)
    C_sum = mask_nodata(shared.C_data_sum_tracks.copy(), nodata)
    PM_obs = mask_nodata(shared.PM_obs_net.copy(), nodata)
    f_conc = np.asarray(shared.f_conc[:n_date], dtype=float).copy()
    f_conc[f_conc == nodata] = np.nan

    # Apply MATLAB-equivalent mask r where f_conc or PM_obs(x_size) are invalid
    invalid_c = np.isnan(f_conc) | np.isnan(PM_obs[x_size, :n_date])
    if np.any(invalid_c):
        C_sum[:, :, :, invalid_c] = np.nan
        PM_obs[x_size, invalid_c] = np.nan

    # Build concentration component series for chosen size fraction
    y_total_series = np.sum(
        C_sum[
            constants.all_source_index,
            x_size,
            constants.C_total_index,
            :n_date,
        ],
        axis=0,
    )
    y_obs_series = PM_obs[x_size, :n_date]
    y_salt_na_series = C_sum[
        constants.salt_index[0], x_size, constants.C_total_index, :n_date
    ]
    y_wear_series = np.sum(
        C_sum[constants.wear_index, x_size, constants.C_total_index, :n_date], axis=0
    )
    y_sand_series = C_sum[
        constants.sand_index, x_size, constants.C_total_index, :n_date
    ]
    y_salt_2_series = C_sum[
        constants.salt_index[1], x_size, constants.C_total_index, :n_date
    ]

    # Average series
    x_str, xplot, y_total = average_data_func(
        date_num, y_total_series, i_min, i_max, av
    )
    _, _, y_obs = average_data_func(date_num, y_obs_series, i_min, i_max, av)
    _, _, y_salt_na = average_data_func(date_num, y_salt_na_series, i_min, i_max, av)
    _, _, y_wear = average_data_func(date_num, y_wear_series, i_min, i_max, av)
    _, _, y_sand = average_data_func(date_num, y_sand_series, i_min, i_max, av)
    _, _, y_salt_2 = average_data_func(date_num, y_salt_2_series, i_min, i_max, av)
    dt_x = matlab_datenum_to_datetime_array(xplot)

    # Mass loading arrays at suspendable size; convert to g/m²
    M_sum = mask_nodata(shared.M_road_data_sum_tracks.copy(), nodata)
    y_mass_dust_series = M_sum[constants.total_dust_index, x_load, :n_date] * b_factor
    y_mass_salt_na_series = M_sum[constants.salt_index[0], x_load, :n_date] * b_factor
    y_mass_sand_series = M_sum[constants.sand_index, x_load, :n_date] * b_factor
    # Optional: dissolved salt and non-suspendable sand shown in comments in MATLAB

    _, _, y_mass_dust = average_data_func(
        date_num, y_mass_dust_series, i_min, i_max, av
    )
    _, _, y_mass_salt_na = average_data_func(
        date_num, y_mass_salt_na_series, i_min, i_max, av
    )
    _, _, y_mass_sand = average_data_func(
        date_num, y_mass_sand_series, i_min, i_max, av
    )

    # Cleaning activity (stairs) normalization to max mass plot if applicable
    activity = mask_nodata(shared.activity_data_ro.copy(), nodata)
    _, _, y_clean = average_data_func(
        date_num, activity[constants.t_cleaning_index, :n_date], i_min, i_max, av
    )

    # Emission factor panel inputs
    E_sum = mask_nodata(shared.E_road_data_sum_tracks.copy(), nodata)
    N_total_series = mask_nodata(shared.traffic_data_ro.copy(), nodata)[
        constants.N_total_index, :n_date
    ]
    E_all_series = E_sum[
        constants.total_dust_index, x_size, constants.E_total_index, :n_date
    ]
    # Observed emissions: PM_obs / f_conc
    with np.errstate(divide="ignore", invalid="ignore"):
        E_obs_series = PM_obs[x_size, :n_date] / f_conc

    _xs_e, xp_e, y_E_all = average_data_func(date_num, E_all_series, i_min, i_max, av)
    _, _, y_E_obs = average_data_func(date_num, E_obs_series, i_min, i_max, av)
    _, _, y_N_total = average_data_func(date_num, N_total_series, i_min, i_max, av)
    with np.errstate(divide="ignore", invalid="ignore"):
        y_ef_mod = np.asarray(y_E_all) / np.asarray(y_N_total)
        y_ef_obs = np.asarray(y_E_obs) / np.asarray(y_N_total)
    dt_x_e = matlab_datenum_to_datetime_array(xp_e)

    # --- Figure and axes layout ---
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(9, 8))
    fig.subplots_adjust(hspace=0.35)
    try:
        fig.canvas.manager.set_window_title("Figure 8: AE plot")  # type: ignore
    except Exception:
        pass

    title_str = paths.title_str or "AE plot"

    # Panel 1: PM10 concentrations
    ax1.set_title(f"{title_str}: PM10 concentrations")
    ax1.plot(dt_x, np.asarray(y_obs).squeeze(), "k--", linewidth=1, label="Observed")
    # Optional components per flags
    if shared.use_sanding_data_flag and np.nanmax(np.abs(y_sand)) > 0:
        ax1.plot(
            dt_x, np.asarray(y_sand).squeeze(), "r:", linewidth=1, label="Modelled sand"
        )
    if (shared.use_salting_data_1_flag or shared.use_salting_data_2_flag) and np.nanmax(
        np.abs(y_salt_na)
    ) > 0:
        ax1.plot(
            dt_x,
            np.asarray(y_salt_na).squeeze(),
            "g:",
            linewidth=1,
            label="Modelled salt",
        )
    ax1.plot(
        dt_x, np.asarray(y_total).squeeze(), "b-", linewidth=1, label="Modelled+exhaust"
    )
    ax1.set_ylabel(r"PM10 concentration ($\mu$g.m$^{-3}$)")
    ax1.legend(loc="best")
    format_time_axis(ax1, dt_x, shared.av[0], day_tick_limit=day_tick_limit)
    if len(dt_x) > 0:
        ax1.set_xlim(dt_x[0], dt_x[-1])
    try:
        y_stack = np.vstack(
            [
                np.asarray(y_total).squeeze(),
                np.asarray(y_obs).squeeze(),
                np.asarray(y_salt_na).squeeze(),
                np.asarray(y_wear).squeeze(),
                np.asarray(y_sand).squeeze(),
            ]
        )
        y_max = float(np.nanmax(y_stack)) * 1.1
        if np.isfinite(y_max):
            ax1.set_ylim(0, y_max)
    except Exception:
        pass

    # Panel 2: Mass loading
    ax2.set_title("Mass loading")
    ax2.plot(
        dt_x,
        np.asarray(y_mass_dust).squeeze(),
        "b-",
        linewidth=1,
        label="Suspendable dust",
    )
    if np.nanmax(np.abs(y_mass_salt_na)) > 0:
        ax2.plot(
            dt_x, np.asarray(y_mass_salt_na).squeeze(), "g:", linewidth=1, label="Salt"
        )
    if np.nanmax(np.abs(y_mass_sand)) > 0:
        ax2.plot(
            dt_x,
            np.asarray(y_mass_sand).squeeze(),
            "r:",
            linewidth=1,
            label="Suspendable sand",
        )
    # Optional cleaning stairs normalized to max
    max_plot_m = float(
        np.nanmax(
            np.vstack(
                [
                    np.asarray(y_mass_dust).squeeze(),
                    np.asarray(y_mass_salt_na).squeeze(),
                    np.asarray(y_mass_sand).squeeze(),
                ]
            )
        )
    )
    if (
        shared.use_cleaning_data_flag
        and np.isfinite(max_plot_m)
        and max_plot_m > 0
        and np.nanmax(y_clean) > 0
    ):
        y_clean_norm = np.asarray(y_clean).squeeze() / np.nanmax(y_clean) * max_plot_m
        ax2.step(
            dt_x, y_clean_norm, where="post", color="b", linewidth=0.5, label="Cleaning"
        )
    ax2.set_ylabel(r"Mass loading (g.m$^{-2}$)")
    ax2.legend(loc="best")
    format_time_axis(ax2, dt_x, shared.av[0], day_tick_limit=day_tick_limit)
    if len(dt_x) > 0:
        ax2.set_xlim(dt_x[0], dt_x[-1])

    # Panel 3: Emission factor (g/km/veh)
    ax3.set_title("Emission factor")
    ax3.plot(
        dt_x_e,
        np.asarray(y_ef_mod).squeeze(),
        "b-",
        linewidth=1,
        label="Modelled emission factor",
    )
    ax3.plot(
        dt_x_e,
        np.asarray(y_ef_obs).squeeze(),
        "k--",
        linewidth=1,
        label="Observed emission factor",
    )
    ax3.set_ylabel(r"Emission factor (g.km$^{-1}$.veh$^{-1}$)")
    ax3.set_xlabel("Date")
    ax3.legend(loc="best")
    format_time_axis(ax3, dt_x_e, shared.av[0], day_tick_limit=day_tick_limit)
    if len(dt_x_e) > 0:
        ax3.set_xlim(dt_x_e[0], dt_x_e[-1])

    plt.tight_layout()

    if shared.save_plots:
        plot_file_name = generate_plot_filename(
            title_str=paths.title_str,
            plot_type_flag=shared.av[0],
            figure_number=8,  # AE plot is figure 8
            plot_name="AE",
            date_num=shared.date_num,
            min_time=shared.i_min,
            max_time=shared.i_max,
        )
        plt.savefig(os.path.join(paths.path_outputfig, plot_file_name))
