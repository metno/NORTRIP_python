import numpy as np
import matplotlib.pyplot as plt

import constants
from functions.average_data_func import average_data_func
from config_classes import model_file_paths
from .shared_plot_data import shared_plot_data
from .helpers import (
    matlab_datenum_to_datetime_array,
    format_time_axis,
)


def plot_summary(shared: shared_plot_data, paths: model_file_paths) -> None:
    """
    Render the first panel of the summary chart (concentrations over time).

    This implements the MATLAB section under plot_figure(13) → first subplot,
    computing averaged series and plotting observed vs source contributions.
    """
    # Shorthands
    date_num = shared.date_num
    n_date = shared.date_num.shape[0]
    nodata = shared.nodata
    i_min, i_max = shared.i_min, shared.i_max
    av = list(shared.av)
    x_size = shared.plot_size_fraction

    # Prepare local float copies and apply MATLAB-equivalent masking
    # Invalid times r: when f_conc == nodata OR PM_obs_net(x, :) == nodata
    f_conc = shared.f_conc[:n_date].copy()
    PM_obs = shared.PM_obs_net.copy()
    invalid_mask = (f_conc == nodata) | (PM_obs[x_size, :n_date] == nodata)

    # Concentration components (sum across tracks already in shared)
    C_sum = shared.C_data_sum_tracks.astype(float).copy()
    C_sum[C_sum == nodata] = np.nan
    # Apply invalid mask across all sources/sizes/processes for those timesteps
    C_sum[:, :, :, invalid_mask] = np.nan

    # Observations (size x only) with same invalid mask
    PM_obs_temp = PM_obs.astype(float).copy()
    PM_obs_temp[PM_obs_temp == nodata] = np.nan
    pm_obs_series = PM_obs_temp[x_size, :n_date]
    pm_obs_series[invalid_mask] = np.nan

    # Build series per MATLAB logic
    # 1) Modelled total: sum over all sources of C_total at chosen size
    y_total_model_series = np.sum(
        C_sum[
            constants.all_source_index,
            x_size,
            constants.C_total_index,
            :n_date,
        ],
        axis=0,
    )
    # 2) Observed (net)
    y_obs_series = pm_obs_series
    # 3) Modelled salt(na)
    y_salt_na_series = C_sum[
        constants.salt_index[0], x_size, constants.C_total_index, :n_date
    ]
    # 4) Modelled wear (sum over wear sources)
    y_wear_series = np.sum(
        C_sum[constants.wear_index, x_size, constants.C_total_index, :n_date], axis=0
    )
    # 5) Modelled sand
    y_sand_series = C_sum[
        constants.sand_index, x_size, constants.C_total_index, :n_date
    ]
    # 6) Modelled salt(mg)
    y_salt_mg_series = C_sum[
        constants.salt_index[1], x_size, constants.C_total_index, :n_date
    ]

    # Average all series using the common averaging function
    x_str, xplot, y_total_model = average_data_func(
        date_num, y_total_model_series, i_min, i_max, av
    )
    _, _, y_obs = average_data_func(date_num, y_obs_series, i_min, i_max, av)
    _, _, y_salt_na = average_data_func(date_num, y_salt_na_series, i_min, i_max, av)
    _, _, y_wear = average_data_func(date_num, y_wear_series, i_min, i_max, av)
    _, _, y_sand = average_data_func(date_num, y_sand_series, i_min, i_max, av)
    _, _, y_salt_mg = average_data_func(date_num, y_salt_mg_series, i_min, i_max, av)

    # Convert MATLAB datenums to datetimes for plotting
    dt_x = matlab_datenum_to_datetime_array(xplot)

    # Figure with 4 stacked panels (we'll fill panel 1 and 2 now)
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=False)
    fig.subplots_adjust(hspace=0.5)
    try:
        fig.canvas.manager.set_window_title("Figure 13: Summary")
    except Exception:
        pass

    # Title text with PM fraction
    if x_size == constants.pm_10:
        pm_text = "PM10"
    elif x_size == constants.pm_25:
        pm_text = "PM2.5"
    else:
        pm_text = "PM"

    title_str = getattr(paths, "title_str", "") or "Summary"
    ax1 = axes[0]
    ax1.set_title(f"{title_str}: {pm_text}")

    # Hide values below zero (set to NaN) to mirror MATLAB not displaying negatives

    # Plot series in the MATLAB order/styles
    ax1.plot(dt_x, y_obs.squeeze(), "k--", linewidth=1, label="Observed")
    ax1.plot(dt_x, y_salt_na.squeeze(), "g-", linewidth=1, label="Modelled salt(na)")
    ax1.plot(dt_x, y_salt_mg.squeeze(), "g--", linewidth=1, label="Modelled salt(mg)")
    ax1.plot(dt_x, y_wear.squeeze(), "r:", linewidth=1, label="Modelled wear")
    ax1.plot(dt_x, y_sand.squeeze(), "m--", linewidth=1, label="Modelled sand")
    ax1.plot(dt_x, y_total_model.squeeze(), "b-", linewidth=1, label="Modelled total")

    # Labels
    ax1.set_ylabel(f"{pm_text} concentration (µg/m³)")
    ax1.set_xlabel(shared.xlabel_text)

    # Axis formatting similar to MATLAB date tick handling
    format_time_axis(ax1, dt_x, shared.av[0], day_tick_limit=150)

    # Legend
    ax1.legend(loc="upper left")

    # y-limit (kept optional to match MATLAB tight behavior; user disabled setting)
    try:
        y_stack = np.vstack(
            [
                y_total_model.squeeze(),
                y_obs.squeeze(),
                y_salt_na.squeeze(),
                y_wear.squeeze(),
                y_sand.squeeze(),
            ]
        )
        y_max = float(np.nanmax(y_stack)) * 1.1
        if np.isfinite(y_max):
            ax1.set_ylim(0, y_max)
    except Exception:
        pass

    # ---------------- Panel 2: Mass loading (g/m²) ----------------
    ax2 = axes[1]
    ax2.set_title("")
    ax2.set_ylabel("Mass loading (g/m²)")

    # Local masked copies
    M_sum = shared.M_road_data_sum_tracks.copy()
    M_sum[M_sum == nodata] = np.nan
    activity = shared.activity_data_ro.copy()
    activity[activity == nodata] = np.nan

    b_factor = shared.b_factor
    x_load = getattr(constants, "pm_200", 0)

    # Build mass series at suspendable size, convert to g/m²
    y_mass_dust = (
        np.asarray(M_sum[constants.total_dust_index, x_load, :n_date]) * b_factor
    )
    y_mass_salt_na = (
        np.asarray(M_sum[constants.salt_index[0], x_load, :n_date]) * b_factor
    )
    y_mass_sand = np.asarray(M_sum[constants.sand_index, x_load, :n_date]) * b_factor
    y_mass_salt_mg = (
        np.asarray(M_sum[constants.salt_index[1], x_load, :n_date]) * b_factor
    )

    # Enforce non-negative for plotting
    y_mass_dust = np.maximum(y_mass_dust, 0)
    y_mass_salt_na = np.maximum(y_mass_salt_na, 0)
    y_mass_sand = np.maximum(y_mass_sand, 0)
    y_mass_salt_mg = np.maximum(y_mass_salt_mg, 0)

    # Average
    x_str2, xplot2, y_dust = average_data_func(date_num, y_mass_dust, i_min, i_max, av)
    _, _, y_salt_na = average_data_func(date_num, y_mass_salt_na, i_min, i_max, av)
    _, _, y_sand = average_data_func(date_num, y_mass_sand, i_min, i_max, av)
    _, _, y_salt_mg = average_data_func(date_num, y_mass_salt_mg, i_min, i_max, av)

    dt_x2 = matlab_datenum_to_datetime_array(xplot2)

    # Cleaning stairs normalization
    _, _, y_clean = average_data_func(
        date_num, activity[constants.t_cleaning_index, :n_date], i_min, i_max, av
    )
    max_plot = float(
        np.nanmax(np.vstack([y_dust.squeeze(), y_salt_na.squeeze(), y_sand.squeeze()]))
    )
    has_clean_raw = np.any(
        np.nan_to_num(activity[constants.t_cleaning_index, :n_date], nan=0.0) != 0
    )
    legend_entries = [
        "Suspendable dust",
        "Salt(na)",
        "Salt(mg)",
        "Suspendable sand",
    ]
    if (
        has_clean_raw
        and np.nanmax(y_clean) > 0
        and np.isfinite(max_plot)
        and max_plot > 0
    ):
        y_clean_norm = y_clean.squeeze() / np.nanmax(y_clean) * max_plot
        ax2.step(
            dt_x2,
            y_clean_norm,
            where="post",
            color="b",
            linewidth=0.5,
            label="Cleaning",
        )
        legend_entries = [
            "Cleaning",
            "Suspendable dust",
            "Salt(na)",
            "Salt(mg)",
            "Suspendable sand",
        ]

    # Plot series
    ax2.plot(dt_x2, y_dust.squeeze(), "k-", linewidth=1, label="Suspendable dust")
    ax2.plot(dt_x2, y_salt_na.squeeze(), "g-", linewidth=1, label="Salt(na)")
    ax2.plot(dt_x2, y_salt_mg.squeeze(), "g--", linewidth=1, label="Salt(mg)")
    ax2.plot(dt_x2, y_sand.squeeze(), "r--", linewidth=1, label="Suspendable sand")
    ax2.set_xlabel(shared.xlabel_text)
    format_time_axis(ax2, dt_x2, shared.av[0], day_tick_limit=150)
    ax2.legend(legend_entries, loc="upper left")

    # y-limit: 0..1.1 * max of plotted series (excluding NaNs)
    try:
        y_stack2 = np.vstack(
            [
                y_dust.squeeze(),
                y_salt_na.squeeze(),
                y_salt_mg.squeeze(),
                y_sand.squeeze(),
            ]
        )
        y_max2 = float(np.nanmax(y_stack2)) * 1.1
        if np.isfinite(y_max2):
            ax2.set_ylim(-0.05, y_max2)
    except Exception:
        pass

    plt.tight_layout()
