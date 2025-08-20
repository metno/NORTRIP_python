from __future__ import annotations


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

    # Select size fraction and set labels
    x = shared.plot_size_fraction
    xlabel_text = "Date"  # first panel always labels date

    # Build temporary masked copies aligned with MATLAB logic (mask nodata by NaN)
    n_date = shared.date_num.shape[0]
    nodata = shared.nodata

    # Model concentrations by source: sum over sources for total
    # C_data_sum_tracks shape: [num_source_all, num_size, num_process, n_date]
    C_data_temp2 = np.array(shared.C_data_sum_tracks, copy=True)

    # Observations: PM_obs_net per size [num_size, n_date]
    PM_obs_net_temp = np.array(shared.PM_obs_net, copy=True)

    # Mask times with missing either f_conc or obs for this fraction
    f_conc = shared.f_conc
    bad = np.where((np.isnan(f_conc)) | (PM_obs_net_temp[x, :n_date] == nodata))[0]
    if bad.size:
        C_data_temp2[:, :, :, bad] = np.nan
        PM_obs_net_temp[x, bad] = np.nan

    # Compute averaged series using the configured averaging flag
    av = list(shared.av)
    date_num = shared.date_num

    # Total modeled concentration for size x
    y_total = np.nansum(
        C_data_temp2[constants.all_source_index, x, constants.C_total_index, :n_date],
        axis=0,
    )
    y_total = np.maximum(y_total, 0)
    x_str, xplot, yplot_total = average_data_func(date_num, y_total, 0, n_date - 1, av)

    # Observed net
    y_obs = PM_obs_net_temp[x, :n_date]
    y_obs = np.maximum(y_obs, 0)
    _, _, yplot_obs = average_data_func(date_num, y_obs, 0, n_date - 1, av)

    # Salt na and second salt
    y_salt_na = C_data_temp2[
        constants.salt_index[0], x, constants.C_total_index, :n_date
    ]
    y_salt_na = np.maximum(y_salt_na, 0)
    y_salt_2 = C_data_temp2[
        constants.salt_index[1], x, constants.C_total_index, :n_date
    ]
    y_salt_2 = np.maximum(y_salt_2, 0)
    _, _, yplot_salt_na = average_data_func(date_num, y_salt_na, 0, n_date - 1, av)
    _, _, yplot_salt_2 = average_data_func(date_num, y_salt_2, 0, n_date - 1, av)

    # Wear (road + tyre + brake)
    y_wear = np.nansum(
        C_data_temp2[constants.wear_index, x, constants.C_total_index, :n_date], axis=0
    )
    y_wear = np.maximum(y_wear, 0)
    _, _, yplot_wear = average_data_func(date_num, y_wear, 0, n_date - 1, av)

    # Sand
    y_sand = C_data_temp2[constants.sand_index, x, constants.C_total_index, :n_date]
    y_sand = np.maximum(y_sand, 0)
    _, _, yplot_sand = average_data_func(date_num, y_sand, 0, n_date - 1, av)

    # Convert MATLAB datenums to Python datetimes for axis formatting
    dt_x = matlab_datenum_to_datetime_array(xplot)

    # Determine PM text for title
    if x == constants.pm_10:
        pm_text = "PM10"
    elif x == constants.pm_25:
        pm_text = "PM2.5"
    elif x == constants.pm_200:
        pm_text = "PM200"
    else:
        pm_text = "PM"

    # Plot figure with two stacked panels (first: concentrations, second: mass loading)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), sharex=False)
    fig.subplots_adjust(hspace=0.25)

    # Panel 1: concentrations
    # Match MATLAB: title([title_str,': ',pm_text]) on the first subplot
    if hasattr(paths, "title_str") and paths.title_str:
        ax1.set_title(f"{paths.title_str}: {pm_text}")
    else:
        ax1.set_title(f"{pm_text}")
    ax1.set_xlabel(xlabel_text)
    ax1.set_ylabel("Concentration (ug/m³)")

    # Lines mimic MATLAB styles (approximate)
    ax1.plot(dt_x, yplot_obs.squeeze(), "k--", linewidth=1, label="Observed")
    ax1.plot(
        dt_x, yplot_salt_na.squeeze(), "g-", linewidth=1, label="Modelled salt(na)"
    )
    ax1.plot(
        dt_x, yplot_salt_2.squeeze(), "g--", linewidth=1, label="Modelled salt(mg)"
    )
    ax1.plot(dt_x, yplot_wear.squeeze(), "r:", linewidth=1, label="Modelled wear")
    ax1.plot(dt_x, yplot_sand.squeeze(), "m--", linewidth=1, label="Modelled sand")
    ax1.plot(dt_x, yplot_total.squeeze(), "b-", linewidth=1, label="Modelled total")

    ax1.legend(loc="upper left")
    ax1.grid(False)
    # Format x-axis to show months like MATLAB
    format_time_axis(ax1, dt_x, shared.av[0], day_tick_limit=150)
    # Zoom x-limits to the selected time window (min_time..max_time)
    # using the averaged x-range indices
    if len(dt_x) > 0:
        dt_min = matlab_datenum_to_datetime_array([shared.date_num[shared.i_min]])[0]
        dt_max = matlab_datenum_to_datetime_array([shared.date_num[shared.i_max]])[0]
        ax1.set_xlim(dt_min, dt_max)
        # Ensure tick labels are visible on the top panel even with sharex

    # Panel 2: mass loading (g/m^2)
    x_load = constants.pm_200
    y_mass_dust = np.maximum(
        shared.M_road_data_sum_tracks[constants.total_dust_index, x_load, :n_date]
        * shared.b_factor,
        0,
    )
    y_mass_salt_na = np.maximum(
        shared.M_road_data_sum_tracks[constants.salt_index[0], x_load, :n_date]
        * shared.b_factor,
        0,
    )
    y_mass_salt_2 = np.maximum(
        shared.M_road_data_sum_tracks[constants.salt_index[1], x_load, :n_date]
        * shared.b_factor,
        0,
    )
    y_mass_sand = np.maximum(
        shared.M_road_data_sum_tracks[constants.sand_index, x_load, :n_date]
        * shared.b_factor,
        0,
    )

    _s, xplot2, y_mass_dust_av = average_data_func(
        date_num, y_mass_dust, 0, n_date - 1, av
    )
    _s, _xp2, y_mass_salt_na_av = average_data_func(
        date_num, y_mass_salt_na, 0, n_date - 1, av
    )
    _s, _xp3, y_mass_salt_2_av = average_data_func(
        date_num, y_mass_salt_2, 0, n_date - 1, av
    )
    _s, _xp4, y_mass_sand_av = average_data_func(
        date_num, y_mass_sand, 0, n_date - 1, av
    )
    dt_x2 = matlab_datenum_to_datetime_array(xplot2)

    ax2.set_title("Mass loading")
    ax2.set_xlabel(xlabel_text)
    ax2.set_ylabel("Mass loading (g/m²)")

    # Optional cleaning indicator as normalized step to max of main series
    y_clean = shared.activity_data_ro[constants.t_cleaning_index, :n_date]
    _s, _xp5, y_clean_av = average_data_func(date_num, y_clean, 0, n_date - 1, av)
    max_plot = np.nanmax(
        np.vstack(
            [
                y_mass_dust_av.squeeze(),
                y_mass_salt_na_av.squeeze(),
                y_mass_sand_av.squeeze(),
            ]
        )
    )
    legend_entries = ["Suspendable dust", "Salt(na)", "Salt(mg)", "Suspendable sand"]
    if np.nanmax(y_clean_av) > 0 and max_plot > 0:
        y_clean_norm = y_clean_av.squeeze() / np.nanmax(y_clean_av) * max_plot
        ax2.step(
            dt_x2, y_clean_norm, where="post", color="b", linewidth=1, label="Cleaning"
        )
        legend_entries = ["Cleaning"] + legend_entries

    ax2.plot(
        dt_x2, y_mass_dust_av.squeeze(), "k-", linewidth=1, label="Suspendable dust"
    )
    ax2.plot(dt_x2, y_mass_salt_na_av.squeeze(), "g-", linewidth=1, label="Salt(na)")
    ax2.plot(dt_x2, y_mass_salt_2_av.squeeze(), "g--", linewidth=1, label="Salt(2)")
    ax2.plot(
        dt_x2, y_mass_sand_av.squeeze(), "r--", linewidth=1, label="Suspendable sand"
    )

    ax2.legend(legend_entries, loc="upper left")
    ax2.grid(False)
    format_time_axis(ax2, dt_x2, shared.av[0], day_tick_limit=150)
    if len(dt_x2) > 0:
        dt_min = matlab_datenum_to_datetime_array([shared.date_num[shared.i_min]])[0]
        dt_max = matlab_datenum_to_datetime_array([shared.date_num[shared.i_max]])[0]
        ax2.set_xlim(dt_min, dt_max)

    plt.tight_layout()
    plt.show()
