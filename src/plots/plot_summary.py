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

    # Figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(10, 3))
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
    ax.set_title(f"{title_str}: {pm_text}")

    # Hide values below zero (set to NaN) to mirror MATLAB not displaying negatives
    def _hide_negative(a: np.ndarray) -> np.ndarray:
        arr = np.asarray(a).copy()
        # arr[arr < 0] = np.nan
        return arr

    y_obs_plot = _hide_negative(y_obs).squeeze()
    y_salt_na_plot = _hide_negative(y_salt_na).squeeze()
    y_salt_mg_plot = _hide_negative(y_salt_mg).squeeze()
    y_wear_plot = _hide_negative(y_wear).squeeze()
    y_sand_plot = _hide_negative(y_sand).squeeze()
    y_total_model_plot = _hide_negative(y_total_model).squeeze()

    # Plot series in the MATLAB order/styles
    ax.plot(dt_x, y_obs_plot, "k--", linewidth=1, label="Observed")
    ax.plot(dt_x, y_salt_na_plot, "g-", linewidth=1, label="Modelled salt(na)")
    ax.plot(dt_x, y_salt_mg_plot, "g--", linewidth=1, label="Modelled salt(mg)")
    ax.plot(dt_x, y_wear_plot, "r:", linewidth=1, label="Modelled wear")
    ax.plot(dt_x, y_sand_plot, "m--", linewidth=1, label="Modelled sand")
    ax.plot(dt_x, y_total_model_plot, "b-", linewidth=1, label="Modelled total")

    # Labels
    ax.set_ylabel(f"{pm_text} concentration (µg/m³)")
    ax.set_xlabel(shared.xlabel_text)

    # Axis formatting similar to MATLAB date tick handling
    format_time_axis(ax, dt_x, shared.av[0], day_tick_limit=150)

    # Legend
    ax.legend(loc="upper left")

    # y-limit (kept optional to match MATLAB tight behavior; user disabled setting)
    try:
        y_stack = np.vstack(
            [y_total_model_plot, y_obs_plot, y_salt_na_plot, y_wear_plot, y_sand_plot]
        )
        y_max = float(np.nanmax(y_stack)) * 1.1
        if np.isfinite(y_max):
            ax.set_ylim(0, y_max)
    except Exception:
        pass

    plt.tight_layout()
