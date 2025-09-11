import numpy as np
import matplotlib.pyplot as plt
import os

import constants
from config_classes import model_file_paths
from functions.average_data_func import average_data_func
from .shared_plot_data import shared_plot_data
from .helpers import matlab_datenum_to_datetime_array, format_time_axis, generate_matlab_style_filename


def plot_concentrations(shared: shared_plot_data, paths: model_file_paths) -> None:
    """
    Plot figure 7: Concentrations (PM10, PM2.5, NOX), mirroring MATLAB code.
    """

    # Shorthands
    date_num = shared.date_num
    n_date = shared.date_num.shape[0]
    nodata = shared.nodata
    i_min, i_max = shared.i_min, shared.i_max
    av = list(shared.av)
    day_tick_limit = 150

    # Local copies and masking helpers
    C_sum = shared.C_data_sum_tracks.astype(float).copy()
    C_sum[C_sum == nodata] = np.nan

    PM_obs_net = shared.PM_obs_net.astype(float).copy()
    PM_obs_net[PM_obs_net == nodata] = np.nan

    Salt_obs = shared.Salt_obs.astype(float).copy()
    Salt_obs[Salt_obs == nodata] = np.nan
    Salt_obs_available = shared.Salt_obs_available

    # --- Panel 1: PM10 net concentrations and components ---
    # Mask times where either dispersion factor or PM10 obs is nodata
    f_conc = shared.f_conc[:n_date].astype(float)
    invalid = (f_conc == nodata) | np.isnan(PM_obs_net[constants.pm_10, :n_date])
    C_sum[:, :, :, invalid] = np.nan
    pm10_obs_series = PM_obs_net[constants.pm_10, :n_date]

    # Build modelled components
    y_total_pm10_series = np.sum(
        C_sum[
            constants.all_source_index,
            constants.pm_10,
            constants.C_total_index,
            :n_date,
        ],
        axis=0,
    )
    y_salt_na_pm10_series = C_sum[
        constants.salt_index[0], constants.pm_10, constants.C_total_index, :n_date
    ]
    y_salt_2_pm10_series = C_sum[
        constants.salt_index[1], constants.pm_10, constants.C_total_index, :n_date
    ]
    y_wear_pm10_series = np.sum(
        C_sum[constants.wear_index, constants.pm_10, constants.C_total_index, :n_date],
        axis=0,
    )
    y_sand_pm10_series = C_sum[
        constants.sand_index, constants.pm_10, constants.C_total_index, :n_date
    ]

    # Average to requested resolution
    x_str, xplot, y_total_pm10 = average_data_func(
        date_num, y_total_pm10_series, i_min, i_max, av
    )
    _, _, y_obs_pm10 = average_data_func(date_num, pm10_obs_series, i_min, i_max, av)
    _, _, y_salt_na_pm10 = average_data_func(
        date_num, y_salt_na_pm10_series, i_min, i_max, av
    )
    _, _, y_wear_pm10 = average_data_func(
        date_num, y_wear_pm10_series, i_min, i_max, av
    )
    _, _, y_sand_pm10 = average_data_func(
        date_num, y_sand_pm10_series, i_min, i_max, av
    )
    _, _, y_salt_2_pm10 = average_data_func(
        date_num, y_salt_2_pm10_series, i_min, i_max, av
    )

    dt_x1 = matlab_datenum_to_datetime_array(xplot)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(9, 8))
    fig.subplots_adjust(hspace=0.35)
    try:
        fig.canvas.manager.set_window_title("Figure 7: Concentrations")  # type: ignore
    except Exception:
        pass

    title_str = getattr(paths, "title_str", "") or "Concentrations"
    ax1.set_title(f"{title_str}: Net concentrations")
    ax1.plot(
        dt_x1, np.asarray(y_obs_pm10).squeeze(), "k--", linewidth=1, label="Observed"
    )
    ax1.plot(
        dt_x1,
        np.asarray(y_salt_na_pm10).squeeze(),
        "g-",
        linewidth=1,
        label="Modelled salt(na)",
    )
    ax1.plot(
        dt_x1,
        np.asarray(y_salt_2_pm10).squeeze(),
        "g--",
        linewidth=1,
        label=f"Modelled salt({shared.salt2_str})",
    )
    ax1.plot(
        dt_x1,
        np.asarray(y_wear_pm10).squeeze(),
        "r:",
        linewidth=1,
        label="Modelled wear",
    )
    ax1.plot(
        dt_x1,
        np.asarray(y_sand_pm10).squeeze(),
        "m--",
        linewidth=1,
        label="Modelled sand",
    )
    ax1.plot(
        dt_x1,
        np.asarray(y_total_pm10).squeeze(),
        "b-",
        linewidth=1,
        label="Modelled total",
    )

    if int(Salt_obs_available[constants.na]) == 1:
        _, _, y_salt_obs = average_data_func(
            date_num, Salt_obs[constants.na, :n_date], i_min, i_max, av
        )
        ax1.plot(
            dt_x1,
            np.asarray(y_salt_obs).squeeze(),
            "g-",
            linewidth=1,
            label="Observed salt",
        )

    ax1.set_ylabel("PM10 concentration (µg/m³)")
    format_time_axis(ax1, dt_x1, shared.av[0], day_tick_limit=day_tick_limit)
    ax1.legend(loc="upper left")
    if len(dt_x1) > 0:
        ax1.set_xlim(dt_x1[0], dt_x1[-1])
    # y-limit similar to MATLAB: 0..1.1*max of plotted series
    try:
        y_stack = np.vstack(
            [
                np.asarray(y_total_pm10).squeeze(),
                np.asarray(y_obs_pm10).squeeze(),
                np.asarray(y_salt_na_pm10).squeeze(),
                np.asarray(y_wear_pm10).squeeze(),
                np.asarray(y_sand_pm10).squeeze(),
            ]
        )
        y_max = float(np.nanmax(y_stack)) * 1.1
        if np.isfinite(y_max):
            ax1.set_ylim(0, y_max)
    except Exception:
        pass

    # --- Panel 2: PM2.5 ---
    # Mask times where f_conc or PM2.5 obs is nodata
    invalid25 = (f_conc == nodata) | np.isnan(PM_obs_net[constants.pm_25, :n_date])
    C_sum[:, :, :, invalid25] = np.nan
    pm25_obs_series = PM_obs_net[constants.pm_25, :n_date]

    y_total_pm25_series = np.sum(
        C_sum[
            constants.all_source_index,
            constants.pm_25,
            constants.C_total_index,
            :n_date,
        ],
        axis=0,
    )
    y_exhaust_pm25_series = C_sum[
        constants.exhaust_index, constants.pm_25, constants.C_total_index, :n_date
    ]

    x_str2, xplot2, y_total_pm25 = average_data_func(
        date_num, y_total_pm25_series, i_min, i_max, av
    )
    _, _, y_obs_pm25 = average_data_func(date_num, pm25_obs_series, i_min, i_max, av)
    _, _, y_exhaust_pm25 = average_data_func(
        date_num, y_exhaust_pm25_series, i_min, i_max, av
    )

    dt_x2 = matlab_datenum_to_datetime_array(xplot2)
    ax2.set_title("")
    ax2.plot(
        dt_x2, np.asarray(y_obs_pm25).squeeze(), "k--", linewidth=1, label="Observed"
    )
    ax2.plot(
        dt_x2, np.asarray(y_exhaust_pm25).squeeze(), "m:", linewidth=1, label="Exhaust"
    )
    ax2.plot(
        dt_x2,
        np.asarray(y_total_pm25).squeeze(),
        "b-",
        linewidth=1,
        label="Modelled total",
    )
    ax2.set_ylabel("PM2.5 concentration (µg/m³)")
    format_time_axis(ax2, dt_x2, shared.av[0], day_tick_limit=day_tick_limit)
    ax2.legend(loc="upper left")
    if len(dt_x2) > 0:
        ax2.set_xlim(dt_x2[0], dt_x2[-1])

    # --- Panel 3: NOX ---
    NOX_obs = shared.NOX_obs.astype(float).copy()
    NOX_background = shared.NOX_background.astype(float).copy()
    NOX_obs_net = shared.NOX_obs_net.astype(float).copy()

    NOX_obs[NOX_obs == nodata] = np.nan
    NOX_background[NOX_background == nodata] = np.nan
    NOX_obs_net[NOX_obs_net == nodata] = np.nan

    x_str3, xplot3, y_NOX_obs = average_data_func(
        date_num, NOX_obs[:n_date], i_min, i_max, av
    )
    _, _, y_NOX_bg = average_data_func(
        date_num, NOX_background[:n_date], i_min, i_max, av
    )
    _, _, y_NOX_net = average_data_func(
        date_num, NOX_obs_net[:n_date], i_min, i_max, av
    )

    dt_x3 = matlab_datenum_to_datetime_array(xplot3)
    ax3.set_title("")
    ax3.plot(
        dt_x3, np.asarray(y_NOX_obs).squeeze(), "r--", linewidth=0.5, label="Observed"
    )
    ax3.plot(
        dt_x3, np.asarray(y_NOX_bg).squeeze(), "b--", linewidth=0.5, label="Background"
    )
    ax3.plot(dt_x3, np.asarray(y_NOX_net).squeeze(), "k-", linewidth=1, label="Net")
    ax3.set_ylabel("NOX concentration (µg/m³)")
    format_time_axis(ax3, dt_x3, shared.av[0], day_tick_limit=day_tick_limit)
    ax3.legend(loc="upper left")
    if len(dt_x3) > 0:
        ax3.set_xlim(dt_x3[0], dt_x3[-1])

    plt.tight_layout()
    if shared.save_plots:
        plot_file_name = generate_matlab_style_filename(
            title_str=paths.title_str,
            plot_type_flag=shared.av[0],
            figure_number=7,  # Concentrations is figure 7
            plot_name="Concentrations",
            date_num=shared.date_num,
            min_time=shared.i_min,
            max_time=shared.i_max
        )
        plt.savefig(os.path.join(paths.path_outputfig, plot_file_name))
