from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import os

import constants
from config_classes import model_file_paths
from functions import average_data_func
from .shared_plot_data import shared_plot_data
from .helpers import matlab_datenum_to_datetime_array, format_time_axis, mask_nodata, generate_matlab_style_filename


def plot_wetness(shared: shared_plot_data, paths: model_file_paths) -> None:
    """
    Plot figure 4: Road wetness (3 panels)

    1) Surface wetness (mm): modelled water depth, optionally observed water depth
    2) Surface snow and ice (mm w.e.): snow depth, ice depth, optional ploughing stairs
    3) Retention factor f_q: road and brake, optionally observed
    """

    # Shorthands
    date_num = shared.date_num
    n_date = shared.date_num.shape[0]
    nodata = shared.nodata
    i_min, i_max = shared.i_min, shared.i_max
    av = list(shared.av)

    g_road = mask_nodata(shared.g_road_weighted.copy(), nodata)
    activity = mask_nodata(shared.activity_data_ro.copy(), nodata)
    meteo = mask_nodata(shared.meteo_data_ro.copy(), nodata)
    f_q_all = mask_nodata(shared.f_q_weighted.copy(), nodata)
    f_q_obs = mask_nodata(shared.f_q_obs_weighted.copy(), nodata)

    # ------------- Panel 1: Surface wetness -------------
    x_str, xplot, water_mm = average_data_func(
        date_num, g_road[constants.water_index, :n_date], i_min, i_max, av
    )

    # Observed water depth availability from shared flags (require mm units like MATLAB)
    obs_available = bool(
        shared.road_wetness_obs_available and shared.road_wetness_obs_in_mm
    )
    obs_wet_series = meteo[constants.road_wetness_obs_input_index, :n_date]
    if obs_available and np.any(~np.isnan(obs_wet_series)):
        _, _, obs_wet = average_data_func(date_num, obs_wet_series, i_min, i_max, av)
    else:
        obs_wet = None

    dt_x = matlab_datenum_to_datetime_array(xplot)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 6), sharex=False)
    fig.subplots_adjust(hspace=0.35)
    try:
        fig.canvas.manager.set_window_title("Figure 4: Road wetness")  # type: ignore
    except Exception:
        pass

    ax1.set_title(f"{paths.title_str}: Road surface condition")

    ax1.plot(dt_x, water_mm.squeeze(), "b-", linewidth=1, label="Modelled water depth")
    y_max = np.nanmax(water_mm) * 1.1 if np.isfinite(np.nanmax(water_mm)) else np.nan
    if obs_available and obs_wet is not None:
        ax1.plot(
            dt_x, obs_wet.squeeze(), "k--", linewidth=1, label="Observed water depth"
        )
        y_max = np.nanmax([y_max, np.nanmax(obs_wet) * 1.1])
    ax1.set_ylabel("Surface wetness (mm)")
    ax1.legend(loc="upper left")
    format_time_axis(ax1, dt_x, shared.av[0], day_tick_limit=150)
    if np.isfinite(y_max):
        ax1.set_ylim(0, y_max)

    # ------------- Panel 2: Snow and ice depths -------------
    _, _, snow_mm = average_data_func(
        date_num, g_road[constants.snow_index, :n_date], i_min, i_max, av
    )
    _, _, ice_mm = average_data_func(
        date_num, g_road[constants.ice_index, :n_date], i_min, i_max, av
    )
    max_plot = float(np.nanmax(np.vstack([snow_mm.squeeze(), ice_mm.squeeze()])))
    _, _, plough = average_data_func(
        date_num, activity[constants.t_ploughing_index, :n_date], i_min, i_max, av
    )
    has_plough = np.any(plough != 0)

    legend_entries = ["Road snow depth", "Road ice depth"]
    if has_plough and max_plot > 0:
        plough_norm = (
            plough.squeeze() / np.nanmax(plough) * max_plot
            if np.nanmax(plough) > 0
            else plough.squeeze()
        )
        ax2.step(
            dt_x, plough_norm, where="post", color="g", linewidth=0.5, label="Ploughing"
        )
        legend_entries = ["Ploughing"] + legend_entries

    ax2.plot(dt_x, snow_mm.squeeze(), "b-", linewidth=1, label="Road snow depth")
    ax2.plot(dt_x, ice_mm.squeeze(), "b--", linewidth=1, label="Road ice depth")
    ax2.set_ylabel("Surface snow and ice (mm w.e.)")
    ax2.legend(legend_entries, loc="upper left")
    format_time_axis(ax2, dt_x, shared.av[0], day_tick_limit=150)

    # ------------- Panel 3: Retention factor f_q -------------
    _, _, fq_road = average_data_func(
        date_num, f_q_all[constants.road_index, :n_date], i_min, i_max, av
    )
    _, _, fq_brake = average_data_func(
        date_num, f_q_all[constants.brake_index, :n_date], i_min, i_max, av
    )
    obs_fq_avail = bool(np.any(~np.isnan(f_q_obs[:n_date])))
    if obs_fq_avail:
        _, _, fq_obs = average_data_func(date_num, f_q_obs[:n_date], i_min, i_max, av)
    else:
        fq_obs = None

    legend_fq = ["Road", "Brake"] + (["Observed"] if obs_fq_avail else [])
    ax3.plot(dt_x, fq_road.squeeze(), "b-", linewidth=1, label="Road")
    ax3.plot(dt_x, fq_brake.squeeze(), "m--", linewidth=0.5, label="Brake")
    if obs_fq_avail and fq_obs is not None:
        ax3.plot(dt_x, fq_obs.squeeze(), "k--", linewidth=1, label="Observed")
    ax3.set_ylabel("Retention factor f_q")
    ax3.set_ylim(-0.05, 1.05)
    ax3.legend(legend_fq, loc="upper left")
    format_time_axis(ax3, dt_x, shared.av[0], day_tick_limit=150)

    plt.tight_layout()
    if shared.save_plots:
        plot_file_name = generate_matlab_style_filename(
            title_str=paths.title_str,
            plot_type_flag=shared.av[0],
            figure_number=4,  # Road wetness is figure 4
            plot_name="Road_wetness",
            date_num=shared.date_num,
            min_time=shared.i_min,
            max_time=shared.i_max
        )
        plt.savefig(os.path.join(paths.path_outputfig, plot_file_name))
