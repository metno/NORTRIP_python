from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

import constants
from config_classes import model_file_paths
from functions import average_data_func
from .shared_plot_data import shared_plot_data
from .helpers import matlab_datenum_to_datetime_array, format_time_axis, mask_nodata


def plot_meteorology(shared: shared_plot_data, paths: model_file_paths) -> None:
    """
    Plot meteorology figure (Figure 2 in MATLAB):
    - Temperature (air <0 and >=0 in different colors), road temperature, observed road temp (if available),
      freezing temperature (if available), and sub-surface temperature
    - Relative humidity (air, road, salt)
    - Cloud cover (%)
    - Wind speed (m/s)
    - Precipitation (rain and snow as step plots)
    """

    # Shorthands and local masked copies
    date_num = shared.date_num
    n_date = shared.date_num.shape[0]
    nodata = shared.nodata
    i_min, i_max = shared.i_min, shared.i_max
    av = list(shared.av)

    meteo = mask_nodata(shared.meteo_data_ro.copy(), nodata)
    roadm = mask_nodata(shared.road_meteo_weighted.copy(), nodata)

    # ---------- Panel 1: Temperature ----------
    x_str, xplot, Ta = average_data_func(
        date_num, meteo[constants.T_a_index, :n_date], i_min, i_max, av
    )
    _, _, Ts = average_data_func(
        date_num, roadm[constants.T_s_index, :n_date], i_min, i_max, av
    )

    # Optional observed road temperature
    road_obs_arr = roadm[constants.road_temperature_obs_index, :n_date]
    has_road_obs = np.any(~np.isnan(road_obs_arr))
    if has_road_obs:
        _, _, Tobs = average_data_func(date_num, road_obs_arr, i_min, i_max, av)
    else:
        Tobs = None

    # Optional freezing temperature (salt humidity flag equivalent)
    Tmelt_arr = roadm[constants.T_melt_index, :n_date]
    has_Tmelt = np.any(~np.isnan(Tmelt_arr))
    if has_Tmelt:
        _, _, Tmelt = average_data_func(date_num, Tmelt_arr, i_min, i_max, av)
    else:
        Tmelt = None

    # Sub-surface temperature
    _, _, Tsub = average_data_func(
        date_num, roadm[constants.T_sub_index, :n_date], i_min, i_max, av
    )

    dt_x = matlab_datenum_to_datetime_array(xplot)

    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(10, 6), sharex=False)
    fig.subplots_adjust(hspace=0.35)
    try:
        fig.canvas.manager.set_window_title("Figure 2: Meteorology")  # type: ignore
    except Exception:
        pass

    # Title on first panel

    ax1.set_title(f"{paths.title_str}: Meteorology")

    # Split air temperature into <0C and >=0C segments
    Ta_flat = Ta.squeeze()
    Ta_neg = np.where(Ta_flat < 0, Ta_flat, np.nan)
    Ta_pos = np.where(Ta_flat >= 0, Ta_flat, np.nan)
    ax1.plot(dt_x, Ta_neg, "b-", linewidth=1, label="Temperature (C) < 0")
    ax1.plot(dt_x, Ta_pos, "r-", linewidth=1, label="Temperature (C) > 0")
    ax1.plot(dt_x, Ts.squeeze(), "m--", linewidth=1, label="Road temperature (C)")
    if has_road_obs and Tobs is not None:
        ax1.plot(dt_x, Tobs.squeeze(), "k--", linewidth=1, label="Observed road")
    if has_Tmelt and Tmelt is not None:
        ax1.plot(
            dt_x, Tmelt.squeeze(), "g--", linewidth=1, label="Freezing temperature"
        )
    ax1.plot(dt_x, Tsub.squeeze(), "r:", linewidth=0.5, label="Sub-surface temperature")
    ax1.set_ylabel("Temperature (C)", fontsize=6)
    ax1.legend(loc="upper left")
    format_time_axis(ax1, dt_x, shared.av[0], day_tick_limit=150)

    # ---------- Panel 2: Relative humidity ----------
    _, _, RH_air = average_data_func(
        date_num, meteo[constants.RH_index, :n_date], i_min, i_max, av
    )
    _, _, RH_road = average_data_func(
        date_num, roadm[constants.RH_s_index, :n_date], i_min, i_max, av
    )
    _, _, RH_salt = average_data_func(
        date_num, roadm[constants.RH_salt_final_index, :n_date], i_min, i_max, av
    )
    ax2.plot(dt_x, RH_air.squeeze(), "b-", linewidth=1, label="RH air")
    ax2.plot(dt_x, RH_road.squeeze(), "m-", linewidth=0.8, label="RH road")
    ax2.plot(dt_x, RH_salt.squeeze(), "r:", linewidth=0.8, label="RH salt")
    ax2.set_ylabel("Relative humidity (%)", fontsize=6)
    ax2.legend(loc="upper left")
    format_time_axis(ax2, dt_x, shared.av[0], day_tick_limit=150)

    # ---------- Panel 3: Cloud cover ----------
    _, _, cloud = average_data_func(
        date_num, meteo[constants.cloud_cover_index, :n_date] * 100.0, i_min, i_max, av
    )
    ax3.plot(dt_x, cloud.squeeze(), "k:", linewidth=0.5, label="Cloud cover")
    ax3.set_ylabel("Cloud cover (%)", fontsize=6)
    ax3.legend(loc="upper left")
    format_time_axis(ax3, dt_x, shared.av[0], day_tick_limit=150)

    # ---------- Panel 4: Wind speed ----------
    _, _, FF = average_data_func(
        date_num, meteo[constants.FF_index, :n_date], i_min, i_max, av
    )
    ax4.plot(dt_x, FF.squeeze(), "b-", linewidth=1, label="Wind speed")
    ax4.set_ylabel("Wind speed (m/s)", fontsize=6)
    ax4.legend(loc="upper left")
    format_time_axis(ax4, dt_x, shared.av[0], day_tick_limit=150)

    # ---------- Panel 5: Precipitation ----------
    _, _, Rain = average_data_func(
        date_num, meteo[constants.Rain_precip_index, :n_date], i_min, i_max, av
    )
    _, _, Snow = average_data_func(
        date_num, meteo[constants.Snow_precip_index, :n_date], i_min, i_max, av
    )
    ax5.step(dt_x, Rain.squeeze(), where="post", color="b", linewidth=1, label="Rain")
    ax5.step(dt_x, Snow.squeeze(), where="post", color="m", linewidth=1, label="Snow")
    ax5.set_ylabel("Precipitation (mm/hr)", fontsize=6)
    ax5.set_xlabel("Date")
    ax5.legend(loc="upper left")
    format_time_axis(ax5, dt_x, shared.av[0], day_tick_limit=150)

    plt.tight_layout()
