import numpy as np
import matplotlib.pyplot as plt
import os

import constants
from config_classes import model_file_paths
from functions.average_data_func import average_data_func
from .shared_plot_data import shared_plot_data
from .helpers import matlab_datenum_to_datetime_array, format_time_axis, generate_matlab_style_filename


def plot_other_factors(shared: shared_plot_data, paths: model_file_paths) -> None:
    """
    Plot figure 5: Other factors (MATLAB figure 5)

    Panels:
      1) Effective emission factor (modelled vs observed)
      2) Dispersion factor (f_conc)
      3) Ratios: salt solution, PM10/PM200 (surface), PM2.5/PM10 (surface),
         PM2.5/PM10 (air, model), PM2.5/PM10 (air, observed)
      4) Bulk transfer coefficient (1/r_aero) with/without traffic
    """

    # Shorthands
    date_num = shared.date_num
    n_date = shared.date_num.shape[0]
    nodata = shared.nodata
    i_min, i_max = shared.i_min, shared.i_max
    av = list(shared.av)
    x_size = shared.plot_size_fraction

    # Convert x-axis to Python datetimes after averaging
    day_tick_limit = 150

    # --- Panel 1: Effective emission factor ---
    # Traffic: total vehicles per hour
    N_total_series = shared.traffic_data_ro[constants.N_total_index, :n_date].astype(
        float
    )

    # Model total emissions for selected PM fraction
    E_all_series = shared.E_road_data_sum_tracks[
        constants.total_dust_index, x_size, constants.E_total_index, :n_date
    ].astype(float)

    # Observed emissions derived from concentrations and dispersion factor
    f_conc = shared.f_conc[:n_date].astype(float).copy()
    PM_obs = shared.PM_obs_net.astype(float).copy()
    # Mask invalid times where f_conc or PM_obs(x_size) is nodata
    invalid = (f_conc == nodata) | (PM_obs[x_size, :n_date] == nodata)
    f_conc[invalid] = np.nan
    PM_obs[x_size, invalid] = np.nan
    E_obs_series = PM_obs[x_size, :n_date] / f_conc

    # Average required series
    x_str, xplot, y_E_all = average_data_func(date_num, E_all_series, i_min, i_max, av)
    _, _, y_E_obs = average_data_func(date_num, E_obs_series, i_min, i_max, av)
    _, _, y_N_total = average_data_func(date_num, N_total_series, i_min, i_max, av)

    # Emission factors (g/km/veh)
    with np.errstate(divide="ignore", invalid="ignore"):
        y_ef_mod = np.asarray(y_E_all) / np.asarray(y_N_total)
        y_ef_obs = np.asarray(y_E_obs) / np.asarray(y_N_total)

    dt_x = matlab_datenum_to_datetime_array(xplot)

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(8, 8))
    fig.subplots_adjust(hspace=0.35)
    try:
        fig.canvas.manager.set_window_title("Figure 5: Other factors")  # type: ignore
    except Exception:
        pass

    title_str = getattr(paths, "title_str", "") or "Other factors"
    ax1.set_title(f"{title_str}: Other factors")
    ax1.plot(
        dt_x,
        np.asarray(y_ef_mod).squeeze(),
        "b-",
        linewidth=1,
        label="Modelled emission factor",
    )
    ax1.plot(
        dt_x,
        np.asarray(y_ef_obs).squeeze(),
        "k--",
        linewidth=1,
        label="Observed emission factor",
    )
    ax1.set_ylabel("Emission factor (g/km/veh)", fontsize=6)
    ax1.legend(loc="upper left")
    format_time_axis(ax1, dt_x, shared.av[0], day_tick_limit=day_tick_limit)
    if len(dt_x) > 0:
        ax1.set_xlim(dt_x[0], dt_x[-1])

    # --- Panel 2: Dispersion factor ---
    f_conc2 = shared.f_conc[:n_date].astype(float).copy()
    f_conc2[f_conc2 == nodata] = np.nan
    _xstr2, xplot2, y_fconc = average_data_func(date_num, f_conc2, i_min, i_max, av)
    dt_x2 = matlab_datenum_to_datetime_array(xplot2)
    ax2.plot(
        dt_x2,
        np.asarray(y_fconc).squeeze(),
        "b-",
        linewidth=0.5,
        label="Concentration emission dispersion factor",
    )
    ax2.set_ylabel("Dispersion factor (µg/m³ per (g/km/hr))", fontsize=6)
    ax2.legend(loc="upper left")
    format_time_axis(ax2, dt_x2, shared.av[0], day_tick_limit=day_tick_limit)
    if len(dt_x2) > 0:
        ax2.set_xlim(dt_x2[0], dt_x2[-1])

    # --- Panel 3: Ratios ---
    # Apply MATLAB-equivalent masking to C_data using f_conc nodata times
    C_sum = shared.C_data_sum_tracks.astype(float).copy()
    C_sum[C_sum == nodata] = np.nan
    invalid_c = shared.f_conc[:n_date] == nodata
    if np.any(invalid_c):
        C_sum[:, :, :, invalid_c] = np.nan

    # Observed PM2.5/PM10 ratio in air with filtering
    PM_obs_all = shared.PM_obs_net.astype(float).copy()
    PM_obs_all[PM_obs_all == nodata] = np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        C_obs_ratio = PM_obs_all[constants.pm_25, :n_date] / (
            PM_obs_all[constants.pm_10, :n_date] + 0.001
        )
    r_obs = (
        np.isnan(PM_obs_all[constants.pm_10, :n_date])
        | np.isnan(PM_obs_all[constants.pm_25, :n_date])
        | (C_obs_ratio > 1.5)
        | (C_obs_ratio < -0.1)
        | (PM_obs_all[constants.pm_10, :n_date] < 2)
    )
    C_obs_ratio[r_obs] = np.nan

    # Dissolved salt ratio using road mass and water mass
    b_factor = shared.b_factor
    M_sum = shared.M_road_data_sum_tracks.astype(float).copy()
    M_sum[M_sum == nodata] = np.nan
    g_road = shared.g_road_weighted.astype(float).copy()
    g_road[g_road == nodata] = np.nan
    # Salt mass at pm_all, converted to g/m² via b_factor
    M2_salt_na = M_sum[constants.salt_index[0], constants.pm_all, :n_date] * b_factor
    M2_salt_2 = M_sum[constants.salt_index[1], constants.pm_all, :n_date] * b_factor
    water_gm2 = g_road[constants.water_index, :n_date] * 1000.0
    with np.errstate(divide="ignore", invalid="ignore"):
        salt_na_ratio = M2_salt_na / (M2_salt_na + water_gm2)
        salt_2_ratio = M2_salt_2 / (M2_salt_2 + water_gm2)

    # Surface mass ratios
    y_M_pm10_series = M_sum[constants.total_dust_index, constants.pm_10, :n_date]
    y_M_pm200_series = M_sum[constants.total_dust_index, constants.pm_200, :n_date]
    y_M_pm25_series = M_sum[constants.total_dust_index, constants.pm_25, :n_date]

    # Air PM2.5/PM10 (model): sum across all sources for each size
    with np.errstate(invalid="ignore"):
        air_pm25_series = np.nansum(
            C_sum[
                constants.all_source_index,
                constants.pm_25,
                constants.C_total_index,
                :n_date,
            ],
            axis=0,
        )
        air_pm10_series = np.nansum(
            C_sum[
                constants.all_source_index,
                constants.pm_10,
                constants.C_total_index,
                :n_date,
            ],
            axis=0,
        )
        air_ratio_series = air_pm25_series / air_pm10_series

    # Average all required time series
    _xs3, xp3, y_salt_na = average_data_func(date_num, salt_na_ratio, i_min, i_max, av)
    _, _, y_salt_2 = average_data_func(date_num, salt_2_ratio, i_min, i_max, av)
    _, _, y_M_pm10 = average_data_func(date_num, y_M_pm10_series, i_min, i_max, av)
    _, _, y_M_pm200 = average_data_func(date_num, y_M_pm200_series, i_min, i_max, av)
    _, _, y_M_pm25 = average_data_func(date_num, y_M_pm25_series, i_min, i_max, av)
    _, _, y_air_ratio = average_data_func(date_num, air_ratio_series, i_min, i_max, av)
    _, _, y_obs_ratio = average_data_func(date_num, C_obs_ratio, i_min, i_max, av)

    # Derived ratios after averaging to mirror MATLAB
    with np.errstate(divide="ignore", invalid="ignore"):
        y_PM10_PM200_surface = np.asarray(y_M_pm10) / np.asarray(y_M_pm200)
        y_PM25_PM10_surface = np.asarray(y_M_pm25) / np.asarray(y_M_pm10)

    dt_x3 = matlab_datenum_to_datetime_array(xp3)
    ax3.plot(
        dt_x3,
        np.asarray(y_salt_na).squeeze() * 100.0,
        "g-",
        linewidth=1,
        label="Salt(na) solution ratio",
    )
    ax3.plot(
        dt_x3,
        np.asarray(y_salt_2).squeeze() * 100.0,
        "g--",
        linewidth=1,
        label=f"Salt({shared.salt2_str}) solution ratio",
    )
    ax3.plot(
        dt_x3,
        np.asarray(y_PM10_PM200_surface).squeeze() * 100.0,
        "r-",
        linewidth=1,
        label="PM10/PM200 ratio surface",
    )
    ax3.plot(
        dt_x3,
        np.asarray(y_PM25_PM10_surface).squeeze() * 100.0,
        "r--",
        linewidth=1,
        label="PM2.5/PM10 ratio surface",
    )
    ax3.plot(
        dt_x3,
        np.asarray(y_air_ratio).squeeze() * 100.0,
        "b-",
        linewidth=1,
        label="PM2.5/PM10 mod ratio air",
    )
    ax3.plot(
        dt_x3,
        np.asarray(y_obs_ratio).squeeze() * 100.0,
        "k--",
        linewidth=1,
        label="PM2.5/PM10 obs ratio air",
    )
    ax3.set_ylabel("Ratio (%)", fontsize=6)
    ax3.legend(loc="upper left")
    format_time_axis(ax3, dt_x3, shared.av[0], day_tick_limit=day_tick_limit)
    if len(dt_x3) > 0:
        ax3.set_xlim(dt_x3[0], dt_x3[-1])

    # --- Panel 4: Bulk transfer coefficient ---
    roadm = shared.road_meteo_weighted.astype(float).copy()
    meteo = shared.meteo_data_ro.astype(float).copy()
    roadm[roadm == nodata] = np.nan
    meteo[meteo == nodata] = np.nan
    FF = meteo[constants.FF_index, :n_date]
    # Avoid unrealistically low wind speeds as in MATLAB: max(0.2, FF)
    FF_capped = np.maximum(0.2, FF)
    with np.errstate(divide="ignore", invalid="ignore"):
        bulk_with = 1.0 / roadm[constants.r_aero_index, :n_date] / FF_capped
        bulk_without = (
            1.0 / roadm[constants.r_aero_notraffic_index, :n_date] / FF_capped
        )

    _xs4, xp4, y_bulk_with = average_data_func(date_num, bulk_with, i_min, i_max, av)
    _, _, y_bulk_without = average_data_func(date_num, bulk_without, i_min, i_max, av)
    dt_x4 = matlab_datenum_to_datetime_array(xp4)
    ax4.plot(
        dt_x4,
        np.asarray(y_bulk_with).squeeze(),
        "b-",
        linewidth=0.5,
        label="With traffic",
    )
    ax4.plot(
        dt_x4,
        np.asarray(y_bulk_without).squeeze(),
        "r-",
        linewidth=0.5,
        label="Without traffic",
    )
    ax4.set_ylabel("Bulk transfer coefficient (m/s)", fontsize=6)
    ax4.legend(loc="upper left")
    format_time_axis(ax4, dt_x4, shared.av[0], day_tick_limit=day_tick_limit)
    if len(dt_x4) > 0:
        ax4.set_xlim(dt_x4[0], dt_x4[-1])

    plt.tight_layout()
    if shared.save_plots:
        plot_file_name = generate_matlab_style_filename(
            title_str=getattr(paths, "title_str", ""),
            plot_type_flag=shared.av[0],
            figure_number=5,  # Other factors is figure 5
            plot_name="Other_factors",
            date_num=shared.date_num,
            min_time=shared.i_min,
            max_time=shared.i_max
        )
        plt.savefig(os.path.join(paths.path_outputfig, plot_file_name))
