from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

import constants
from config_classes import model_file_paths
from functions import average_data_func
from .shared_plot_data import shared_plot_data
from .helpers import matlab_datenum_to_datetime_array, format_time_axis, mask_nodata


def _safe_divide(num: np.ndarray | float, den: np.ndarray | float) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.true_divide(num, den)
        out[~np.isfinite(out)] = np.nan
    return out


def plot_other_factors(shared: shared_plot_data, paths: model_file_paths) -> None:
    """
    Plot figure 5: Other factors (4 panels)

    1) Effective emission factor (g/km/veh): model vs observed
    2) Dispersion factor (f_conc)
    3) Ratios (%): salt solution ratios (na/mg), PM10/PM200 surface, PM2.5/PM10 surface,
       PM2.5/PM10 modeled air, PM2.5/PM10 observed air
    4) Bulk transfer coefficient (1/r_aero) with and without traffic
    """

    # Shorthands
    date_num = shared.date_num
    n_date = shared.date_num.shape[0]
    nodata = shared.nodata
    i_min, i_max = shared.i_min, shared.i_max
    av = list(shared.av)
    x_size = shared.plot_size_fraction
    b_factor = shared.b_factor

    # Local copies
    traffic = mask_nodata(shared.traffic_data_ro.copy(), nodata)
    f_conc = shared.f_conc.astype(float).copy()
    f_conc[f_conc == nodata] = np.nan
    PM_obs = shared.PM_obs_net.astype(float).copy()
    PM_obs[PM_obs == nodata] = np.nan

    E_sum = shared.E_road_data_sum_tracks
    M_sum = shared.M_road_data_sum_tracks
    C_sum = shared.C_data_sum_tracks
    g_road = shared.g_road_weighted
    meteo = shared.meteo_data_ro
    roadm = shared.road_meteo_weighted

    # ---- Panel 1: Effective emission factor ----
    # Model: EF_mod = E_total / N_total; Observed: EF_obs = PM_obs / f_conc, then / N_total
    E_total = np.maximum(
        E_sum[constants.total_dust_index, x_size, constants.E_total_index, :n_date], 0
    )
    N_total = np.maximum(traffic[constants.N_total_index, :n_date], 0)
    PM_obs_size = np.maximum(PM_obs[x_size, :n_date], 0)
    E_obs_series = np.full(n_date, np.nan)
    valid = ~np.isnan(f_conc[:n_date]) & ~np.isnan(PM_obs_size)
    E_obs_series[valid] = PM_obs_size[valid] / f_conc[:n_date][valid]

    # Average inputs then form ratios (to mirror MATLAB sequence)
    x_str, xplot, y_E_total = average_data_func(date_num, E_total, i_min, i_max, av)
    _, _, y_N_total = average_data_func(date_num, N_total, i_min, i_max, av)
    _, _, y_E_obs = average_data_func(date_num, E_obs_series, i_min, i_max, av)
    ef_mod = _safe_divide(y_E_total.squeeze(), y_N_total.squeeze())
    ef_obs = _safe_divide(y_E_obs.squeeze(), y_N_total.squeeze())
    dt_x = matlab_datenum_to_datetime_array(xplot)

    fig, ((ax1), (ax2), (ax3), (ax4)) = plt.subplots(
        4, 1, figsize=(10, 7), sharex=False
    )
    fig.subplots_adjust(hspace=0.5)
    try:
        fig.canvas.manager.set_window_title("Figure 5: Other factors")  # type: ignore
    except Exception:
        pass

    ax1.set_title(f"{paths.title_str}: Other factors")
    ax1.plot(dt_x, ef_mod, "b-", linewidth=1, label="Modelled emission factor")
    ax1.plot(dt_x, ef_obs, "k--", linewidth=1, label="Observed emission factor")
    ax1.set_ylabel("Emission factor (g/km/veh)")
    ax1.legend(loc="upper left")
    format_time_axis(ax1, dt_x, shared.av[0], day_tick_limit=150)

    # ---- Panel 2: Dispersion factor f_conc ----
    _, _, y_f_conc = average_data_func(date_num, f_conc[:n_date], i_min, i_max, av)
    ax2.plot(
        dt_x,
        y_f_conc.squeeze(),
        "b-",
        linewidth=0.5,
        label="Concentration emission dispersion factor",
    )
    ax2.set_ylabel("Dispersion factor ((µg/m³)/(g/km/hr))")
    ax2.legend(loc="upper left")
    format_time_axis(ax2, dt_x, shared.av[0], day_tick_limit=150)

    # ---- Panel 3: Ratios (%) ----
    # Salt solution ratios based on surface mass and water depth
    M_salt_na = np.maximum(
        M_sum[constants.salt_index[0], constants.pm_all, :n_date] * b_factor, 0
    )
    M_salt_2 = np.maximum(
        M_sum[constants.salt_index[1], constants.pm_all, :n_date] * b_factor, 0
    )
    water_mm = np.maximum(g_road[constants.water_index, :n_date], 0)
    ratio_na = _safe_divide(M_salt_na, M_salt_na + water_mm * 1000.0)
    ratio_2 = _safe_divide(M_salt_2, M_salt_2 + water_mm * 1000.0)
    _, _, y_ratio_na = average_data_func(date_num, ratio_na, i_min, i_max, av)
    _, _, y_ratio_2 = average_data_func(date_num, ratio_2, i_min, i_max, av)

    # PM surface ratios from surface mass loadings (not scaled by b_factor since it cancels)
    pm10_surf = np.maximum(
        M_sum[constants.total_dust_index, constants.pm_10, :n_date], 0
    )
    pm200_surf = np.maximum(
        M_sum[constants.total_dust_index, constants.pm_200, :n_date], 0
    )
    pm25_surf = np.maximum(
        M_sum[constants.total_dust_index, constants.pm_25, :n_date], 0
    )
    _, _, y_pm10_s = average_data_func(date_num, pm10_surf, i_min, i_max, av)
    _, _, y_pm200_s = average_data_func(date_num, pm200_surf, i_min, i_max, av)
    _, _, y_pm25_s = average_data_func(date_num, pm25_surf, i_min, i_max, av)
    y_ratio_pm10_pm200 = _safe_divide(y_pm10_s.squeeze(), y_pm200_s.squeeze())
    y_ratio_pm25_pm10_surf = _safe_divide(y_pm25_s.squeeze(), y_pm10_s.squeeze())

    # PM modeled air ratio from concentrations by size (sum over sources)
    conc_pm25 = np.nansum(
        C_sum[
            constants.all_source_index,
            constants.pm_25,
            constants.C_total_index,
            :n_date,
        ],
        axis=0,
    )
    conc_pm10 = np.nansum(
        C_sum[
            constants.all_source_index,
            constants.pm_10,
            constants.C_total_index,
            :n_date,
        ],
        axis=0,
    )
    _, _, y_conc_pm25 = average_data_func(date_num, conc_pm25, i_min, i_max, av)
    _, _, y_conc_pm10 = average_data_func(date_num, conc_pm10, i_min, i_max, av)
    y_ratio_pm25_pm10_mod = _safe_divide(y_conc_pm25.squeeze(), y_conc_pm10.squeeze())

    # PM observed air ratio with filtering
    obs_ratio_series = np.full(n_date, np.nan)
    pm10_obs = PM_obs[constants.pm_10, :n_date]
    pm25_obs = PM_obs[constants.pm_25, :n_date]
    raw_ratio = _safe_divide(pm25_obs, pm10_obs + 0.001)
    valid_obs = (
        ~np.isnan(pm10_obs)
        & ~np.isnan(pm25_obs)
        & (raw_ratio <= 1.5)
        & (raw_ratio >= -0.1)
        & (pm10_obs >= 2)
    )
    obs_ratio_series[valid_obs] = raw_ratio[valid_obs]
    _, _, y_obs_ratio = average_data_func(date_num, obs_ratio_series, i_min, i_max, av)

    # Plot all ratios as percentages
    ax3.plot(
        dt_x,
        (y_ratio_na.squeeze() * 100.0),
        "g-",
        linewidth=1,
        label="Salt(na) solution ratio",
    )
    ax3.plot(
        dt_x,
        (y_ratio_2.squeeze() * 100.0),
        "g--",
        linewidth=1,
        label="Salt(mg) solution ratio",
    )
    ax3.plot(
        dt_x,
        (y_ratio_pm10_pm200 * 100.0),
        "r-",
        linewidth=1,
        label="PM10/PM200 ratio surface",
    )
    ax3.plot(
        dt_x,
        (y_ratio_pm25_pm10_surf * 100.0),
        "r--",
        linewidth=1,
        label="PM2.5/PM10 ratio surface",
    )
    ax3.plot(
        dt_x,
        (y_ratio_pm25_pm10_mod * 100.0),
        "b-",
        linewidth=1,
        label="PM2.5/PM10 mod ratio air",
    )
    ax3.plot(
        dt_x,
        (y_obs_ratio.squeeze() * 100.0),
        "k--",
        linewidth=1,
        label="PM2.5/PM10 obs ratio air",
    )
    ax3.set_ylabel("Ratio (%)")
    ax3.legend(loc="upper left")
    format_time_axis(ax3, dt_x, shared.av[0], day_tick_limit=150)

    # ---- Panel 4: Bulk transfer coefficient ----
    FF = np.maximum(meteo[constants.FF_index, :n_date], 0)
    denom = np.maximum(FF, 0.2)
    with_traffic = _safe_divide(1.0, roadm[constants.r_aero_index, :n_date] * denom)
    no_traffic = _safe_divide(
        1.0, roadm[constants.r_aero_notraffic_index, :n_date] * denom
    )
    _, _, y_with = average_data_func(date_num, with_traffic, i_min, i_max, av)
    _, _, y_without = average_data_func(date_num, no_traffic, i_min, i_max, av)
    ax4.plot(dt_x, y_with.squeeze(), "b-", linewidth=0.5, label="With traffic")
    ax4.plot(dt_x, y_without.squeeze(), "r-", linewidth=0.5, label="Without traffic")
    ax4.set_ylabel("Bulk transfer coefficient (m/s)")
    ax4.legend(loc="upper left")
    format_time_axis(ax4, dt_x, shared.av[0], day_tick_limit=150)

    plt.tight_layout()
