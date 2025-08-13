from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import constants
from config_classes import model_parameters, model_flags, model_file_paths
from input_classes import converted_data, input_metadata, input_airquality
from initialise import time_config, model_variables
from .style import shrink_y_labels
from .helpers import (
    matlab_datenum_to_datetime_array as _matlab_datenum_to_datetime_array,
    prepare_series as _prepare_series,
    format_time_axis as _format_time_axis,
)

# Helpers moved to plots.helpers for reuse and clarity


def plot_summary(
    *,
    time_config: time_config,
    converted_data: converted_data,
    metadata: input_metadata,
    airquality_data: input_airquality,
    model_parameters: model_parameters,
    model_flags: model_flags,
    model_variables: model_variables,
    paths: Optional[model_file_paths] = None,
    ro: int = 0,
    plot_size_fraction: int = constants.pm_10,
):
    """
    Summary plot roughly matching MATLAB Figure 13 structure (first two panels):
    - Subplot 1: Concentrations (observed vs model total and source splits)
    - Subplot 2: Mass loading (dust, salt types, sand) with cleaning overlay
    """
    # Aliases
    x = plot_size_fraction
    # nodata = metadata.nodata
    av = (model_flags.plot_type_flag,)  # averaging spec

    # Build track-weighted temporary arrays for the selected road
    C_data_temp = np.sum(
        model_variables.C_data[:, :, :, :, : model_parameters.num_track, ro], axis=4
    )

    M_road_data_temp = np.sum(
        model_variables.M_road_data[:, :, :, : model_parameters.num_track, ro], axis=3
    )

    # Additional per-road and track-collapsed data needed across panels
    E_road_data_temp = np.sum(
        model_variables.E_road_data[:, :, :, :, : model_parameters.num_track, ro],
        axis=4,
    )
    traffic_data_temp = converted_data.traffic_data[:, :, ro]
    activity_data_temp = converted_data.activity_data[:, :, ro]

    # Observations
    PM_obs_net = airquality_data.PM_obs_net

    # Date axis for this road
    date_num = converted_data.date_data[constants.datenum_index, :, ro]

    # Dispersion factor for masking (per MATLAB: mask where f_conc or obs are nodata)
    f_conc_temp = model_variables.f_conc[:, ro]

    # Subplot 1: Concentrations
    # Handle NaNs analogous to MATLAB nodata masking
    valid_mask = (
        (PM_obs_net[x, :] != metadata.nodata)
        & ~np.isnan(PM_obs_net[x, :])
        & (f_conc_temp != metadata.nodata)
    )
    valid_indices = np.flatnonzero(valid_mask)
    invalid_mask = ~valid_mask
    C_data_temp2 = C_data_temp.copy()
    C_data_temp2[:, x, :, invalid_mask] = np.nan
    PM_obs_net_temp = PM_obs_net.copy()
    PM_obs_net_temp[x, invalid_mask] = np.nan

    # Build series
    if x == constants.pm_course:
        # Coarse fraction derived as PM10 - PM2.5
        total_series = np.nansum(
            C_data_temp2[
                constants.all_source_index, constants.pm_10, constants.C_total_index, :
            ],
            axis=0,
        ) - np.nansum(
            C_data_temp2[
                constants.all_source_index, constants.pm_25, constants.C_total_index, :
            ],
            axis=0,
        )
        obs_series = (
            PM_obs_net_temp[constants.pm_10, :] - PM_obs_net_temp[constants.pm_25, :]
        )
        salt_na_series = (
            C_data_temp2[
                constants.salt_index[0], constants.pm_10, constants.C_total_index, :
            ]
            - C_data_temp2[
                constants.salt_index[0], constants.pm_25, constants.C_total_index, :
            ]
        )
        salt_2_series = (
            C_data_temp2[
                constants.salt_index[1], constants.pm_10, constants.C_total_index, :
            ]
            - C_data_temp2[
                constants.salt_index[1], constants.pm_25, constants.C_total_index, :
            ]
        )
        wear_series = np.nansum(
            C_data_temp2[
                constants.wear_index, constants.pm_10, constants.C_total_index, :
            ],
            axis=0,
        ) - np.nansum(
            C_data_temp2[
                constants.wear_index, constants.pm_25, constants.C_total_index, :
            ],
            axis=0,
        )
        sand_series = (
            C_data_temp2[
                constants.sand_index, constants.pm_10, constants.C_total_index, :
            ]
            - C_data_temp2[
                constants.sand_index, constants.pm_25, constants.C_total_index, :
            ]
        )

        x_str1, xplot, y_total = _prepare_series(
            date_num=date_num, series=total_series, time_config=time_config, av=av
        )
        _, _, y_obs = _prepare_series(
            date_num=date_num, series=obs_series, time_config=time_config, av=av
        )
        _, _, y_salt_na = _prepare_series(
            date_num=date_num, series=salt_na_series, time_config=time_config, av=av
        )
        _, _, y_salt_2 = _prepare_series(
            date_num=date_num, series=salt_2_series, time_config=time_config, av=av
        )
        _, _, y_wear = _prepare_series(
            date_num=date_num, series=wear_series, time_config=time_config, av=av
        )
        _, _, y_sand = _prepare_series(
            date_num=date_num, series=sand_series, time_config=time_config, av=av
        )
    else:
        x_str1, xplot, y_total = _prepare_series(
            date_num=date_num,
            series=np.nansum(
                C_data_temp2[constants.all_source_index, x, constants.C_total_index, :],
                axis=0,
            ),
            time_config=time_config,
            av=av,
        )
    _, _, y_obs = _prepare_series(
        date_num=date_num,
        series=PM_obs_net_temp[x, :],
        time_config=time_config,
        av=av,
    )
    _, _, y_salt_na = _prepare_series(
        date_num=date_num,
        series=C_data_temp2[constants.salt_index[0], x, constants.C_total_index, :],
        time_config=time_config,
        av=av,
    )
    _, _, y_salt_2 = _prepare_series(
        date_num=date_num,
        series=C_data_temp2[constants.salt_index[1], x, constants.C_total_index, :],
        time_config=time_config,
        av=av,
    )
    _, _, y_wear = _prepare_series(
        date_num=date_num,
        series=np.nansum(
            C_data_temp2[constants.wear_index, x, constants.C_total_index, :],
            axis=0,
        ),
        time_config=time_config,
        av=av,
    )
    _, _, y_sand = _prepare_series(
        date_num=date_num,
        series=C_data_temp2[constants.sand_index, x, constants.C_total_index, :],
        time_config=time_config,
        av=av,
    )

    fig = plt.figure()
    # Set window title if backend supports it
    try:
        fig.canvas.manager.set_window_title("Figure 13: Summary")
    except Exception:
        pass
    gs = GridSpec(4, 3, figure=fig)
    # Expose indices/masks for potential external use
    fig._pm_valid_indices = valid_indices  # type: ignore[attr-defined]
    fig._pm_invalid_mask = invalid_mask  # type: ignore[attr-defined]
    ax1 = fig.add_subplot(gs[0, :])
    title_str = paths.title_str if paths and paths.title_str else "NORTRIP"
    pm_text = (
        "PM10"
        if x == constants.pm_10
        else (
            "PM2.5"
            if x == constants.pm_25
            else ("PM coarse" if x == constants.pm_course else "PM")
        )
    )
    ax1.set_title(f"{title_str}: {pm_text}")
    av_index = model_flags.plot_type_flag
    if av_index in (3, 5):
        # Use string ticks for daily cycle or weekday plots
        ax1.plot(xplot, y_obs, "k--", linewidth=1, label="Observed")
        ax1.plot(xplot, y_salt_na, "g-", linewidth=1, label="Modelled salt(na)")
        ax1.plot(xplot, y_salt_2, "g--", linewidth=1, label="Modelled salt(mg)")
        ax1.plot(xplot, y_wear, "r:", linewidth=1, label="Modelled wear")
        ax1.plot(xplot, y_sand, "m--", linewidth=1, label="Modelled sand")
        ax1.plot(xplot, y_total, "b-", linewidth=1, label="Modelled total")
        ax1.set_xticks(xplot)
        ax1.set_xticklabels(x_str1)
    else:
        dt_x = _matlab_datenum_to_datetime_array(xplot)
        ax1.plot(dt_x, y_obs, "k--", linewidth=1, label="Observed")
        ax1.plot(dt_x, y_salt_na, "g-", linewidth=1, label="Modelled salt(na)")
        ax1.plot(dt_x, y_salt_2, "g--", linewidth=1, label="Modelled salt(mg)")
        ax1.plot(dt_x, y_wear, "r:", linewidth=1, label="Modelled wear")
        ax1.plot(dt_x, y_sand, "m--", linewidth=1, label="Modelled sand")
        ax1.plot(dt_x, y_total, "b-", linewidth=1, label="Modelled total")
        _format_time_axis(ax1, dt_x, av_index)
    ax1.set_ylabel(f"{pm_text} concentration (µg/m³)")
    y_max_1 = float(
        np.nanmax(np.vstack([y_total, y_obs, y_salt_na, y_salt_2, y_wear, y_sand]))
    )
    if np.isfinite(y_max_1) and y_max_1 > 0:
        ax1.set_ylim(0, y_max_1 * 1.1)
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.2)

    # Subplot 2: Mass loading
    # Convert linear mass per road width (mg/m) to areal mass (g/m^2)
    # MATLAB: b_factor = 1/1000/b_road_lanes
    lanes_width = metadata.b_road_lanes
    b_factor = 1.0 / 1000.0 / max(float(lanes_width), 1.0)
    x_load = constants.pm_200
    x_str2, xplot2, y_dust = _prepare_series(
        date_num=date_num,
        series=M_road_data_temp[constants.total_dust_index, x_load, :] * b_factor,
        time_config=time_config,
        av=av,
    )
    _, _, y_salt_na_m = _prepare_series(
        date_num=date_num,
        series=M_road_data_temp[constants.salt_index[0], x_load, :] * b_factor,
        time_config=time_config,
        av=av,
    )
    _, _, y_salt_2_m = _prepare_series(
        date_num=date_num,
        series=M_road_data_temp[constants.salt_index[1], x_load, :] * b_factor,
        time_config=time_config,
        av=av,
    )
    _, _, y_sand_m = _prepare_series(
        date_num=date_num,
        series=M_road_data_temp[constants.sand_index, x_load, :] * b_factor,
        time_config=time_config,
        av=av,
    )
    ax2 = fig.add_subplot(gs[1, :])
    if av_index in (3, 5):
        ax2.plot(xplot2, y_dust, "k-", linewidth=1, label="Suspendable dust")
        ax2.plot(xplot2, y_salt_na_m, "g-", linewidth=1, label="Salt(na)")
        ax2.plot(xplot2, y_salt_2_m, "g--", linewidth=1, label="Salt(mg)")
        ax2.plot(xplot2, y_sand_m, "r--", linewidth=1, label="Suspendable sand")
        ax2.set_xticks(xplot2)
        ax2.set_xticklabels(x_str2)
    else:
        dt_x2 = _matlab_datenum_to_datetime_array(xplot2)
        ax2.plot(dt_x2, y_dust, "k-", linewidth=1, label="Suspendable dust")
        ax2.plot(dt_x2, y_salt_na_m, "g-", linewidth=1, label="Salt(na)")
        ax2.plot(dt_x2, y_salt_2_m, "g--", linewidth=1, label="Salt(mg)")
        ax2.plot(dt_x2, y_sand_m, "r--", linewidth=1, label="Suspendable sand")
        _format_time_axis(ax2, dt_x2, av_index)
    ax2.set_ylabel("Mass loading (g/m²)")
    y_max_2 = float(np.nanmax(np.vstack([y_dust, y_salt_na_m, y_salt_2_m, y_sand_m])))
    if np.isfinite(y_max_2) and y_max_2 > 0:
        ax2.set_ylim(0, y_max_2 * 1.1)

    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.2)

    # Subplot: Scatter daily mean observed vs modelled concentrations
    ax_scatter = fig.add_subplot(gs[2, 0])
    _, _, y_model_sc = _prepare_series(
        date_num=date_num,
        series=np.nansum(
            C_data_temp2[constants.all_source_index, x, constants.C_total_index, :],
            axis=0,
        ),
        time_config=time_config,
        av=av,
    )
    _, _, y_obs_sc = _prepare_series(
        date_num=date_num, series=PM_obs_net_temp[x, :], time_config=time_config, av=av
    )
    # Ensure 1D arrays for regression
    y_model_sc = np.asarray(y_model_sc).reshape(-1)
    y_obs_sc = np.asarray(y_obs_sc).reshape(-1)
    r = np.where(~np.isnan(y_model_sc) & ~np.isnan(y_obs_sc))[0]
    ax_scatter.set_title(
        "Scatter daily mean"
        if getattr(airquality_data, "EP_emis_available", 0)
        else "Scatter daily mean no exhaust"
    )
    if r.size > 0:
        ax_scatter.plot(y_model_sc[r], y_obs_sc[r], "bo", markersize=3)
        max_plot = np.nanmax([np.nanmax(y_model_sc[r]), np.nanmax(y_obs_sc[r])])
        if np.isfinite(max_plot) and max_plot > 0:
            ax_scatter.set_xlim(0, max_plot)
            ax_scatter.set_ylim(0, max_plot)
        a1, a0 = np.polyfit(y_model_sc[r], y_obs_sc[r], 1)
        xmin = float(np.nanmin(y_model_sc[r]))
        xmax = float(np.nanmax(y_model_sc[r]))
        ax_scatter.plot(
            [xmin, xmax], [a0 + a1 * xmin, a0 + a1 * xmax], "-", color=(0.5, 0.5, 0.5)
        )
        corr_matrix = np.corrcoef(y_model_sc[r], y_obs_sc[r])
        r_sq = float(corr_matrix[0, 1] ** 2)
        rmse_val = float(np.sqrt(np.nanmean((y_model_sc[r] - y_obs_sc[r]) ** 2)))
        fb = (
            (np.nanmean(y_model_sc[r]) - np.nanmean(y_obs_sc[r]))
            / max(np.nanmean(y_model_sc[r]) + np.nanmean(y_obs_sc[r]), 1e-9)
            * 2
        )
        ax_scatter.text(
            0.05,
            0.95,
            f"r² = {r_sq:4.2f}, FB = {fb:4.2f}",
            transform=ax_scatter.transAxes,
        )
        ax_scatter.text(
            0.05, 0.87, f"RMSE = {rmse_val:4.1f}", transform=ax_scatter.transAxes
        )
        ax_scatter.text(
            0.05,
            0.79,
            f"OBS = {np.nanmean(y_obs_sc[r]):4.1f}",
            transform=ax_scatter.transAxes,
        )
        ax_scatter.text(
            0.05,
            0.71,
            f"MOD = {np.nanmean(y_model_sc[r]):4.1f}",
            transform=ax_scatter.transAxes,
        )
        ax_scatter.text(0.55, 0.18, f"a_0 = {a0:4.1f}", transform=ax_scatter.transAxes)
        ax_scatter.text(0.55, 0.10, f"a_1 = {a1:4.2f}", transform=ax_scatter.transAxes)
    ax_scatter.set_xlabel(f"{pm_text} modelled concentration (µg/m³)")
    ax_scatter.set_ylabel(f"{pm_text} observed concentration (µg/m³)")

    # Subplot: Mean concentrations bar chart and emissions EF bar chart
    time_slice = slice(time_config.min_time, time_config.max_time + 1)
    # Concentration bars
    ax_concbar = fig.add_subplot(gs[2, 2])
    r_mask = (PM_obs_net_temp[x, time_slice] != metadata.nodata) & (
        f_conc_temp[time_slice] != metadata.nodata
    )
    C_all_m_temp2 = C_data_temp[:, :, constants.C_total_index, time_slice]

    def _nanmean_or_nan(arr: np.ndarray) -> float:
        arr = np.asarray(arr)
        if arr.size == 0:
            return float("nan")
        mask = ~np.isnan(arr)
        if not np.any(mask):
            return float("nan")
        return float(np.mean(arr[mask]))

    def _mean_or_zero(arr: np.ndarray) -> float:
        arr = np.asarray(arr)
        if arr.size == 0:
            return 0.0
        mask = ~np.isnan(arr)
        if not np.any(mask):
            return 0.0
        val = float(np.mean(arr[mask]))
        return val if np.isfinite(val) else 0.0

    # dust_conc retained in MATLAB for reporting; not displayed directly in bars here
    _ = _mean_or_zero(
        C_all_m_temp2[constants.total_dust_index, x, r_mask]
        - C_all_m_temp2[constants.sand_index, x, r_mask]
        - C_all_m_temp2[constants.exhaust_index, x, r_mask]
    )
    sand_conc = _mean_or_zero(C_all_m_temp2[constants.sand_index, x, r_mask])
    salt_na_conc = _mean_or_zero(C_all_m_temp2[constants.salt_index[0], x, r_mask])
    salt_2_conc = _mean_or_zero(C_all_m_temp2[constants.salt_index[1], x, r_mask])
    exhaust_conc = _mean_or_zero(C_all_m_temp2[constants.exhaust_index, x, r_mask])
    roadwear_conc = _mean_or_zero(C_all_m_temp2[constants.road_index, x, r_mask])
    tyrewear_conc = _mean_or_zero(C_all_m_temp2[constants.tyre_index, x, r_mask])
    brakewear_conc = _mean_or_zero(C_all_m_temp2[constants.brake_index, x, r_mask])
    total_conc = _mean_or_zero(
        np.nansum(C_all_m_temp2[0 : constants.num_source, x, r_mask], axis=0)
    )
    observed_conc = _mean_or_zero(PM_obs_net_temp[x, time_slice][r_mask])

    conc_vals = [
        observed_conc,
        total_conc,
        roadwear_conc,
        tyrewear_conc,
        brakewear_conc,
        sand_conc,
        salt_na_conc,
        salt_2_conc,
        exhaust_conc,
    ]
    bar_conc = ax_concbar.bar(range(len(conc_vals)), conc_vals, color="r")
    # Observed bar in black
    if len(bar_conc) > 0:
        bar_conc[0].set_color("k")
    ax_concbar.set_xticks(range(len(conc_vals)))
    ax_concbar.set_xticklabels(
        ["obs", "mod", "road", "tyre", "brake", "sand", "na", "mg", "exh"]
    )
    ax_concbar.set_title("Mean concentrations")
    ax_concbar.set_ylabel(f"Concentration {pm_text} (µg/m³)")
    for rect, val in zip(bar_conc, conc_vals, strict=False):
        if np.isfinite(val):
            ax_concbar.text(
                rect.get_x() + rect.get_width() / 2.0,
                rect.get_height(),
                f"{val:5.1f}",
                ha="center",
                va="bottom",
            )

    # Emission factor bars
    ax_emis = fig.add_subplot(gs[2, 1])
    # Build emissions series according to selected size (handle coarse as PM10-PM2.5)
    if x == constants.pm_course:
        E_total_series = (
            E_road_data_temp[
                constants.total_dust_index,
                constants.pm_10,
                constants.E_total_index,
                time_slice,
            ]
            - E_road_data_temp[
                constants.total_dust_index,
                constants.pm_25,
                constants.E_total_index,
                time_slice,
            ]
        )
        E_dir_series = np.nansum(
            E_road_data_temp[
                constants.dust_noexhaust_index,
                constants.pm_10,
                constants.E_direct_index,
                time_slice,
            ],
            axis=0,
        ) - np.nansum(
            E_road_data_temp[
                constants.dust_noexhaust_index,
                constants.pm_25,
                constants.E_direct_index,
                time_slice,
            ],
            axis=0,
        )
        E_susp_series = np.nansum(
            E_road_data_temp[
                constants.dust_noexhaust_index,
                constants.pm_10,
                constants.E_suspension_index,
                time_slice,
            ],
            axis=0,
        ) - np.nansum(
            E_road_data_temp[
                constants.dust_noexhaust_index,
                constants.pm_25,
                constants.E_suspension_index,
                time_slice,
            ],
            axis=0,
        )
        E_exh_series = (
            E_road_data_temp[
                constants.exhaust_index,
                constants.pm_10,
                constants.E_total_index,
                time_slice,
            ]
            - E_road_data_temp[
                constants.exhaust_index,
                constants.pm_25,
                constants.E_total_index,
                time_slice,
            ]
        )
    else:
        E_total_series = E_road_data_temp[
            constants.total_dust_index, x, constants.E_total_index, time_slice
        ]
        E_dir_series = np.nansum(
            E_road_data_temp[
                constants.dust_noexhaust_index, x, constants.E_direct_index, time_slice
            ],
            axis=0,
        )
        E_susp_series = np.nansum(
            E_road_data_temp[
                constants.dust_noexhaust_index,
                x,
                constants.E_suspension_index,
                time_slice,
            ],
            axis=0,
        )
        E_exh_series = E_road_data_temp[
            constants.exhaust_index, x, constants.E_total_index, time_slice
        ]

    # Fill NaNs with 0 to emulate MATLAB behavior (arrays are initialized to 0)
    E_all = float(np.mean(np.nan_to_num(E_total_series, nan=0.0)))
    E_dir_all = float(np.mean(np.nan_to_num(E_dir_series, nan=0.0)))
    E_susp_all = float(np.mean(np.nan_to_num(E_susp_series, nan=0.0)))
    E_exh = float(np.mean(np.nan_to_num(E_exh_series, nan=0.0)))
    # If modeled exhaust is unavailable/zero, fall back to EP_emis if provided (MATLAB uses EP_emis when available)
    if (not np.isfinite(E_exh)) or E_exh <= 0:
        if getattr(airquality_data, "EP_emis_available", 0):
            ep_slice = airquality_data.EP_emis[
                time_config.min_time : time_config.max_time + 1
            ]
            ep_slice = np.asarray(ep_slice, dtype=float)
            # mask nodata
            ep_mask = (ep_slice != metadata.nodata) & ~np.isnan(ep_slice)
            if np.any(ep_mask):
                ep_mean = float(np.nanmean(ep_slice[ep_mask]))
                if np.isfinite(ep_mean) and ep_mean > 0:
                    E_exh = ep_mean

    # Mean hourly traffic (match MATLAB mean_AHT calculation)
    dt = time_config.dt
    mean_ADT_all_ef = np.zeros((constants.num_tyre, constants.num_veh))
    for t in range(constants.num_tyre):
        for v in range(constants.num_veh):
            idx = constants.N_t_v_index[(t, v)]
            mean_ADT_all_ef[t, v] = (
                float(np.nanmean(traffic_data_temp[idx, time_slice])) * 24.0 * dt
            )
    mean_ADT_ef = np.nansum(mean_ADT_all_ef, axis=0)
    mean_AHT = float(np.nansum(mean_ADT_ef)) / 24.0
    denom_ef = max(mean_AHT, 1e-9)
    f_conc_slice = f_conc_temp[time_slice]
    mask_f = (f_conc_slice != metadata.nodata) & ~np.isnan(f_conc_slice)
    mean_f_conc_val = _nanmean_or_nan(f_conc_slice[mask_f])
    obs_emission = (
        observed_conc / mean_f_conc_val
        if (np.isfinite(mean_f_conc_val) and mean_f_conc_val > 0)
        else float("nan")
    )
    # Model emission from modeled concentrations if E_all is zero/NaN (fallback to match MATLAB intent)
    mod_emission_from_conc = (
        total_conc / mean_f_conc_val
        if (np.isfinite(mean_f_conc_val) and mean_f_conc_val > 0)
        else float("nan")
    )
    mod_emission_val = (
        E_all if (np.isfinite(E_all) and E_all > 0) else mod_emission_from_conc
    )
    emis_vals = [obs_emission, mod_emission_val, E_dir_all, E_susp_all, E_exh]
    emis_bar_heights = (np.array(emis_vals) / denom_ef) * 1000.0
    bar_emis = ax_emis.bar([0, 1, 2, 3, 4], emis_bar_heights, color="g")
    # Observed bar in black
    if len(bar_emis) > 0:
        bar_emis[0].set_color("k")
    ax_emis.set_xticks([0, 1, 2, 3, 4])
    ax_emis.set_xticklabels(["Obs.", "Mod.", "Dir.", "Sus.", "Exh."])
    ax_emis.set_title("Mean emission factor")
    ax_emis.set_ylabel(f"Emission factor {pm_text} (mg/km/veh)")
    for rect, val in zip(bar_emis, emis_bar_heights, strict=False):
        if np.isfinite(val):
            ax_emis.text(
                rect.get_x() + rect.get_width() / 2.0,
                rect.get_height(),
                f"{val:5.0f}",
                ha="center",
                va="bottom",
            )

    # Bottom row: three text panels
    # Use global style's font sizes for panel text (no local overrides)
    # Traffic and activity
    ax_text1 = fig.add_subplot(gs[3, 0])
    ax_text1.axis("off")
    dt = time_config.dt
    num_days = (
        len(converted_data.date_data[constants.year_index, time_slice]) * dt / 24.0
    )
    mean_ADT_all = np.zeros((constants.num_tyre, constants.num_veh))
    for t in range(constants.num_tyre):
        for v in range(constants.num_veh):
            idx = constants.N_t_v_index[(t, v)]
            mean_ADT_all[t, v] = float(
                np.nanmean(traffic_data_temp[idx, time_slice]) * 24.0 * dt
            )
    mean_ADT = np.nansum(mean_ADT_all, axis=0)
    mean_speed_li = float(
        np.nansum(
            traffic_data_temp[constants.V_veh_index[constants.li], time_slice]
            * traffic_data_temp[constants.N_v_index[constants.li], time_slice]
        )
        / max(
            np.nansum(traffic_data_temp[constants.N_v_index[constants.li], time_slice]),
            1e-9,
        )
    )
    mean_speed_he = float(
        np.nansum(
            traffic_data_temp[constants.V_veh_index[constants.he], time_slice]
            * traffic_data_temp[constants.N_v_index[constants.he], time_slice]
        )
        / max(
            np.nansum(traffic_data_temp[constants.N_v_index[constants.he], time_slice]),
            1e-9,
        )
    )
    rsalting1 = np.where(
        activity_data_temp[constants.M_salting_index[0], time_slice] > 0
    )[0]
    rsalting2 = np.where(
        activity_data_temp[constants.M_salting_index[1], time_slice] > 0
    )[0]
    rsanding = np.where(activity_data_temp[constants.M_sanding_index, time_slice] > 0)[
        0
    ]
    rcleaning = np.where(
        activity_data_temp[constants.t_cleaning_index, time_slice] > 0
    )[0]
    rploughing = np.where(
        activity_data_temp[constants.t_ploughing_index, time_slice] > 0
    )[0]
    salting_total = float(
        np.nansum(activity_data_temp[constants.M_salting_index, time_slice])
        * max(getattr(metadata, "b_road_lanes", 1.0), 1.0)
        * 1000.0
    )
    lines1 = [
        "Traffic and activity",
        f"Mean ADT = {float(np.nansum(mean_ADT)):.0f} (veh)",
        f"Mean ADT (li / he) = {float(mean_ADT[constants.li] / max(np.nansum(mean_ADT), 1e-9) * 100):4.1f} / {float(mean_ADT[constants.he] / max(np.nansum(mean_ADT), 1e-9) * 100):4.1f} (%)",
        f"Mean speed (li / he) = {mean_speed_li:4.1f} / {mean_speed_he:4.1f} (km/hr)",
        f"Number of days = {num_days:4.1f}",
        f"Number salting events (na/mg) = {len(rsalting1):3.0f}/{len(rsalting2):3.0f}",
        f"Number sanding events = {len(rsanding):4.0f}",
        f"Number cleaning events = {len(rcleaning):4.0f}",
        f"Number ploughing events = {len(rploughing):4.0f}",
        f"Total salt (ton/km) = {salting_total * 1e-6:4.2f}",
    ]
    for i, tline in enumerate(lines1):
        ax_text1.text(0.0, 1 - i * 0.1, tline)

    # Meteorology
    ax_text2 = fig.add_subplot(gs[3, 1])
    ax_text2.axis("off")
    total_precip = float(
        np.nansum(
            converted_data.meteo_data[constants.Snow_precip_index, time_slice, ro]
            + converted_data.meteo_data[constants.Rain_precip_index, time_slice, ro]
        )
    )
    rsnow = np.where(
        converted_data.meteo_data[constants.Snow_precip_index, time_slice, ro] > 0
    )[0]
    rrain = np.where(
        converted_data.meteo_data[constants.Rain_precip_index, time_slice, ro] > 0
    )[0]
    freq_precip = (len(rrain) + len(rsnow)) / max(
        len(converted_data.meteo_data[constants.Rain_precip_index, time_slice, ro]), 1
    )
    mean_RH = float(
        np.nanmean(converted_data.meteo_data[constants.RH_index, time_slice, ro])
    )
    mean_Ta = float(
        np.nanmean(converted_data.meteo_data[constants.T_a_index, time_slice, ro])
    )
    mean_cloud = float(
        np.nanmean(
            converted_data.meteo_data[constants.cloud_cover_index, time_slice, ro]
        )
    )
    mean_short_rad = float(
        np.nanmean(
            converted_data.meteo_data[constants.short_rad_in_index, time_slice, ro]
        )
    )
    # If road-level shortwave net not available, skip it
    if (
        hasattr(constants, "short_rad_net_index")
        and model_variables.road_meteo_data.size
    ):
        # approximate by track-sum/mean if present; here use first track
        mean_short_rad_net = float(
            np.nanmean(
                model_variables.road_meteo_data[
                    constants.short_rad_net_index, time_slice, 0, ro
                ]
            )
        )
    else:
        mean_short_rad_net = np.nan
    lines2 = [
        "Meteorology",
        f"Mean Temp (C) = {mean_Ta:4.2f}",
        f"Mean RH (%) = {mean_RH:4.1f}",
        f"Mean global/net (W/m²) = {mean_short_rad:4.1f}/{mean_short_rad_net:4.1f}",
        f"Mean cloud cover (%) = {mean_cloud * 100:4.1f}",
        f"Total precip (mm) = {total_precip:4.1f}",
        f"Frequency precip (%) = {freq_precip * 100:4.1f}",
    ]
    for i, tline in enumerate(lines2):
        ax_text2.text(0.0, 1 - i * 0.1, tline)

    # Concentration stats
    ax_text3 = fig.add_subplot(gs[3, 2])
    ax_text3.axis("off")
    _, _, PM10_mod_net_av = _prepare_series(
        date_num=date_num,
        series=np.nansum(
            C_data_temp[
                0 : constants.num_source, constants.pm_10, constants.C_total_index, :
            ],
            axis=0,
        ),
        time_config=time_config,
        av=av,
    )
    _, _, PM10_obs_net_av = _prepare_series(
        date_num=date_num,
        series=airquality_data.PM_obs_net[constants.pm_10, :],
        time_config=time_config,
        av=av,
    )
    r_av = np.where(~np.isnan(PM10_mod_net_av) & ~np.isnan(PM10_obs_net_av))[0]
    if r_av.size:
        rmse_net = float(
            np.sqrt(np.nanmean((PM10_obs_net_av[r_av] - PM10_mod_net_av[r_av]) ** 2))
        )
        nrmse_net = (
            rmse_net / max(float(np.nanmean(PM10_obs_net_av[r_av])), 1e-9) * 100.0
        )
        fb_net = (
            (
                float(np.nanmean(PM10_mod_net_av[r_av]))
                - float(np.nanmean(PM10_obs_net_av[r_av]))
            )
            / max(
                float(
                    np.nanmean(PM10_mod_net_av[r_av])
                    + np.nanmean(PM10_obs_net_av[r_av])
                ),
                1e-9,
            )
            * 2.0
            * 100.0
        )
        lines3 = [
            f"Mean obs (net) = {float(np.nanmean(PM10_obs_net_av[r_av])):4.1f}",
            f"Mean mod (net) = {float(np.nanmean(PM10_mod_net_av[r_av])):4.1f}",
            f"NRMSE(%) = {nrmse_net:4.1f}",
            f"FB(%) = {fb_net:4.1f}",
        ]
    else:
        lines3 = ["Concentration stats unavailable"]
    for i, tline in enumerate(lines3):
        ax_text3.text(0.0, 1 - i * 0.12, tline)

    # Make y-axis labels slightly smaller than the global label size
    shrink_y_labels([ax1, ax2, ax_scatter, ax_concbar, ax_emis], points=2)

    fig.tight_layout()
    plt.show()
