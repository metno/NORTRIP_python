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

    # ---------------- Panel 1: Concentrations (µg/m³) ----------------

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

    # Figure using a 4x3 grid: rows 0-1 span all 3 cols; row 2 has 3 small panels
    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(
        4, 3, hspace=0.3, wspace=0.25, left=0.06, right=0.99, top=0.95, bottom=0.06
    )
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
    ax1 = fig.add_subplot(gs[0, :])
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
    ax1.set_ylabel(f"{pm_text} concentration (µg/m³)", fontsize=6)
    # ax1.set_xlabel(shared.xlabel_text)

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
            ax1.set_ylim(-10.0, y_max)
    except Exception:
        pass

    # ---------------- Panel 2: Mass loading (g/m²) ----------------
    ax2 = fig.add_subplot(gs[1, :])
    ax2.set_title("")
    ax2.set_ylabel("Mass loading (g/m²)", fontsize=6)

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
    # ax2.set_xlabel(shared.xlabel_text)
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
            ax2.set_ylim(-10.0, y_max2)
    except Exception:
        pass

    # ---------------- Panel 3: Scatter daily mean (modeled vs observed) ----------------
    ax3 = fig.add_subplot(gs[2, 0])
    # Title depending on exhaust availability is not in shared; default to generic
    ax3.set_title("Scatter daily mean")
    ax3.set_ylabel(f"{pm_text} observed concentration (µg/m³)", fontsize=6)
    ax3.set_xlabel(f"{pm_text} modelled concentration (µg/m³)", fontsize=7)

    x_vals = np.asarray(y_total_model).squeeze()
    y_vals = np.asarray(y_obs).squeeze()
    valid = np.isfinite(x_vals) & np.isfinite(y_vals)

    if np.any(valid):
        xm = x_vals[valid]
        ym = y_vals[valid]
        ax3.scatter(xm, ym, facecolor="none", edgecolor="blue", s=10)

        max_plot = float(np.nanmax([np.nanmax(xm), np.nanmax(ym)]))
        if np.isfinite(max_plot):
            ax3.set_xlim(0, max_plot)
            ax3.set_ylim(0, max_plot)

        # Statistics
        if xm.size >= 2:
            try:
                R = np.corrcoef(xm, ym)
                r_sq = float(R[0, 1] ** 2)
            except Exception:
                r_sq = np.nan
            rmse = float(np.sqrt(np.nanmean((xm - ym) ** 2)))
            mean_obs = float(np.nanmean(ym))
            mean_mod = float(np.nanmean(xm))
            with np.errstate(invalid="ignore", divide="ignore"):
                _fbias = float((mean_mod - mean_obs) / (mean_mod + mean_obs) * 2.0)
            try:
                a1, a0 = np.polyfit(xm, ym, 1)
            except Exception:
                a1, a0 = np.nan, np.nan

            # Text annotations (normalized coordinates)
            ax3.text(
                0.05, 0.95, f"r²  = {r_sq:4.2f}", transform=ax3.transAxes, va="top"
            )
            ax3.text(
                0.05,
                0.85,
                f"RMSE = {rmse:4.1f} (µg/m³)",
                transform=ax3.transAxes,
                va="top",
            )
            ax3.text(
                0.05,
                0.75,
                f"OBS  = {mean_obs:4.1f} (µg/m³)",
                transform=ax3.transAxes,
                va="top",
            )
            ax3.text(
                0.05,
                0.65,
                f"MOD  = {mean_mod:4.1f} (µg/m³)",
                transform=ax3.transAxes,
                va="top",
            )

            # Regression line across observed x-range
            xmin = float(np.nanmin(xm))
            xmax = float(np.nanmax(xm))
            if (
                np.isfinite(a0)
                and np.isfinite(a1)
                and np.isfinite(xmin)
                and np.isfinite(xmax)
            ):
                ax3.plot(
                    [xmin, xmax],
                    [a0 + a1 * xmin, a0 + a1 * xmax],
                    "-",
                    color=(0.5, 0.5, 0.5),
                )

    # ---------------- Panel 4: Mean emission factor bar chart ----------------
    ax4 = fig.add_subplot(gs[2, 1])
    # Emission arrays and traffic within selected window
    mask_range = slice(i_min, i_max + 1)
    E = shared.E_road_data_sum_tracks.astype(float).copy()
    E[E == nodata] = np.nan
    traffic = shared.traffic_data_ro.astype(float).copy()
    traffic[traffic == nodata] = np.nan

    # Series across time window
    E_total_series = E[
        constants.total_dust_index, x_size, constants.E_total_index, mask_range
    ]
    E_exhaust_series = E[
        constants.exhaust_index, x_size, constants.E_total_index, mask_range
    ]
    E_direct_series = np.sum(
        E[constants.dust_noexhaust_index, x_size, constants.E_direct_index, mask_range],
        axis=0,
    )
    E_susp_series = np.sum(
        E[
            constants.dust_noexhaust_index,
            x_size,
            constants.E_suspension_index,
            mask_range,
        ],
        axis=0,
    )
    N_total_series = traffic[constants.N_total_index, mask_range]

    # Means across window
    total_emissions = float(np.nanmean(E_total_series))
    direct_emissions = float(np.nanmean(E_direct_series))
    suspension_emissions = float(np.nanmean(E_susp_series))
    exhaust_emissions = float(np.nanmean(E_exhaust_series))
    mean_AHT = float(np.nanmean(N_total_series))  # average hourly traffic (veh/hr)

    # Observed emission from concentrations and f_conc (window masks follow MATLAB)
    PM_obs = shared.PM_obs_net.astype(float).copy()
    PM_obs[PM_obs == nodata] = np.nan
    obs_series_win = PM_obs[x_size, mask_range]
    f_conc = shared.f_conc.astype(float).copy()
    f_conc[f_conc == nodata] = np.nan
    f_conc_win = f_conc[mask_range]
    valid_obs = np.isfinite(obs_series_win) & np.isfinite(f_conc_win)
    observed_concentrations = (
        float(np.nanmean(obs_series_win[valid_obs])) if np.any(valid_obs) else np.nan
    )
    valid_fconc = np.isfinite(f_conc_win)
    mean_f_conc = (
        float(np.nanmean(f_conc_win[valid_fconc])) if np.any(valid_fconc) else np.nan
    )
    observed_emission = (
        observed_concentrations / mean_f_conc
        if np.isfinite(observed_concentrations)
        and np.isfinite(mean_f_conc)
        and mean_f_conc != 0
        else np.nan
    )

    # Convert to mg/km/veh via divide by mean_AHT (veh/hr) then *1000
    def to_mg_per_km_per_veh(val: float) -> float:
        if not np.isfinite(val) or not np.isfinite(mean_AHT) or mean_AHT == 0:
            return np.nan
        return float(val / mean_AHT * 1000.0)

    ploty1 = [
        to_mg_per_km_per_veh(observed_emission),
        to_mg_per_km_per_veh(total_emissions),
        to_mg_per_km_per_veh(direct_emissions),
        to_mg_per_km_per_veh(suspension_emissions),
        to_mg_per_km_per_veh(exhaust_emissions),
    ]
    ploty2 = [ploty1[0], 0, 0, 0, 0]

    # Bars
    x_positions = np.arange(1, 6)
    ax4.bar(x_positions, ploty1, color="#30ff30", edgecolor="k", linewidth=0.5)
    ax4.bar(x_positions, ploty2, color="k")
    ax4.set_xticks(x_positions)
    ax4.set_xticklabels(["Obs.", "Mod.", "Dir.", "Sus.", "Exh."])
    ax4.set_title("Mean emission factor")
    ax4.set_ylabel(f"Emission factor {pm_text} (mg/km/veh)", fontsize=6)

    # Value labels above bars
    for xpos, val in zip(x_positions, ploty1, strict=False):
        if np.isfinite(val) and val > 0:
            ax4.text(
                xpos,
                val,
                f"{val:5.0f}",
                ha="center",
                va="bottom",
            )
    ax4.set_xlim(0, 6)
    ax4.set_ylim(bottom=0)
