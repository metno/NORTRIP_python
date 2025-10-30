import numpy as np
import matplotlib.pyplot as plt
import os

import constants
from functions.average_data_func import average_data_func
from config_classes import model_file_paths
from .shared_plot_data import shared_plot_data
from .helpers import (
    unix_timestamp_to_datetime_array,
    format_time_axis,
    generate_plot_filename,
)


def plot_summary(shared: shared_plot_data, paths: model_file_paths) -> None:
    """
    Render the full summary figure (MATLAB figure 13) using a 4x3 layout.

    Panels:
    - Row 1 (spans 3 cols):
      1) Time series of concentrations: observed vs modelled source contributions
    - Row 2 (spans 3 cols):
      2) Mass loading (g/m²): suspendable dust, salt(na), salt(salt2_str), sand, optional cleaning stairs
    - Row 3 (3 cols):
      3) Scatter (daily mean): modelled vs observed, with stats and regression line
      4) Mean emission factor (mg/km/veh): observed, model total, direct, suspension, exhaust
      5) Mean concentrations (µg/m³): observed, total, road/tyre/brake/sand/salt(na)/salt(salt2_str)/exhaust
    - Row 4 (3 cols):
      6) Traffic and activity summary (text block)
      7) Meteorology summary (text block)
      8) Concentration statistics (text block): percentiles, 36th-highest, exceedances, comparable hours

    Inputs are provided via `shared_plot_data` and `model_file_paths`.
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
    if shared.av[0] in (3, 5):
        dt_x = xplot
    else:
        dt_x = unix_timestamp_to_datetime_array(xplot)

    # Figure using a 4x3 grid: rows 0-1 span all 3 cols; row 2 has 3 small panels
    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(
        4, 3, hspace=0.3, wspace=0.25, left=0.06, right=0.99, top=0.95, bottom=0.06
    )
    try:
        fig.canvas.manager.set_window_title("Figure 11: Summary")  # type: ignore
    except Exception:
        pass

    # Title text with PM fraction
    if x_size == constants.pm_10:
        pm_text = "PM10"
    elif x_size == constants.pm_25:
        pm_text = "PM2.5"
    else:
        pm_text = "PM"

    title_str = paths.title_str or "Summary"
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_title(f"{title_str}: {pm_text}")

    # Hide values below zero (set to NaN) to mirror MATLAB not displaying negatives

    # Plot series in the MATLAB order/styles
    ax1.plot(dt_x, y_obs.squeeze(), "k--", linewidth=1, label="Observed")
    ax1.plot(dt_x, y_salt_na.squeeze(), "g-", linewidth=1, label="Modelled salt(na)")
    ax1.plot(
        dt_x,
        y_salt_mg.squeeze(),
        "g--",
        linewidth=1,
        label=f"Modelled salt({shared.salt2_str})",
    )
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

    if shared.av[0] in (3, 5):
        dt_x2 = xplot2
    else:
        dt_x2 = unix_timestamp_to_datetime_array(xplot2)

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
        f"Salt({shared.salt2_str})",
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
            f"Salt({shared.salt2_str})",
            "Suspendable sand",
        ]

    # Plot series
    ax2.plot(dt_x2, y_dust.squeeze(), "k-", linewidth=1, label="Suspendable dust")
    ax2.plot(dt_x2, y_salt_na.squeeze(), "g-", linewidth=1, label="Salt(na)")
    ax2.plot(
        dt_x2,
        y_salt_mg.squeeze(),
        "g--",
        linewidth=1,
        label=f"Salt({shared.salt2_str})",
    )
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
            # Fractional bias not displayed in scatter panel; compute later for tables
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
                float(xpos),
                val,
                f"{val:5.0f}",
                ha="center",
                va="bottom",
            )
    ax4.set_xlim(0, 6)
    ax4.set_ylim(bottom=0)

    # ---------------- Panel 5: Mean concentrations bar chart ----------------
    ax5 = fig.add_subplot(gs[2, 2])
    # Concentration arrays within selected window
    C = shared.C_data_sum_tracks.astype(float).copy()
    C[C == nodata] = np.nan
    # Use same window and validity mask as for observed emissions
    # (valid when both PM_obs and f_conc are finite)
    # Build series
    total_c_series = np.sum(
        C[
            constants.all_source_index,
            x_size,
            constants.C_total_index,
            mask_range,
        ],
        axis=0,
    )
    roadwear_c_series = C[
        constants.road_index, x_size, constants.C_total_index, mask_range
    ]
    tyrewear_c_series = C[
        constants.tyre_index, x_size, constants.C_total_index, mask_range
    ]
    brakewear_c_series = C[
        constants.brake_index, x_size, constants.C_total_index, mask_range
    ]
    sand_c_series = C[constants.sand_index, x_size, constants.C_total_index, mask_range]
    salt_na_c_series = C[
        constants.salt_index[0], x_size, constants.C_total_index, mask_range
    ]
    salt_mg_c_series = C[
        constants.salt_index[1], x_size, constants.C_total_index, mask_range
    ]
    exhaust_c_series = C[
        constants.exhaust_index, x_size, constants.C_total_index, mask_range
    ]

    # Means using valid_obs mask from earlier computation
    if np.any(valid_obs):
        total_concentrations = float(np.nanmean(total_c_series[valid_obs]))
        roadwear_concentrations = float(np.nanmean(roadwear_c_series[valid_obs]))
        tyrewear_concentrations = float(np.nanmean(tyrewear_c_series[valid_obs]))
        brakewear_concentrations = float(np.nanmean(brakewear_c_series[valid_obs]))
        sand_concentrations = float(np.nanmean(sand_c_series[valid_obs]))
        salt_na_concentrations = float(np.nanmean(salt_na_c_series[valid_obs]))
        salt_mg_concentrations = float(np.nanmean(salt_mg_c_series[valid_obs]))
        exhaust_concentrations = float(np.nanmean(exhaust_c_series[valid_obs]))
    else:
        total_concentrations = np.nan
        roadwear_concentrations = np.nan
        tyrewear_concentrations = np.nan
        brakewear_concentrations = np.nan
        sand_concentrations = np.nan
        salt_na_concentrations = np.nan
        salt_mg_concentrations = np.nan
        exhaust_concentrations = np.nan

    ploty1_c = [
        observed_concentrations,
        total_concentrations,
        roadwear_concentrations,
        tyrewear_concentrations,
        brakewear_concentrations,
        sand_concentrations,
        salt_na_concentrations,
        salt_mg_concentrations,
        exhaust_concentrations,
    ]
    ploty2_c = [ploty1_c[0]] + [0] * 8

    x_positions_c = np.arange(1, 10)
    ax5.bar(x_positions_c, ploty1_c, color="#d62728", edgecolor="k", linewidth=0.5)
    ax5.bar(x_positions_c, ploty2_c, color="k")
    ax5.set_xticks(x_positions_c)
    ax5.set_xticklabels(
        [
            "obs",
            "mod",
            "road",
            "tyre",
            "brake",
            "sand",
            "na",
            f"{shared.salt2_str}",
            "exh",
        ]
    )
    ax5.set_title("Mean concentrations")
    ax5.set_ylabel(f"Concentration {pm_text} (µg/m³)")

    for xpos, val in zip(x_positions_c, ploty1_c, strict=False):
        if np.isfinite(val) and val > 0:
            ax5.text(float(xpos), val, f"{val:5.1f}", ha="center", va="bottom")
    ax5.set_xlim(0, 10)
    ax5.set_ylim(bottom=0)

    # ---------------- Panel 6: Traffic and activity text block ----------------
    ax6 = fig.add_subplot(gs[3, 0])
    ax6.axis("off")

    # Compute metrics over selected window
    li = constants.li
    he = constants.he
    st = constants.st
    tmask = mask_range
    traffic = shared.traffic_data_ro.astype(float).copy()
    traffic[traffic == nodata] = np.nan

    # Mean hourly flows
    N_li = traffic[constants.N_v_index[li], tmask]
    N_he = traffic[constants.N_v_index[he], tmask]
    mean_hour_li = float(np.nanmean(N_li)) if np.any(np.isfinite(N_li)) else np.nan
    mean_hour_he = float(np.nanmean(N_he)) if np.any(np.isfinite(N_he)) else np.nan
    mean_ADT_li = mean_hour_li * 24.0 if np.isfinite(mean_hour_li) else np.nan
    mean_ADT_he = mean_hour_he * 24.0 if np.isfinite(mean_hour_he) else np.nan
    mean_ADT_total = (
        mean_ADT_li + mean_ADT_he
        if np.isfinite(mean_ADT_li) and np.isfinite(mean_ADT_he)
        else np.nan
    )

    # Mean speeds (traffic-weighted)
    V_li = traffic[constants.V_veh_index[li], tmask]
    V_he = traffic[constants.V_veh_index[he], tmask]

    def weighted_mean(val: np.ndarray, w: np.ndarray) -> float:
        val = np.asarray(val, dtype=float)
        w = np.asarray(w, dtype=float)
        with np.errstate(invalid="ignore"):
            num = np.nansum(val * w)
            den = np.nansum(w)
        return float(num / den) if den > 0 else np.nan

    mean_speed_li = weighted_mean(V_li, N_li)
    mean_speed_he = weighted_mean(V_he, N_he)

    # Studded proportions by vehicle type
    N_st_li = traffic[constants.N_t_v_index[(st, li)], tmask]
    N_st_he = traffic[constants.N_t_v_index[(st, he)], tmask]
    mean_ADT_st_li = (
        float(np.nanmean(N_st_li)) * 24.0 if np.any(np.isfinite(N_st_li)) else np.nan
    )
    mean_ADT_st_he = (
        float(np.nanmean(N_st_he)) * 24.0 if np.any(np.isfinite(N_st_he)) else np.nan
    )
    prop_st_li = (
        float(mean_ADT_st_li / mean_ADT_li)
        if np.isfinite(mean_ADT_st_li) and np.isfinite(mean_ADT_li) and mean_ADT_li > 0
        else np.nan
    )
    prop_st_he = (
        float(mean_ADT_st_he / mean_ADT_he)
        if np.isfinite(mean_ADT_st_he) and np.isfinite(mean_ADT_he) and mean_ADT_he > 0
        else np.nan
    )

    # Event counts
    activity = shared.activity_data_ro.astype(float).copy()
    activity[activity == nodata] = np.nan
    num_salting_na = int(
        np.nansum((activity[constants.M_salting_index[0], tmask] > 0).astype(int))
    )
    num_salting_2 = int(
        np.nansum((activity[constants.M_salting_index[1], tmask] > 0).astype(int))
    )
    num_sanding = int(
        np.nansum((activity[constants.M_sanding_index, tmask] > 0).astype(int))
    )
    num_cleaning = int(
        np.nansum((activity[constants.t_cleaning_index, tmask] > 0).astype(int))
    )
    num_ploughing = int(
        np.nansum((activity[constants.t_ploughing_index, tmask] > 0).astype(int))
    )

    # Number of days in window from datetimes
    dt_slice = unix_timestamp_to_datetime_array(shared.date_num[i_min : i_max + 1])
    if len(dt_slice) >= 2:
        num_days = (dt_slice[-1] - dt_slice[0]).total_seconds() / 86400.0
    else:
        num_days = 0.0

    # Render as stacked text with offsets
    title = "Traffic and activity"
    lines = [
        f"Mean ADT  = {mean_ADT_total:4.0f} (veh)",
        f"Mean ADT (li / he) = {mean_ADT_li / (mean_ADT_total) * 100:4.1f} / {mean_ADT_he / (mean_ADT_total) * 100:4.1f} (%)"
        if np.isfinite(mean_ADT_total) and mean_ADT_total > 0
        else "Mean ADT (li / he) = n/a / n/a (%)",
        f"Mean speed (li / he) = {mean_speed_li:4.1f} / {mean_speed_he:4.1f} (km/hr)",
        f"Studded (li / he) = {prop_st_li * 100:4.1f} / {prop_st_he * 100:4.1f} (%)"
        if np.isfinite(prop_st_li) and np.isfinite(prop_st_he)
        else "Studded (li / he) = n/a / n/a (%)",
        f"Number of days = {num_days:4.1f}",
        f"Number salting events (na/{shared.salt2_str}) = {num_salting_na:3.0f}/{num_salting_2:3.0f}",
        f"Number sanding events = {num_sanding:4.0f}",
        f"Number cleaning events = {num_cleaning:4.0f}",
        f"Number ploughing events = {num_ploughing:4.0f}",
    ]

    y = 1.0
    dy = 0.1
    ax6.text(0.0, y, title, transform=ax6.transAxes, fontweight="bold")
    for text in lines:
        y -= dy
        ax6.text(0.0, y, text, transform=ax6.transAxes)

    # ---------------- Panel 7: Meteorology text block ----------------
    ax7 = fig.add_subplot(gs[3, 1])
    ax7.axis("off")

    meteo = shared.meteo_data_ro.astype(float).copy()
    meteo[meteo == nodata] = np.nan
    roadm = shared.road_meteo_weighted.astype(float).copy()
    roadm[roadm == nodata] = np.nan

    # Means over window
    mean_Ta = float(np.nanmean(meteo[constants.T_a_index, mask_range]))
    mean_RH = float(np.nanmean(meteo[constants.RH_index, mask_range]))
    # Short-wave radiation: global from meteo, net from road meteo
    mean_short_rad = float(np.nanmean(meteo[constants.short_rad_in_index, mask_range]))
    mean_short_rad_net = float(
        np.nanmean(roadm[constants.short_rad_net_index, mask_range])
    )
    mean_cloud = float(np.nanmean(meteo[constants.cloud_cover_index, mask_range]))

    # Precip totals and frequency
    rain = meteo[constants.Rain_precip_index, mask_range]
    snow = meteo[constants.Snow_precip_index, mask_range]
    total_precip = float(np.nansum(rain + snow))
    # Frequency based on any precip per timestep
    # Count any precip in a timestep only once
    precip_any = (np.nan_to_num(rain, nan=0.0) > 0) | (np.nan_to_num(snow, nan=0.0) > 0)
    freq_precip = float(np.mean(precip_any)) if precip_any.size > 0 else 0.0

    # Wet/dry proportions using f_q
    fq = shared.f_q_weighted.astype(float).copy()
    fq[fq == nodata] = np.nan
    fq_obs = shared.f_q_obs_weighted.astype(float).copy()
    fq_obs[fq_obs == nodata] = np.nan
    fq_road_win = fq[constants.road_index, mask_range]
    fq_obs_win = fq_obs[mask_range]

    is_wet_mod = fq_road_win < 0.5
    is_wet_obs = fq_obs_win < 0.5
    prop_wet = (
        float(np.nanmean(is_wet_mod.astype(float))) if is_wet_mod.size > 0 else 0.0
    )
    # relative freq = model wet count / observed wet count
    wet_mod_count = int(np.nansum(is_wet_mod.astype(float)))
    wet_obs_count = int(np.nansum(is_wet_obs.astype(float)))
    rel_prop_wet = float(wet_mod_count / wet_obs_count) if wet_obs_count > 0 else np.nan

    # Moisture hits (matching sign of wet/dry classification)
    # Map to -1 (wet) and +1 (dry)
    fq_obs_sign = np.where(is_wet_obs, -1, 1)
    fq_mod_sign = np.where(is_wet_mod, -1, 1)
    valid_hits = np.isfinite(fq_obs_win) & np.isfinite(fq_road_win)
    hits = np.sum((fq_obs_sign[valid_hits] * fq_mod_sign[valid_hits]) > 0)
    total_classified = int(np.sum(valid_hits))
    f_q_hits = (
        float(hits / total_classified * 100.0) if total_classified > 0 else np.nan
    )

    # Mean dispersion from earlier computed mean_f_conc (fallback to window mean)
    if np.isfinite(mean_f_conc):
        f_conc_win = shared.f_conc[mask_range].astype(float)
        f_conc_win[f_conc_win == nodata] = np.nan
        mean_f_conc = float(np.nanmean(f_conc_win))

    title2 = "Meteorology"
    lines2 = [
        f"Mean Temperature = {mean_Ta:4.2f} (°C)",
        f"Mean RH = {mean_RH:4.1f} (%)",
        f"Mean short radiation global/net = {mean_short_rad:4.1f}/{mean_short_rad_net:4.1f} (W/m²)",
        f"Mean cloud cover = {mean_cloud * 100:4.1f} (%)",
        f"Total precipitation = {total_precip:4.1f} (mm)",
        f"Frequency precipitation = {freq_precip * 100:4.1f} (%)",
        f"Frequency wet road = {prop_wet * 100:4.1f} (%)",
        f"Relative freq wet road = {rel_prop_wet:4.2f}",
        f"Surface moisture hits = {f_q_hits:4.1f} (%)",
        f"Mean dispersion = {mean_f_conc:4.3f} (µg/m³·(g/km/hr)⁻¹)",
    ]

    y = 1.0
    dy = 0.1
    ax7.text(0.0, y, title2, transform=ax7.transAxes, fontweight="bold")
    for text in lines2:
        y -= dy
        ax7.text(0.0, y, text, transform=ax7.transAxes)

    # ---------------- Panel 8: Concentrations text block ----------------
    ax8 = fig.add_subplot(gs[3, 2])
    ax8.axis("off")

    # Build averaged series for percentiles/exceedances (net and with background)
    PM_obs_bg_all = shared.PM_obs_bg.astype(float).copy()
    PM_obs_bg_all[PM_obs_bg_all == nodata] = np.nan
    obs_bg_series_win = PM_obs_bg_all[x_size, mask_range]

    # Daily/selected averaging using same av as charts
    _xstr_bg, _xplot_bg, y_obs_bg = average_data_func(
        date_num, PM_obs_bg_all[x_size, :n_date], i_min, i_max, av
    )

    # Mask for availability across all three averaged series
    y_obs_net_av = np.asarray(y_obs).squeeze()
    y_mod_net_av = np.asarray(y_total_model).squeeze()
    y_obs_bg_av = np.asarray(y_obs_bg).squeeze()
    r_av = (
        np.isfinite(y_obs_net_av) & np.isfinite(y_mod_net_av) & np.isfinite(y_obs_bg_av)
    )

    # Percentiles (match MATLAB index rounding approach)
    def percentile_from_sorted(arr: np.ndarray, p: float) -> float:
        arr_valid = np.asarray(arr, dtype=float)
        arr_valid = arr_valid[np.isfinite(arr_valid)]
        if arr_valid.size == 0:
            return float("nan")
        arr_sorted = np.sort(arr_valid)
        idx = int(np.round(arr_sorted.size * p / 100.0)) - 1
        idx = max(0, min(arr_sorted.size - 1, idx))
        return float(arr_sorted[idx])

    per = 90.0
    obs_c_per = percentile_from_sorted(y_obs_net_av[r_av], per)
    mod_c_per = percentile_from_sorted(y_mod_net_av[r_av], per)
    obs_c_bg_per = percentile_from_sorted((y_obs_net_av + y_obs_bg_av)[r_av], per)
    mod_c_bg_per = percentile_from_sorted((y_mod_net_av + y_obs_bg_av)[r_av], per)

    # 36th highest values
    days_lim = 36

    def _nth_highest(arr: np.ndarray, n: int) -> float:
        arr_valid = np.asarray(arr, dtype=float)
        arr_valid = arr_valid[np.isfinite(arr_valid)]
        if arr_valid.size < n or arr_valid.size == 0:
            return 0.0
        arr_sorted = np.sort(arr_valid)  # ascending
        return float(arr_sorted[-n])

    high36_obs = _nth_highest(y_obs_net_av[r_av], days_lim)
    high36_mod = _nth_highest(y_mod_net_av[r_av], days_lim)
    high36_obs_bg = _nth_highest((y_obs_net_av + y_obs_bg_av)[r_av], days_lim)
    high36_mod_bg = _nth_highest((y_mod_net_av + y_obs_bg_av)[r_av], days_lim)

    # Exceedances over limit for averaged series
    limit = 50.0
    obs_c_ex = int(np.nansum(y_obs_net_av[r_av] > limit))
    mod_c_ex = int(np.nansum(y_mod_net_av[r_av] > limit))
    obs_c_bg_ex = int(np.nansum((y_obs_net_av + y_obs_bg_av)[r_av] > limit))
    mod_c_bg_ex = int(np.nansum((y_mod_net_av + y_obs_bg_av)[r_av] > limit))

    # Comparable hours: fraction of timesteps with both obs and f_conc available
    f_conc_win_ch = shared.f_conc[mask_range].astype(float)
    f_conc_win_ch[f_conc_win_ch == nodata] = np.nan
    pm_obs_win_ch = shared.PM_obs_net[x_size, mask_range].astype(float)
    pm_obs_win_ch[pm_obs_win_ch == nodata] = np.nan
    valid_hours = np.isfinite(f_conc_win_ch) & np.isfinite(pm_obs_win_ch)
    comparable_hours = (
        float(np.nansum(valid_hours) / valid_hours.size)
        if valid_hours.size > 0
        else float("nan")
    )

    # Means for text (net and total)
    # observed_concentrations was computed earlier; compute background mean (with mask including bg)
    valid_bg = (
        np.isfinite(obs_bg_series_win)
        & np.isfinite(f_conc_win_ch)
        & np.isfinite(pm_obs_win_ch)
    )
    observed_concentrations_bg = (
        float(np.nanmean(obs_bg_series_win[valid_bg]))
        if np.any(valid_bg)
        else float("nan")
    )

    title3 = f"Concentrations {pm_text}"
    lines3 = [
        f"Mean obs (net,total) = {observed_concentrations:4.1f}, {observed_concentrations + observed_concentrations_bg:4.1f} (µg/m³)",
        f"Mean model (net,total) = {total_concentrations:4.1f}, {total_concentrations + observed_concentrations_bg:4.1f} (µg/m³)",
        f"Mean background obs = {observed_concentrations_bg:4.1f} (µg/m³)",
        f"90th per obs (net,total)  = {obs_c_per:4.1f}, {obs_c_bg_per:4.1f} (µg/m³)",
        f"90th per model (net,total) = {mod_c_per:4.1f}, {mod_c_bg_per:4.1f} (µg/m³)",
        f"36th highest obs (net,total)  = {high36_obs:4.1f}, {high36_obs_bg:4.1f} (µg/m³)",
        f"36th highest model (net,total) = {high36_mod:4.1f}, {high36_mod_bg:4.1f} (µg/m³)",
        f"Days>50 µg/m³ obs (net,total) = {obs_c_ex:4.0f}, {obs_c_bg_ex:4.0f} (days)",
        f"Days>50 µg/m³ model (net,total) = {mod_c_ex:4.0f}, {mod_c_bg_ex:4.0f} (days)",
        f"Comparable hours = {comparable_hours * 100:4.1f} %",
    ]

    y = 1.0
    dy = 0.1
    ax8.text(0.0, y, title3, transform=ax8.transAxes, fontweight="bold")
    for text in lines3:
        y -= dy
        ax8.text(0.0, y, text, transform=ax8.transAxes)

    # ---------------- Optional textual summaries (post-plot prints) ----------------
    if shared.print_result:
        # 1) Traffic and activity data
        # Mean ADT totals and HDV share
        mean_ADT_li_safe = mean_ADT_li if np.isfinite(mean_ADT_li) else 0.0
        mean_ADT_he_safe = mean_ADT_he if np.isfinite(mean_ADT_he) else 0.0
        mean_ADT_total_safe = (
            mean_ADT_total
            if np.isfinite(mean_ADT_total)
            else (mean_ADT_li_safe + mean_ADT_he_safe)
        )
        hdv_pct = (
            mean_ADT_he_safe / mean_ADT_total_safe * 100.0
            if mean_ADT_total_safe > 0
            else float("nan")
        )

        # Mean speed as simple average of light/heavy
        speeds = [v for v in [mean_speed_li, mean_speed_he] if np.isfinite(v)]
        mean_speed_avg = float(np.mean(speeds)) if len(speeds) > 0 else float("nan")

        # Studded proportions
        mean_studded_ldv_pct = (
            prop_st_li * 100.0 if np.isfinite(prop_st_li) else float("nan")
        )

        # Max studded share for LDV across window
        N_li_series = traffic[constants.N_v_index[li], mask_range]
        N_st_li_series = traffic[constants.N_t_v_index[(st, li)], mask_range]
        with np.errstate(invalid="ignore", divide="ignore"):
            frac_st_li_series = N_st_li_series / N_li_series
        max_prop_studded_li_pct = (
            float(np.nanmax(frac_st_li_series) * 100.0)
            if np.any(np.isfinite(frac_st_li_series))
            else float("nan")
        )

        # Total salt activity over window (ton/km)
        # Algebra from MATLAB: salting_total = sum(M_salting) * b_road_lanes * 1000
        # With b_factor = 1/1000/ b_road_lanes, so salting_total = sum(M_salting) / b_factor
        activity_win = shared.activity_data_ro.astype(float).copy()
        activity_win[activity_win == nodata] = np.nan
        salting_sum = float(
            np.nansum(activity_win[constants.M_salting_index[0], mask_range])
            + np.nansum(activity_win[constants.M_salting_index[1], mask_range])
        )
        salting_total_g_per_km = (
            salting_sum / shared.b_factor
            if shared.b_factor not in (0.0, np.nan)
            else float("nan")
        )
        salting_total_ton_per_km = (
            salting_total_g_per_km * 1e-6
            if np.isfinite(salting_total_g_per_km)
            else float("nan")
        )

        print("Traffic and activity data")
        print(
            f"{'Number of days':<18}\t{'Mean ADT':<18}\t{'HDV (%)':<18}\t{'Mean speed (km/hr)':<18}\t"
            f"{'Mean studded (%LDV)':<18}\t{'Max studded (%LDV)':<18}\t{'Total salt (ton/km)':<18}\t"
            f"{'Salting(1) events':<18}\t{'Salting(2) events':<18}\t{'Sanding events':<18}\t{'Cleaning events':<18}\t{'Ploughing events':<18}"
        )
        print(
            f"{num_days:<18.2f}\t{(mean_ADT_li_safe + mean_ADT_he_safe):<18.0f}\t{hdv_pct:<18.2f}\t{mean_speed_avg:<18.2f}\t"
            f"{mean_studded_ldv_pct:<18.2f}\t{max_prop_studded_li_pct:<18.2f}\t{salting_total_ton_per_km:<18.2f}\t"
            f"{num_salting_na:<18.2f}\t{num_salting_2:<18.2f}\t{num_sanding:<18.2f}\t{num_cleaning:<18.2f}\t{num_ploughing:<18.2f}"
        )

        # 2) Meteorological data
        print("Meteorological data")
        print(
            f"{'Mean Temp(C)':<18}\t{'Mean RH(%)':<18}\t{'Mean global(W/m^2)':<18}\t{'Mean cloud cover(%)':<18}\t"
            f"{'Total precip(mm)':<18}\t{'Frequency precip(%)':<18}\t{'Frequency wet(%)':<18}\t{'Mean dispersion':<18}"
        )
        print(
            f"{mean_Ta:<18.2f}\t{mean_RH:<18.2f}\t{mean_short_rad:<18.2f}\t{(mean_cloud * 100.0):<18.2f}\t"
            f"{total_precip:<18.2f}\t{(freq_precip * 100.0):<18.2f}\t{(prop_wet * 100.0):<18.2f}\t{mean_f_conc:<18.3f}"
        )

        # 3) Source contribution (ug/m^3)
        print("Source contribution (ug/m^3)")
        print(
            f"{'Observed total':<18}\t{'Model total':<18}\t{'Model road':<18}\t{'Model tyre':<18}\t{'Model brake':<18}\t"
            f"{'Model sand':<18}\t{'Model salt(na)':<18}\t{'Model salt(' + shared.salt2_str + ')':<18}\t{'Model exhaust':<18}"
        )
        print(
            f"{observed_concentrations:<18.2f}\t{total_concentrations:<18.2f}\t{roadwear_concentrations:<18.2f}\t{tyrewear_concentrations:<18.2f}\t{brakewear_concentrations:<18.2f}\t"
            f"{sand_concentrations:<18.2f}\t{salt_na_concentrations:<18.2f}\t{salt_mg_concentrations:<18.2f}\t{exhaust_concentrations:<18.2f}"
        )

        # 4) Net concentration results and with background
        # Build averaged series for stats (already computed above as y_obs_net_av, y_mod_net_av, y_obs_bg_av, r_av)
        def _safe_corrcoef(a: np.ndarray, b: np.ndarray) -> float:
            try:
                R = np.corrcoef(a, b)
                return float(R[0, 1] ** 2)
            except Exception:
                return float("nan")

        # RMSE and fractional bias helper
        def _rmse(a: np.ndarray, b: np.ndarray) -> float:
            return float(np.sqrt(np.nanmean((a - b) ** 2)))

        def _fbias(mod: np.ndarray, obs: np.ndarray) -> float:
            m_mod = float(np.nanmean(mod))
            m_obs = float(np.nanmean(obs))
            with np.errstate(invalid="ignore", divide="ignore"):
                fb = (m_mod - m_obs) / (m_mod + m_obs) * 2.0 * 100.0
            return float(fb)

        # r^2 values
        r_sq_net_pm10 = _safe_corrcoef(y_obs_net_av[r_av], y_mod_net_av[r_av])
        r_sq_bg_pm10 = _safe_corrcoef(
            (y_obs_net_av + y_obs_bg_av)[r_av], (y_mod_net_av + y_obs_bg_av)[r_av]
        )

        # RMSE and NRMSE
        rmse_net = _rmse(y_obs_net_av[r_av], y_mod_net_av[r_av])
        rmse_bg = _rmse(
            (y_obs_net_av + y_obs_bg_av)[r_av], (y_mod_net_av + y_obs_bg_av)[r_av]
        )
        nrmse_net = (
            rmse_net / float(np.nanmean(y_obs_net_av[r_av])) * 100.0
            if np.any(r_av)
            else float("nan")
        )
        nrmse_bg = (
            rmse_bg / float(np.nanmean((y_obs_net_av + y_obs_bg_av)[r_av])) * 100.0
            if np.any(r_av)
            else float("nan")
        )
        fb_net = _fbias(y_mod_net_av[r_av], y_obs_net_av[r_av])
        fb_bg = _fbias(
            (y_mod_net_av + y_obs_bg_av)[r_av], (y_obs_net_av + y_obs_bg_av)[r_av]
        )

        # Background-only exceedances to mirror MATLAB difference logic
        bg_exceed_only = int(np.nansum(y_obs_bg_av[r_av] > limit))
        obs_c_dif_ex = int(obs_c_bg_ex - bg_exceed_only)
        mod_c_dif_ex = int(mod_c_bg_ex - bg_exceed_only)

        print("Net concentration results (ug/m^3)")
        print(
            f"{'Obs_mean':<12}\t{'Mod_mean':<12}\t{'Obs_90_per':<12}\t{'Mod_90_per':<12}\t{'Obs_36_high':<12}\t{'Mod_36_high':<12}\t"
            f"{'Obs_ex_50':<12}\t{'Mod_ex_50':<12}\t{'R_sq':<12}\t{'RMSE':<12}\t{'NRMSE(%)':<12}\t{'FB(%)':<12}"
        )
        print(
            f"{observed_concentrations:<12.2f}\t{total_concentrations:<12.2f}\t{obs_c_per:<12.2f}\t{mod_c_per:<12.2f}\t{high36_obs:<12.2f}\t{high36_mod:<12.2f}\t"
            f"{obs_c_dif_ex:<12.2f}\t{mod_c_dif_ex:<12.2f}\t{r_sq_net_pm10:<12.2f}\t{rmse_net:<12.2f}\t{nrmse_net:<12.2f}\t{fb_net:<12.2f}"
        )

        print("With background concentration results (ug/m^3)")
        print(
            f"{'Obs_mean':<12}\t{'Mod_mean':<12}\t{'Obs_90_per':<12}\t{'Mod_90_per':<12}\t{'Obs_36_high':<12}\t{'Mod_36_high':<12}\t"
            f"{'Obs_ex_50':<12}\t{'Mod_ex_50':<12}\t{'R_sq':<12}\t{'RMSE':<12}\t{'NRMSE(%)':<12}\t{'FB(%)':<12}"
        )
        print(
            f"{(observed_concentrations + observed_concentrations_bg):<12.2f}\t{(total_concentrations + observed_concentrations_bg):<12.2f}\t"
            f"{obs_c_bg_per:<12.2f}\t{mod_c_bg_per:<12.2f}\t{high36_obs_bg:<12.2f}\t{high36_mod_bg:<12.2f}\t{obs_c_bg_ex:<12.2f}\t{mod_c_bg_ex:<12.2f}\t"
            f"{r_sq_bg_pm10:<12.2f}\t{rmse_bg:<12.2f}\t{nrmse_bg:<12.2f}\t{fb_bg:<12.2f}"
        )

        # 5) Sensitivity outputs (always printed when print_result is True)
        print("Sensitivity outputs (ug/m^3)")
        print(
            f"{'Obs_mean':<12}\t{'Mod_mean':<12}\t{'Obs_per':<12}\t{'Mod_per':<12}\t{'R_sq':<12}\t{'FB(%)':<12}\t{'rel_WET_freq':<12}\t{'f_q_HITS':<12}"
        )
        print(
            f"{observed_concentrations:<12.2f}\t{total_concentrations:<12.2f}\t{obs_c_per:<12.2f}\t{mod_c_per:<12.2f}\t{r_sq_net_pm10:<12.2f}\t{fb_net:<12.2f}\t{rel_prop_wet:<12.2f}\t{f_q_hits:<12.2f}"
        )

    if shared.save_plots:
        plot_file_name = generate_plot_filename(
            title_str=paths.title_str,
            plot_type_flag=shared.av[0],
            figure_number=11,  # Summary is figure 11
            plot_name="Summary",
            date_num=shared.date_num,
            min_time=shared.i_min,
            max_time=shared.i_max,
        )
        plt.savefig(os.path.join(paths.path_outputfig, plot_file_name))
