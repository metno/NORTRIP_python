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

    # Panel-local copies with nodata masked to NaN (MATLAB-style r/NaN approach)
    f_conc_temp = shared.f_conc.astype(float).copy()
    f_conc_temp[f_conc_temp == nodata] = np.nan
    # Identify bad indices for this panel and set to NaN in local copies
    bad = np.where(
        np.isnan(f_conc_temp)
        | np.isnan(PM_obs_net_temp[x, :n_date])
        | (PM_obs_net_temp[x, :n_date] == nodata)
    )[0]
    if bad.size:
        C_data_temp2[:, :, :, bad] = np.nan
        PM_obs_net_temp[x, bad] = np.nan

    # Compute averaged series using the configured averaging flag
    av = list(shared.av)
    date_num = shared.date_num
    i_min = int(shared.i_min)
    i_max = int(shared.i_max)
    t_min = date_num[i_min]
    t_max = date_num[i_max]

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

    # Plot figure arranged as MATLAB: 4 rows x 3 cols grid
    fig = plt.figure(figsize=(10, 8))
    # Set window title for the figure window
    try:
        fig.canvas.manager.set_window_title("Figure 13: Summary")
    except Exception:
        pass
    gs = fig.add_gridspec(
        4, 3, height_ratios=[1.2, 1.0, 1.0, 0.9], hspace=0.6, wspace=0.4
    )
    # Row 1 spans all columns (concentrations)
    ax1 = fig.add_subplot(gs[0, :])
    # Row 2 spans all columns (mass loading)
    ax2 = fig.add_subplot(gs[1, :])
    # Row 3: scatter, ef bars, conc bars
    ax3 = fig.add_subplot(gs[2, 0])
    ax5 = fig.add_subplot(gs[2, 1])
    ax6 = fig.add_subplot(gs[2, 2])
    # Row 4: three text fields
    ax4 = fig.add_subplot(gs[3, 0])
    ax4.axis("off")
    ax7 = fig.add_subplot(gs[3, 1])
    ax7.axis("off")
    ax8 = fig.add_subplot(gs[3, 2])
    ax8.axis("off")

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

    # Panel 3: scatter plot (model vs observed)
    # Build averaged model total and observed series
    _s, xplot_sc, y_mod_av = average_data_func(
        date_num,
        np.nansum(
            C_data_temp2[
                constants.all_source_index, x, constants.C_total_index, :n_date
            ],
            axis=0,
        ),
        0,
        n_date - 1,
        av,
    )
    _s, _xps, y_obs_av = average_data_func(date_num, y_obs, 0, n_date - 1, av)
    # Mask valid pairs
    y_mod_flat = y_mod_av.squeeze()
    y_obs_flat = y_obs_av.squeeze()
    # Restrict stats to selected time window
    range_mask = (np.array(xplot_sc) >= t_min) & (np.array(xplot_sc) <= t_max)
    mask = range_mask & ~np.isnan(y_mod_flat) & ~np.isnan(y_obs_flat)
    ax3.scatter(
        y_mod_flat[mask], y_obs_flat[mask], s=12, edgecolors="b", facecolors="none"
    )
    ax3.set_xlabel(f"{pm_text} modelled (µg/m³)")
    ax3.set_ylabel(f"{pm_text} observed (µg/m³)")
    if np.any(mask):
        lim_max = float(
            np.nanmax([np.nanmax(y_mod_flat[mask]), np.nanmax(y_obs_flat[mask])])
        )
        if np.isfinite(lim_max) and lim_max > 0:
            ax3.set_xlim(0, lim_max)
            ax3.set_ylim(0, lim_max)
        # Stats
        try:
            R = np.corrcoef(y_mod_flat[mask], y_obs_flat[mask])
            r_sq = float(R[0, 1] ** 2)
        except Exception:
            r_sq = np.nan
        rmse = float(np.sqrt(np.nanmean((y_mod_flat[mask] - y_obs_flat[mask]) ** 2)))
        mean_obs = float(np.nanmean(y_obs_flat[mask]))
        mean_mod = float(np.nanmean(y_mod_flat[mask]))
        try:
            a1, a0 = np.polyfit(y_mod_flat[mask], y_obs_flat[mask], 1)
        except Exception:
            a0, a1 = np.nan, np.nan
        # Annotate
        ax3.text(0.05, 0.95, f"r² = {r_sq:4.2f}", transform=ax3.transAxes, va="top")
        ax3.text(
            0.05, 0.87, f"RMSE = {rmse:4.1f} (µg/m³)", transform=ax3.transAxes, va="top"
        )
        ax3.text(
            0.05,
            0.79,
            f"OBS = {mean_obs:4.1f} (µg/m³)",
            transform=ax3.transAxes,
            va="top",
        )
        ax3.text(
            0.05,
            0.71,
            f"MOD = {mean_mod:4.1f} (µg/m³)",
            transform=ax3.transAxes,
            va="top",
        )
        if np.isfinite(a0) and np.isfinite(a1):
            ax3.text(0.55, 0.20, f"a₀ = {a0:4.1f} (µg/m³)", transform=ax3.transAxes)
            ax3.text(0.55, 0.12, f"a₁ = {a1:4.2f}", transform=ax3.transAxes)
            # Regression line
            xmin = float(np.nanmin(y_mod_flat[mask]))
            xmax = float(np.nanmax(y_mod_flat[mask]))
            ax3.plot(
                [xmin, xmax], [a0 + a1 * xmin, a0 + a1 * xmax], color=(0.5, 0.5, 0.5)
            )
    ax3.grid(True, linestyle=":", alpha=0.3)

    # Panel 4: Traffic and activity (text)
    ax4.axis("off")
    # Mean ADT for li/he
    N_he = shared.traffic_data_ro[constants.N_v_index[constants.he], :n_date]
    N_li = shared.traffic_data_ro[constants.N_v_index[constants.li], :n_date]
    # Estimate dt_h from date_num
    if n_date > 1:
        dt_h = float(np.nanmedian(np.diff(shared.date_num)) * 24.0)
        if not np.isfinite(dt_h) or dt_h <= 0:
            dt_h = 1.0
    else:
        dt_h = 1.0
    N_li_win = N_li[i_min : i_max + 1]
    N_he_win = N_he[i_min : i_max + 1]
    mean_ADT_li = float(np.nanmean(N_li_win)) * 24.0 * dt_h
    mean_ADT_he = float(np.nanmean(N_he_win)) * 24.0 * dt_h
    mean_ADT_total = mean_ADT_li + mean_ADT_he
    # Mean speed weighted by flow
    V_he = shared.traffic_data_ro[constants.V_veh_index[constants.he], :n_date]
    V_li = shared.traffic_data_ro[constants.V_veh_index[constants.li], :n_date]
    V_he_win = V_he[i_min : i_max + 1]
    V_li_win = V_li[i_min : i_max + 1]
    mean_speed_he = (
        float(np.nansum(V_he_win * N_he_win) / np.nansum(N_he_win))
        if np.nansum(N_he) > 0
        else float(np.nan)
    )
    mean_speed_li = (
        float(np.nansum(V_li_win * N_li_win) / np.nansum(N_li_win))
        if np.nansum(N_li) > 0
        else float(np.nan)
    )
    # Events counts
    salting_na = shared.activity_data_ro[constants.M_salting_index[0], :n_date]
    salting_2 = shared.activity_data_ro[constants.M_salting_index[1], :n_date]
    sanding = shared.activity_data_ro[constants.M_sanding_index, :n_date]
    cleaning = shared.activity_data_ro[constants.t_cleaning_index, :n_date]
    ploughing = shared.activity_data_ro[constants.t_ploughing_index, :n_date]
    num_salting_na = int(np.nansum(salting_na[i_min : i_max + 1] > 0))
    num_salting_2 = int(np.nansum(salting_2[i_min : i_max + 1] > 0))
    num_sanding = int(np.nansum(sanding[i_min : i_max + 1] > 0))
    num_cleaning = int(np.nansum(cleaning[i_min : i_max + 1] > 0))
    num_ploughing = int(np.nansum(ploughing[i_min : i_max + 1] > 0))
    # Number of days based on time steps and dt (match MATLAB: length * dt / 24)
    num_steps = int(shared.i_max - shared.i_min + 1)
    num_days = float(num_steps * dt_h / 24.0)
    # Compose text
    y0 = 1.0
    dy = 0.12
    ax4.text(
        0.0, y0, "Traffic and activity", fontweight="bold", transform=ax4.transAxes
    )
    ax4.text(
        0.0, y0 - dy, f"Mean ADT = {mean_ADT_total:4.0f} (veh)", transform=ax4.transAxes
    )
    # Percent split li/he
    total_ADT = mean_ADT_total if mean_ADT_total > 0 else 1.0
    pct_li = mean_ADT_li / total_ADT * 100.0
    pct_he = mean_ADT_he / total_ADT * 100.0
    ax4.text(
        0.0,
        y0 - 2 * dy,
        f"Mean ADT (li / he) = {pct_li:4.1f} / {pct_he:4.1f} (%)",
        transform=ax4.transAxes,
    )
    ax4.text(
        0.0,
        y0 - 3 * dy,
        f"Mean speed (li / he) = {mean_speed_li:4.1f} / {mean_speed_he:4.1f} (km/hr)",
        transform=ax4.transAxes,
    )
    ax4.text(
        0.0, y0 - 4 * dy, f"Number of days = {num_days:4.1f}", transform=ax4.transAxes
    )
    ax4.text(
        0.0,
        y0 - 5 * dy,
        f"Number salting events (na/mg) = {num_salting_na:3d}/{num_salting_2:3d}",
        transform=ax4.transAxes,
    )
    ax4.text(
        0.0,
        y0 - 6 * dy,
        f"Number sanding events = {num_sanding:4d}",
        transform=ax4.transAxes,
    )
    ax4.text(
        0.0,
        y0 - 7 * dy,
        f"Number cleaning events = {num_cleaning:4d}",
        transform=ax4.transAxes,
    )
    ax4.text(
        0.0,
        y0 - 8 * dy,
        f"Number ploughing events = {num_ploughing:4d}",
        transform=ax4.transAxes,
    )

    # Panel (row3 col2): Mean emission factor bars (mg/km/veh)
    # Compute mean hourly traffic (veh/hr)
    N_total = shared.traffic_data_ro[constants.N_total_index, :n_date]
    N_total_win = N_total[i_min : i_max + 1]
    mean_AHT = float(np.nanmean(N_total_win)) if np.nanmean(N_total_win) > 0 else 1.0

    # Model emissions (g/km/hr) arrays over time
    E_total_dust = shared.E_road_data_sum_tracks[
        constants.total_dust_index, x, constants.E_total_index, :n_date
    ]
    E_direct_dust = np.nansum(
        shared.E_road_data_sum_tracks[
            constants.dust_noexhaust_index, x, constants.E_direct_index, :n_date
        ],
        axis=0,
    )
    E_susp_dust = np.nansum(
        shared.E_road_data_sum_tracks[
            constants.dust_noexhaust_index, x, constants.E_suspension_index, :n_date
        ],
        axis=0,
    )
    E_exhaust = shared.E_road_data_sum_tracks[
        constants.exhaust_index, x, constants.E_total_index, :n_date
    ]
    E_total_dust_win = E_total_dust[i_min : i_max + 1]
    E_direct_dust_win = E_direct_dust[i_min : i_max + 1]
    E_susp_dust_win = E_susp_dust[i_min : i_max + 1]
    E_exhaust_win = E_exhaust[i_min : i_max + 1]

    # Observed emissions from concentrations
    # E_obs = PM_obs_net / f_conc (mask invalid incl. nodata)
    f_valid = (
        ~np.isnan(f_conc_temp)
        & (~np.isnan(PM_obs_net_temp[x, :n_date]))
        & (PM_obs_net_temp[x, :n_date] != nodata)
    )
    E_obs_series = np.full(n_date, np.nan)
    E_obs_series[f_valid] = PM_obs_net_temp[x, f_valid] / f_conc_temp[f_valid]
    E_obs_series_win = E_obs_series[i_min : i_max + 1]

    # Convert to mg/km/veh using mean hourly traffic
    obs_ef_mg = float(np.nanmean(E_obs_series_win)) / max(mean_AHT, 1e-9) * 1000.0
    mod_ef_mg = float(np.nanmean(E_total_dust_win)) / max(mean_AHT, 1e-9) * 1000.0
    dir_ef_mg = float(np.nanmean(E_direct_dust_win)) / max(mean_AHT, 1e-9) * 1000.0
    sus_ef_mg = float(np.nanmean(E_susp_dust_win)) / max(mean_AHT, 1e-9) * 1000.0
    exh_ef_mg = float(np.nanmean(E_exhaust_win)) / max(mean_AHT, 1e-9) * 1000.0

    bars_ef = [obs_ef_mg, mod_ef_mg, dir_ef_mg, sus_ef_mg, exh_ef_mg]
    labels_ef = ["Obs.", "Mod.", "Dir.", "Sus.", "Exh."]
    ax5.bar(range(1, 6), bars_ef, color=["k", "g", "g", "g", "g"], alpha=0.8)
    ax5.set_xticks(range(1, 6), labels_ef)
    ax5.set_ylabel(f"Emission factor {pm_text} (mg/km/veh)")
    ax5.set_title("Mean emission factor")
    for i, val in enumerate(bars_ef, start=1):
        if np.isfinite(val) and val > 0:
            ax5.text(i, val, f"{val:5.0f}", ha="center", va="bottom", fontsize=8)

    # Panel (row3 col3): Mean concentrations by source bars (µg/m³)
    # Build mask of valid times for concentrations (like MATLAB r), excluding nodata
    c_valid = (
        (~np.isnan(PM_obs_net_temp[x, :n_date]))
        & (PM_obs_net_temp[x, :n_date] != nodata)
        & (~np.isnan(f_conc_temp))
    )
    c_valid_range = c_valid[i_min : i_max + 1]

    # Source means
    def mean_source(idx: int) -> float:
        vals = C_data_temp2[idx, x, constants.C_total_index, :n_date]
        vals = np.maximum(vals, 0)
        vals_r = vals[i_min : i_max + 1]
        return float(np.nanmean(vals_r[c_valid_range]))

    obs_series = PM_obs_net_temp[x, :n_date]
    obs_series_r = obs_series[i_min : i_max + 1]
    observed_conc = float(np.nanmean(obs_series_r[c_valid_range]))
    total_series = np.nansum(
        np.maximum(
            C_data_temp2[
                constants.all_source_index, x, constants.C_total_index, :n_date
            ],
            0,
        ),
        axis=0,
    )
    total_series_r = total_series[i_min : i_max + 1]
    total_conc = float(np.nanmean(total_series_r[c_valid_range]))
    roadwear_conc = mean_source(constants.road_index)
    tyrewear_conc = mean_source(constants.tyre_index)
    brakewear_conc = mean_source(constants.brake_index)
    sand_conc = mean_source(constants.sand_index)
    salt_na_conc = mean_source(constants.salt_index[0])
    salt_2_conc = mean_source(constants.salt_index[1])
    exhaust_conc = mean_source(constants.exhaust_index)

    bars_conc = [
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
    labels_conc = ["obs", "mod", "road", "tyre", "brake", "sand", "na", "mg", "exh"]
    colors_conc = ["k"] + ["r"] * 8  # observed in black, others in red
    ax6.bar(range(1, 10), bars_conc, color=colors_conc, alpha=0.8)
    ax6.set_xticks(range(1, 10), labels_conc)
    ax6.set_ylabel(f"Concentration {pm_text} (µg/m³)")
    ax6.set_title("Mean concentrations")
    for i, val in enumerate(bars_conc, start=1):
        if np.isfinite(val) and val > 0:
            ax6.text(i, val, f"{val:5.1f}", ha="center", va="bottom", fontsize=8)

    # Panel (row4 col2): Meteorology text
    # Helper to mask nodata -> NaN
    def mask_nd(arr: np.ndarray) -> np.ndarray:
        a = arr.astype(float).copy()
        a[a == nodata] = np.nan
        return a

    # Compute meteorology stats with nodata handling
    Ta_arr = mask_nd(
        shared.meteo_data_ro[constants.T_a_index, shared.i_min : shared.i_max]
    )
    RH_arr = mask_nd(
        shared.meteo_data_ro[constants.RH_index, shared.i_min : shared.i_max]
    )
    short_rad_arr = mask_nd(
        shared.meteo_data_ro[constants.short_rad_in_index, shared.i_min : shared.i_max]
    )
    cloud_arr = mask_nd(
        shared.meteo_data_ro[constants.cloud_cover_index, shared.i_min : shared.i_max]
    )
    rain_arr = mask_nd(
        shared.meteo_data_ro[constants.Rain_precip_index, shared.i_min : shared.i_max]
    )
    snow_arr = mask_nd(
        shared.meteo_data_ro[constants.Snow_precip_index, shared.i_min : shared.i_max]
    )

    mean_Ta = float(np.nanmean(Ta_arr))
    mean_RH = float(np.nanmean(RH_arr))
    mean_short_rad_glob = float(np.nanmean(short_rad_arr))

    # Cloud cover: if values are fractions (<=1.2), convert to %; else assume already %
    valid_cloud = cloud_arr[~np.isnan(cloud_arr)]
    if valid_cloud.size > 0:
        mean_cloud_val = float(np.nanmean(valid_cloud))
        if float(np.nanmax(valid_cloud)) <= 1.2:
            mean_cloud = mean_cloud_val * 100.0
        else:
            mean_cloud = mean_cloud_val
    else:
        mean_cloud = float("nan")

    total_precip = float(np.nansum(np.nan_to_num(rain_arr) + np.nan_to_num(snow_arr)))
    freq_precip = float(
        np.nanmean((np.nan_to_num(rain_arr) + np.nan_to_num(snow_arr)) > 0) * 100.0
    )
    # Wet frequency and mean dispersion
    fq_road = mask_nd(shared.f_q_weighted[constants.road_index, :n_date])
    fq_road_win = fq_road[i_min : i_max + 1]
    prop_wet = float(np.nanmean(fq_road_win < 0.5) * 100.0)
    mean_f_conc = float(np.nanmean(f_conc_temp[i_min : i_max + 1]))

    y0m = 1.0
    dym = 0.12
    ax7.text(0.0, y0m, "Meteorology", fontweight="bold", transform=ax7.transAxes)
    ax7.text(
        0.0,
        y0m - dym,
        f"Mean Temperature = {mean_Ta:4.2f} (°C)",
        transform=ax7.transAxes,
    )
    ax7.text(
        0.0, y0m - 2 * dym, f"Mean RH = {mean_RH:4.1f} (%)", transform=ax7.transAxes
    )
    ax7.text(
        0.0,
        y0m - 3 * dym,
        f"Mean short radiation = {mean_short_rad_glob:4.1f} (W/m²)",
        transform=ax7.transAxes,
    )
    ax7.text(
        0.0,
        y0m - 4 * dym,
        f"Mean cloud cover = {mean_cloud:4.1f} (%)",
        transform=ax7.transAxes,
    )
    ax7.text(
        0.0,
        y0m - 5 * dym,
        f"Total precipitation = {total_precip:4.1f} (mm)",
        transform=ax7.transAxes,
    )
    ax7.text(
        0.0,
        y0m - 6 * dym,
        f"Frequency precipitation = {freq_precip:4.1f} (%)",
        transform=ax7.transAxes,
    )
    ax7.text(
        0.0,
        y0m - 7 * dym,
        f"Frequency wet road = {prop_wet:4.1f} (%)",
        transform=ax7.transAxes,
    )
    ax7.text(
        0.0,
        y0m - 8 * dym,
        f"Mean dispersion = {mean_f_conc:4.3f} (µg/m³·(g/km/hr)⁻¹)",
        transform=ax7.transAxes,
    )

    # Panel (row4 col3): Concentrations info text (net and with background)
    # Build averaged series for background too
    _s, _xpb, y_obs_bg_av = average_data_func(
        date_num, shared.PM_obs_bg[x, :n_date], 0, n_date - 1, av
    )

    # Compute percentiles from averaged series
    def percentile(arr: np.ndarray, p: float) -> float:
        vals = arr.squeeze()
        vals = vals[~np.isnan(vals)]
        if vals.size == 0:
            return float("nan")
        k = max(0, min(len(vals) - 1, int(round(len(vals) * p / 100.0)) - 1))
        return float(np.sort(vals)[k])

    # Restrict averaged stats to selected time window
    range_mask2 = (np.array(xplot_sc) >= t_min) & (np.array(xplot_sc) <= t_max)
    y_obs_av_r = y_obs_av.squeeze()[range_mask2]
    y_mod_av_r = y_mod_av.squeeze()[range_mask2]
    y_obs_bg_av_r = y_obs_bg_av.squeeze()[range_mask2]
    obs_mean = float(np.nanmean(y_obs_av_r))
    mod_mean = float(np.nanmean(y_mod_av_r))
    obs_bg_mean = float(np.nanmean(y_obs_bg_av_r))
    obs_per90 = percentile(y_obs_av_r, 90)
    mod_per90 = percentile(y_mod_av_r, 90)
    obs_bg_per90 = percentile(y_obs_av_r + y_obs_bg_av_r, 90)
    mod_bg_per90 = percentile(y_mod_av_r + y_obs_bg_av_r, 90)

    # 36th highest approximated from sorted averaged values if daily
    def kth_highest(vals: np.ndarray, k: int) -> float:
        v = vals.squeeze()
        v = v[~np.isnan(v)]
        if v.size < k:
            return 0.0
        return float(np.sort(v)[-k])

    high36_obs = kth_highest(y_obs_av_r, 36)
    high36_mod = kth_highest(y_mod_av_r, 36)
    high36_obs_bg = kth_highest(y_obs_av_r + y_obs_bg_av_r, 36)
    high36_mod_bg = kth_highest(y_mod_av_r + y_obs_bg_av_r, 36)

    # Exceedances > 50 (daily means approximation)
    limit = 50.0
    obs_ex_50 = int(np.nansum(y_obs_av_r.squeeze() > limit))
    mod_ex_50 = int(np.nansum(y_mod_av_r.squeeze() > limit))
    obs_bg_ex_50 = int(np.nansum((y_obs_av_r + y_obs_bg_av_r).squeeze() > limit))
    mod_bg_ex_50 = int(np.nansum((y_mod_av_r + y_obs_bg_av_r).squeeze() > limit))

    ax8.text(
        0.0,
        1.0,
        f"Concentrations {pm_text}",
        fontweight="bold",
        transform=ax8.transAxes,
    )
    ax8.text(
        0.0,
        0.90,
        f"Mean obs (net,total) = {obs_mean:4.1f}, {obs_mean + obs_bg_mean:4.1f} (µg/m³)",
        transform=ax8.transAxes,
    )
    ax8.text(
        0.0,
        0.82,
        f"Mean model (net,total) = {mod_mean:4.1f}, {mod_mean + obs_bg_mean:4.1f} (µg/m³)",
        transform=ax8.transAxes,
    )
    ax8.text(
        0.0,
        0.74,
        f"Mean background obs = {obs_bg_mean:4.1f} (µg/m³)",
        transform=ax8.transAxes,
    )
    ax8.text(
        0.0,
        0.66,
        f"90th per obs (net,total) = {obs_per90:4.1f}, {obs_bg_per90:4.1f} (µg/m³)",
        transform=ax8.transAxes,
    )
    ax8.text(
        0.0,
        0.58,
        f"90th per model (net,total) = {mod_per90:4.1f}, {mod_bg_per90:4.1f} (µg/m³)",
        transform=ax8.transAxes,
    )
    ax8.text(
        0.0,
        0.50,
        f"36th highest obs (net,total) = {high36_obs:4.1f}, {high36_obs_bg:4.1f} (µg/m³)",
        transform=ax8.transAxes,
    )
    ax8.text(
        0.0,
        0.42,
        f"36th highest model (net,total) = {high36_mod:4.1f}, {high36_mod_bg:4.1f} (µg/m³)",
        transform=ax8.transAxes,
    )
    ax8.text(
        0.0,
        0.34,
        f"Days > 50 µg/m³ obs (net,total) = {obs_ex_50:4d}, {obs_bg_ex_50:4d}",
        transform=ax8.transAxes,
    )
    ax8.text(
        0.0,
        0.26,
        f"Days > 50 µg/m³ model (net,total) = {mod_ex_50:4d}, {mod_bg_ex_50:4d}",
        transform=ax8.transAxes,
    )

    plt.tight_layout()
