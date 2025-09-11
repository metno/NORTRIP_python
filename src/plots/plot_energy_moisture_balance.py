import numpy as np
import matplotlib.pyplot as plt
import os

import constants
from config_classes import model_file_paths
from functions.average_data_func import average_data_func
from .shared_plot_data import shared_plot_data
from .helpers import matlab_datenum_to_datetime_array, format_time_axis, generate_matlab_style_filename


def plot_energy_moisture_balance(
    shared: shared_plot_data, paths: model_file_paths
) -> None:
    """
    Plot figure 6: Energy and moisture balance (MATLAB figure 6).

    Panel 1: Energy balance terms (W/m²)
    Panel 2: Surface water balance rates (mm/hr)
    """

    # Shorthands
    date_num = shared.date_num
    n_date = shared.date_num.shape[0]
    nodata = shared.nodata
    i_min, i_max = shared.i_min, shared.i_max
    av = list(shared.av)
    day_tick_limit = 150

    # Weighted road meteo
    roadm = shared.road_meteo_weighted.astype(float).copy()
    roadm[roadm == nodata] = np.nan

    # --- Panel 1: Energy balance ---
    # Series per MATLAB code
    _, xplot_e, y_rad_net = average_data_func(
        date_num, roadm[constants.rad_net_index, :n_date], i_min, i_max, av
    )
    _, _, y_short_net = average_data_func(
        date_num, roadm[constants.short_rad_net_index, :n_date], i_min, i_max, av
    )
    _, _, y_long_net = average_data_func(
        date_num, roadm[constants.long_rad_net_index, :n_date], i_min, i_max, av
    )
    _, _, y_H = average_data_func(
        date_num, -roadm[constants.H_index, :n_date], i_min, i_max, av
    )
    _, _, y_L = average_data_func(
        date_num, -roadm[constants.L_index, :n_date], i_min, i_max, av
    )
    _, _, y_G = average_data_func(
        date_num, roadm[constants.G_index, :n_date], i_min, i_max, av
    )
    _, _, y_short_clear = average_data_func(
        date_num,
        roadm[constants.short_rad_net_clearsky_index, :n_date],
        i_min,
        i_max,
        av,
    )
    _, _, y_H_traffic = average_data_func(
        date_num, roadm[constants.H_traffic_index, :n_date], i_min, i_max, av
    )
    _, _, y_G_sub = average_data_func(
        date_num, roadm[constants.G_sub_index, :n_date], i_min, i_max, av
    )

    # Optional energy correction
    has_E_corr = bool(shared.use_energy_correction_flag)
    if has_E_corr and constants.E_correction_index < roadm.shape[0]:
        _, _, y_E_corr = average_data_func(
            date_num, roadm[constants.E_correction_index, :n_date], i_min, i_max, av
        )
    else:
        y_E_corr = None

    # Means over available values where shortwave is valid (as in MATLAB using r = find(~isnan(yplot5)))
    r_valid = np.where(~np.isnan(np.asarray(y_short_net).squeeze()))[0]

    def _mean_on_r(y: np.ndarray) -> float:
        yv = np.asarray(y).squeeze()
        if r_valid.size == 0:
            return float("nan")
        return float(np.nanmean(yv[r_valid]))

    mean_short_net = _mean_on_r(y_short_net)
    mean_long_net = _mean_on_r(y_long_net)
    mean_H = _mean_on_r(y_H)
    mean_L = _mean_on_r(y_L)
    mean_H_traffic = _mean_on_r(y_H_traffic)
    mean_G = _mean_on_r(y_G)
    mean_G_sub = _mean_on_r(y_G_sub)
    mean_short_clear = _mean_on_r(y_short_clear)
    mean_E_corr = _mean_on_r(y_E_corr) if y_E_corr is not None else None

    legend_entries = [
        f"Net shortwave flux = {mean_short_net:4.2f} W/m²",
        f"Net longwave flux = {mean_long_net:4.2f} W/m²",
        f"Sensible heat flux = {mean_H:4.2f} W/m²",
        f"Latent heat flux = {mean_L:4.2f} W/m²",
        f"Surface heat flux = {mean_G:4.2f} W/m²",
        f"Traffic heat flux = {mean_H_traffic:4.2f} W/m²",
        f"Clear sky short = {mean_short_clear:4.2f} W/m²",
        f"Sub-surface heat flux = {mean_G_sub:4.2f} W/m²",
    ]
    if mean_E_corr is not None:
        legend_entries.append(f"Energy correction = {mean_E_corr:4.2f} W/m²")

    dt_x_e = matlab_datenum_to_datetime_array(xplot_e)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7))
    fig.subplots_adjust(hspace=0.35)
    try:
        fig.canvas.manager.set_window_title("Figure 6: Energy and moisture balance")  # type: ignore
    except Exception:
        pass

    title_str = paths.title_str or "Energy and moisture balance"
    ax1.set_title(f"{title_str}: Energy balance")
    # Plot order/styles per MATLAB
    # ax1.plot(dt_x_e, np.asarray(y_rad_net).squeeze(), "k-", linewidth=0.5)
    ax1.plot(
        dt_x_e,
        np.asarray(y_short_net).squeeze(),
        "k--",
        linewidth=1,
        label=legend_entries[0],
    )
    ax1.plot(
        dt_x_e,
        np.asarray(y_long_net).squeeze(),
        "k:",
        linewidth=1,
        label=legend_entries[1],
    )
    ax1.plot(
        dt_x_e, np.asarray(y_H).squeeze(), "r-", linewidth=1, label=legend_entries[2]
    )
    ax1.plot(
        dt_x_e, np.asarray(y_L).squeeze(), "b-", linewidth=1, label=legend_entries[3]
    )
    ax1.plot(
        dt_x_e, np.asarray(y_G).squeeze(), "g-", linewidth=1, label=legend_entries[4]
    )
    ax1.plot(
        dt_x_e,
        np.asarray(y_H_traffic).squeeze(),
        "m--",
        linewidth=1,
        label=legend_entries[5],
    )
    ax1.plot(
        dt_x_e,
        np.asarray(y_short_clear).squeeze(),
        "m:",
        linewidth=1,
        label=legend_entries[6],
    )
    ax1.plot(
        dt_x_e,
        np.asarray(y_G_sub).squeeze(),
        "c-",
        linewidth=1,
        label=legend_entries[7],
    )
    if mean_E_corr is not None and y_E_corr is not None:
        ax1.plot(
            dt_x_e,
            np.asarray(y_E_corr).squeeze(),
            "y-",
            linewidth=1,
            label=f"Energy correction = {mean_E_corr:4.2f} W/m²",
        )
    ax1.set_ylabel("Energy (W/m²)")
    ax1.legend(loc="upper left")
    format_time_axis(ax1, dt_x_e, shared.av[0], day_tick_limit=day_tick_limit)
    if len(dt_x_e) > 0:
        ax1.set_xlim(dt_x_e[0], dt_x_e[-1])
    ax1.autoscale(enable=True, axis="y", tight=True)

    # --- Panel 2: Surface water balance (mm/hr) ---
    gbal = shared.g_road_balance_weighted.astype(float).copy()
    gbal[gbal == nodata] = np.nan
    activity = shared.activity_data_ro.astype(float).copy()
    activity[activity == nodata] = np.nan

    # Evap/condensation: -S_evap + P_evap (water index)
    evap_cond_series = (
        -gbal[constants.water_index, constants.S_evap_index, :n_date]
        + gbal[constants.water_index, constants.P_evap_index, :n_date]
    )
    # Rain - drainage - tau drainage
    rain_minus_drain_series = (
        gbal[constants.water_index, constants.P_precip_index, :n_date]
        - gbal[constants.water_index, constants.S_drainage_index, :n_date]
        - gbal[constants.water_index, constants.S_drainage_tau_index, :n_date]
    )
    melt_series = gbal[constants.water_index, constants.P_melt_index, :n_date]
    freeze_series = -gbal[constants.water_index, constants.S_freeze_index, :n_date]
    spray_series = -gbal[constants.water_index, constants.S_spray_index, :n_date]

    # Wetting series from activity, scaled by dt and flag
    dt_hours = float(shared.dt)
    wetting_series = (
        activity[constants.g_road_wetting_index, :n_date]
        / max(dt_hours, 1e-9)
        * float(shared.use_wetting_data_flag)
    )

    # Drain only for legend stats
    drain_series = (
        -gbal[constants.water_index, constants.S_drainage_index, :n_date]
        - gbal[constants.water_index, constants.S_drainage_tau_index, :n_date]
    )
    rain_only_series = gbal[constants.water_index, constants.P_precip_index, :n_date]

    # Average
    _, xplot_w, y_evap = average_data_func(date_num, evap_cond_series, i_min, i_max, av)
    _, _, y_rain_drain = average_data_func(
        date_num, rain_minus_drain_series, i_min, i_max, av
    )
    _, _, y_melt = average_data_func(date_num, melt_series, i_min, i_max, av)
    _, _, y_freeze = average_data_func(date_num, freeze_series, i_min, i_max, av)
    _, _, y_spray = average_data_func(date_num, spray_series, i_min, i_max, av)
    _, _, y_wetting = average_data_func(date_num, wetting_series, i_min, i_max, av)
    _, _, y_drain = average_data_func(date_num, drain_series, i_min, i_max, av)
    _, _, y_rain = average_data_func(date_num, rain_only_series, i_min, i_max, av)

    # Means for legend (convert to µm/hr for text as in MATLAB: *1000)
    r_valid_w = np.where(~np.isnan(np.asarray(y_evap).squeeze()))[0]

    def _mean_on_rw(y: np.ndarray) -> float:
        yv = np.asarray(y).squeeze()
        if r_valid_w.size == 0:
            return float("nan")
        return float(np.nanmean(yv[r_valid_w]))

    mean_evap = _mean_on_rw(y_evap)
    mean_rain_drain = _mean_on_rw(y_rain_drain)
    mean_freeze = _mean_on_rw(y_freeze)
    mean_spray = _mean_on_rw(y_spray)
    mean_wetting = _mean_on_rw(y_wetting)
    mean_melt = _mean_on_rw(y_melt)
    mean_rain = _mean_on_rw(y_rain)
    mean_drain = _mean_on_rw(y_drain)

    dt_x_w = matlab_datenum_to_datetime_array(xplot_w)
    ax2.set_title(f"{title_str}: Surface water balance")
    ax2.plot(
        dt_x_w,
        np.asarray(y_evap).squeeze(),
        "b-",
        linewidth=1,
        label="Evaporation/condensation",
    )
    ax2.plot(
        dt_x_w,
        np.asarray(y_rain_drain).squeeze(),
        "r-",
        linewidth=1,
        label="Melt+Rain-Drainage",
    )
    ax2.plot(dt_x_w, np.asarray(y_melt).squeeze(), "g-", linewidth=1, label="Melt")
    ax2.plot(
        dt_x_w, np.asarray(y_freeze).squeeze(), "c-", linewidth=1, label="Freezing"
    )
    ax2.plot(dt_x_w, np.asarray(y_spray).squeeze(), "m-", linewidth=1, label="Spray")
    ax2.plot(
        dt_x_w, np.asarray(y_wetting).squeeze(), "y-", linewidth=1, label="Wetting"
    )
    ax2.set_ylabel("Rates (mm/hr)")
    format_time_axis(ax2, dt_x_w, shared.av[0], day_tick_limit=day_tick_limit)
    legend_text2 = [
        f"Evap/condens = {mean_evap * 1000:4.2f} (µm/hr)",
        f"Rain-drainage = {mean_rain_drain * 1000:4.2f} (µm/hr)",
        f"Melt = {mean_melt * 1000:4.2f} (µm/hr)",
        f"Freezing = {mean_freeze * 1000:4.2f} (µm/hr)",
        f"Spray = {mean_spray * 1000:4.2f} (µm/hr)",
        f"Wetting = {mean_wetting * 1000:4.2f} (µm/hr)",
    ]
    ax2.legend(legend_text2, loc="upper left")
    if len(dt_x_w) > 0:
        ax2.set_xlim(dt_x_w[0], dt_x_w[-1])
    ax2.autoscale(enable=True, axis="y", tight=True)

    plt.tight_layout()

    # ---------------- Optional textual summaries (post-plot prints) ----------------
    if shared.print_result:
        # Energy budget (W/m^2)
        print("Energy budget (W/m^2)")
        print(
            "\t".join(
                [
                    f"{'Net shortwave':<18}",
                    f"{'Net longwave':<18}",
                    f"{'Net radiation':<18}",
                    f"{'Sensible heat':<18}",
                    f"{'Latent heat':<18}",
                    f"{'Traffic heat':<18}",
                    f"{'Surface heat':<18}",
                    f"{'Sub-surface heat':<18}",
                ]
            )
        )
        print(
            "\t".join(
                [
                    f"{mean_short_net:<18.2f}",
                    f"{mean_long_net:<18.2f}",
                    f"{(mean_short_net + mean_long_net):<18.2f}",
                    f"{mean_H:<18.2f}",
                    f"{mean_L:<18.2f}",
                    f"{mean_H_traffic:<18.2f}",
                    f"{mean_G:<18.2f}",
                    f"{mean_G_sub:<18.2f}",
                ]
            )
        )

        # Moisture budget (mm/day)
        print("Moisture budget (mm/day)")
        print(
            "\t".join(
                [
                    f"{'Rain':<18}",
                    f"{'Drainage':<18}",
                    f"{'Rain-drainage':<18}",
                    f"{'Evaporation':<18}",
                    f"{'Melt':<18}",
                    f"{'Freezing':<18}",
                    f"{'Spray':<18}",
                    f"{'Wetting':<18}",
                ]
            )
        )
        print(
            "\t".join(
                [
                    f"{mean_rain * 24:<18.4f}",
                    f"{mean_drain * 24:<18.4f}",
                    f"{mean_rain_drain * 24:<18.4f}",
                    f"{mean_evap * 24:<18.4f}",
                    f"{mean_melt * 24:<18.4f}",
                    f"{mean_freeze * 24:<18.4f}",
                    f"{mean_spray * 24:<18.4f}",
                    f"{mean_wetting * 24:<18.4f}",
                ]
            )
        )

    if shared.save_plots:
        plot_file_name = generate_matlab_style_filename(
            title_str=paths.title_str,
            plot_type_flag=shared.av[0],
            figure_number=6,  # Energy and moisture balance is figure 6
            plot_name="Energy_moisture_balance",
            date_num=shared.date_num,
            min_time=shared.i_min,
            max_time=shared.i_max
        )
        plt.savefig(os.path.join(paths.path_outputfig, plot_file_name))