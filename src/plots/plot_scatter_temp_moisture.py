import numpy as np
import matplotlib.pyplot as plt
import os

import constants
from config_classes import model_file_paths
from functions.average_data_func import average_data_func
from .shared_plot_data import shared_plot_data
from .helpers import generate_matlab_style_filename

def plot_scatter_temp_moisture(
    shared: shared_plot_data, paths: model_file_paths
) -> None:
    """
    Figure 14: Scatter temperature and moisture (MATLAB figure 14)

    Panels (assuming which_moisture_plot == 3 to include temperature error scatter):
      1) Scatter of ΔT_s (T_s - T_a): modelled vs observed
      2) Scatter of surface temperature error (T_s_model - T_s_obs)

    If road temperature observations are not available, this figure is skipped.
    Moisture scatter/frequency variants (which_moisture_plot 1/2) are not ported yet; we
    can extend if needed once exact rules are confirmed.
    """

    if not shared.road_temperature_obs_available:
        # No figure when observations are unavailable
        return

    date_num = shared.date_num
    i_min, i_max = shared.i_min, shared.i_max
    av = list(shared.av)
    nodata = shared.nodata

    # Local copies
    meteo = np.asarray(shared.meteo_data_ro, dtype=float).copy()
    roadm = np.asarray(shared.road_meteo_weighted, dtype=float).copy()
    meteo[meteo == nodata] = np.nan
    roadm[roadm == nodata] = np.nan

    # Observed road temperature is stored in road_meteo_weighted at index road_temperature_obs_index
    T_a_full = meteo[constants.T_a_index, :]
    T_s_model_full = roadm[constants.T_s_index, :]
    T_s_obs_full = roadm[constants.road_temperature_obs_index, :]

    # Restrict to selected window
    T_a = T_a_full[i_min : i_max + 1]
    T_s_model = T_s_model_full[i_min : i_max + 1]
    T_s_obs = T_s_obs_full[i_min : i_max + 1]
    date_num_window = date_num[i_min : i_max + 1]

    # Mask invalid values as in MATLAB
    invalid = np.isnan(T_a) | np.isnan(T_s_model) | np.isnan(T_s_obs)
    T_a[invalid] = np.nan
    T_s_model[invalid] = np.nan
    T_s_obs[invalid] = np.nan

    # Temperature difference mode
    T_diff_mod = T_s_model - T_a
    T_diff_obs = T_s_obs - T_a

    # Average with same function used elsewhere (over the window)
    _xs, _xp, y_mod_diff = average_data_func(
        date_num_window, T_diff_mod, 0, int(T_diff_mod.shape[0] - 1), av
    )
    _xs2, _xp2, y_obs_diff = average_data_func(
        date_num_window, T_diff_obs, 0, int(T_diff_obs.shape[0] - 1), av
    )

    # Mask to finite pairs
    y_mod_diff = np.asarray(y_mod_diff).squeeze()
    y_obs_diff = np.asarray(y_obs_diff).squeeze()
    valid = np.isfinite(y_mod_diff) & np.isfinite(y_obs_diff)

    # Prepare figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    fig.subplots_adjust(wspace=0.35)
    try:
        fig.canvas.manager.set_window_title(  # type: ignore
            "Figure 14: Scatter temperature and moisture"
        )
    except Exception:
        pass

    title_str = paths.title_str or "Scatter temperature and moisture"
    # Panel 1: ΔT scatter
    ax1.set_title(f"{title_str}: temperature difference")
    if np.any(valid):
        xm = y_mod_diff[valid]
        ym = y_obs_diff[valid]
        ax1.scatter(xm, ym, facecolor="none", edgecolor="black", s=10)
        maxi = float(np.nanmax([np.nanmax(xm), np.nanmax(ym)]))
        mini = float(np.nanmin([np.nanmin(xm), np.nanmin(ym)]))
        if np.isfinite(maxi) and np.isfinite(mini):
            ax1.set_xlim(mini, maxi)
            ax1.set_ylim(mini, maxi)
        # Stats and regression
        if xm.size >= 2:
            try:
                R = np.corrcoef(xm, ym)
                r_sq = float(R[0, 1] ** 2)
            except Exception:
                r_sq = np.nan
            rmse = float(np.sqrt(np.nanmean((xm - ym) ** 2)))
            mae = float(np.nanmean(np.abs(xm - ym)))
            mean_obs = float(np.nanmean(ym))
            mean_mod = float(np.nanmean(xm))
            try:
                a1, a0 = np.polyfit(xm, ym, 1)
            except Exception:
                a1, a0 = np.nan, np.nan
            ax1.text(
                0.05, 0.95, f"r²  = {r_sq:4.2f}", transform=ax1.transAxes, va="top"
            )
            ax1.text(
                0.05,
                0.88,
                f"RMSE = {rmse:4.2f} (°C)",
                transform=ax1.transAxes,
                va="top",
            )
            ax1.text(
                0.05,
                0.81,
                f"OBS MEAN = {mean_obs:4.2f} (°C)",
                transform=ax1.transAxes,
                va="top",
            )
            ax1.text(
                0.05,
                0.74,
                f"MOD MEAN = {mean_mod:4.2f} (°C)",
                transform=ax1.transAxes,
                va="top",
            )
            # Regression coefficients (to match MATLAB annotation positions)
            ax1.text(
                0.80,
                0.2 - 0.1,
                rf"$a_0$  = {a0:4.2f} (°C)",
                transform=ax1.transAxes,
            )
            ax1.text(
                0.80,
                0.13 - 0.1,
                rf"$a_1$  = {a1:4.2f}",
                transform=ax1.transAxes,
            )
            if np.isfinite(a0) and np.isfinite(a1):
                xmin = float(np.nanmin(xm))
                xmax = float(np.nanmax(xm))
                ax1.plot(
                    [xmin, xmax],
                    [a0 + a1 * xmin, a0 + a1 * xmax],
                    "-",
                    color=(0.5, 0.5, 0.5),
                )
            # Console table (left-aligned) mirroring MATLAB fid_print
            if shared.print_result:
                print(
                    f"{'Mean observed':<20}\t{'Mean modelled':<20}\t{'RMSE':<20}\t{'MAE':<20}\t{'Corr (r^2)':<20}\t{'Intercept':<20}\t{'Slope':<20}\t{'Bias':<20}"
                )
                print(
                    f"{mean_obs:<20.3f}\t{mean_mod:<20.3f}\t{rmse:<20.3f}\t{mae:<20.3f}\t{r_sq:<20.3f}\t{a0:<20.3f}\t{a1:<20.3f}\t{(mean_mod - mean_obs):<20.3f}"
                )
    ax1.set_ylabel(r"$\Delta T_s$ observed (°C)")
    ax1.set_xlabel(r"$\Delta T_s$ modelled (°C)")

    # Panel 2: temperature error vs observed T_s
    # Average observed T_s and error across the window
    T_error_mod = T_s_model - T_s_obs
    _xs3, _xp3, y_Ts_obs = average_data_func(
        date_num_window, T_s_obs, 0, int(T_error_mod.shape[0] - 1), av
    )
    _xs4, _xp4, y_T_error = average_data_func(
        date_num_window, T_error_mod, 0, int(T_error_mod.shape[0] - 1), av
    )
    y_Ts_obs = np.asarray(y_Ts_obs).squeeze()
    y_T_error = np.asarray(y_T_error).squeeze()
    valid2 = np.isfinite(y_Ts_obs) & np.isfinite(y_T_error)

    ax2.set_title(f"{title_str}: surface temperature error")
    if np.any(valid2):
        x2 = y_Ts_obs[valid2]
        y2 = y_T_error[valid2]
        ax2.scatter(x2, y2, facecolor="none", edgecolor="blue", s=10)
        x_min = float(np.nanmin(x2))
        x_max = float(np.nanmax(x2))
        y_min = float(np.nanmin(y2))
        y_max = float(np.nanmax(y2))
        if np.isfinite(x_min) and np.isfinite(x_max):
            ax2.set_xlim(x_min, x_max)
        if np.isfinite(y_min) and np.isfinite(y_max):
            ax2.set_ylim(y_min, y_max)
        # Stats and regression
        if x2.size >= 2:
            try:
                R2 = np.corrcoef(x2, y2)
                r_sq_T = float(R2[0, 1] ** 2)
            except Exception:
                r_sq_T = np.nan
            # rmse_T = float(np.sqrt(np.nanmean((x2 - y2) ** 2)))
            mae_T = float(np.nanmean(np.abs(y2)))
            mean_error_T = float(np.nanmean(y2))
            try:
                a1b, a0b = np.polyfit(x2, y2, 1)
            except Exception:
                a1b, a0b = np.nan, np.nan
            ax2.text(
                0.05, 0.95, f"r²  = {r_sq_T:4.2f}", transform=ax2.transAxes, va="top"
            )
            ax2.text(
                0.05,
                0.88,
                f"MAE = {mae_T:4.2f} (°C)",
                transform=ax2.transAxes,
                va="top",
            )
            ax2.text(
                0.05,
                0.81,
                f"MEAN ERROR = {mean_error_T:4.2f} (°C)",
                transform=ax2.transAxes,
                va="top",
            )
            # Regression coefficients (to match MATLAB annotation positions)
            ax2.text(
                0.80,
                0.2 - 0.1,
                rf"$a_0$  = {a0b:4.2f} (°C)",
                transform=ax2.transAxes,
            )
            ax2.text(
                0.80,
                0.13 - 0.1,
                rf"$a_1$  = {a1b:4.2f}",
                transform=ax2.transAxes,
            )
            if np.isfinite(a0b) and np.isfinite(a1b):
                xmin2 = float(np.nanmin(x2))
                xmax2 = float(np.nanmax(x2))
                ax2.plot(
                    [xmin2, xmax2],
                    [a0b + a1b * xmin2, a0b + a1b * xmax2],
                    "-",
                    color=(0.5, 0.5, 0.5),
                )
    ax2.set_ylabel(r"$T_s$ modelled  - $T_s$ observed (°C)")
    ax2.set_xlabel(r"$T_s$ observed (°C)")

    plt.tight_layout()

    if shared.save_plots:
        plot_file_name = generate_matlab_style_filename(
            title_str=paths.title_str,
            plot_type_flag=shared.av[0],
            figure_number=14,  # Scatter temperature and moisture is figure 14
            plot_name="Scatter_temp_moisture",
            date_num=shared.date_num,
            min_time=shared.i_min,
            max_time=shared.i_max
        )
        plt.savefig(os.path.join(paths.path_outputfig, plot_file_name))
