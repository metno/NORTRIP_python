import numpy as np
import matplotlib.pyplot as plt
import os

from config_classes import model_file_paths
from functions.average_data_func import average_data_func
from .shared_plot_data import shared_plot_data
from functions import rmse_func as rmse
import constants
from .helpers import generate_plot_filename


def _compute_scatter_inputs(
    *,
    C_sum: np.ndarray,
    PM_obs_net: np.ndarray,
    f_conc: np.ndarray,
    nodata: float,
    date_num: np.ndarray,
    i_min: int,
    i_max: int,
    av: list[int],
    size_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Prepare model and observed time series averaged per av for a size index.

    Mirrors MATLAB masking and averaging for scatter/QQ plots.
    """
    n_date = date_num.shape[0]

    # Fresh copies for this size so masking does not bleed across sizes
    C_local = C_sum.astype(float).copy()
    C_local[C_local == nodata] = np.nan

    PM_local = PM_obs_net.astype(float).copy()
    PM_local[PM_local == nodata] = np.nan

    # Mask times where either dispersion factor or observations are nodata for this size
    invalid_time = (f_conc[:n_date] == nodata) | np.isnan(PM_local[size_index, :n_date])
    if np.any(invalid_time):
        C_local[:, :, :, invalid_time] = np.nan
        PM_local[size_index, invalid_time] = np.nan

    # Model: sum across sources for total concentration process
    model_series = np.sum(
        C_local[
            constants.all_source_index, size_index, constants.C_total_index, :n_date
        ],
        axis=0,
    )
    obs_series = PM_local[size_index, :n_date]

    # Average to requested resolution
    _, _, y_model = average_data_func(date_num, model_series, i_min, i_max, av)
    _, _, y_obs = average_data_func(date_num, obs_series, i_min, i_max, av)

    return np.asarray(y_model).squeeze(), np.asarray(y_obs).squeeze()


def plot_scatter_qq(shared: shared_plot_data, paths: model_file_paths) -> None:
    """
    Plot figure 11: Scatter and QQ plots for PM10 and PM2.5.
    """
    # Shorthands
    date_num = shared.date_num
    i_min, i_max = shared.i_min, shared.i_max
    av = list(shared.av)
    nodata = shared.nodata
    title_str = getattr(paths, "title_str", "") or "Scatter plots"

    # Prepare inputs for PM10 and PM2.5
    y_mod_10, y_obs_10 = _compute_scatter_inputs(
        C_sum=shared.C_data_sum_tracks,
        PM_obs_net=shared.PM_obs_net,
        f_conc=shared.f_conc,
        nodata=nodata,
        date_num=date_num,
        i_min=i_min,
        i_max=i_max,
        av=av,
        size_index=constants.pm_10,
    )

    y_mod_25, y_obs_25 = _compute_scatter_inputs(
        C_sum=shared.C_data_sum_tracks,
        PM_obs_net=shared.PM_obs_net,
        f_conc=shared.f_conc,
        nodata=nodata,
        date_num=date_num,
        i_min=i_min,
        i_max=i_max,
        av=av,
        size_index=constants.pm_25,
    )

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    fig.subplots_adjust(hspace=0.38, wspace=0.38)
    try:
        fig.canvas.manager.set_window_title("Figure 9: Scatter/QQ plots")  # type: ignore
    except Exception:
        pass

    # --- Panel 1: Scatter PM10 ---
    ax1 = axes[0, 0]
    if int(shared.EP_emis_available) == 1:
        ax1.set_title(f"{title_str}: Scatter plot total net PM10", fontsize=7)
    else:
        ax1.set_title(f"{title_str}: Scatter plot net non-exhaust PM10", fontsize=7)

    # Valid pairs
    r10 = np.where(np.isfinite(y_mod_10) & np.isfinite(y_obs_10))[0]
    if r10.size > 0:
        ax1.scatter(
            y_mod_10[r10], y_obs_10[r10], facecolor="none", edgecolor="blue", s=10
        )

    ax1.set_ylabel("PM10 observed concentration (µg/m³)")
    ax1.set_xlabel("PM10 modelled concentration (µg/m³)")
    ax1.grid(True)

    # Axis limits 0..max of both series among valid points
    if r10.size > 0:
        max_plot_10 = float(
            np.nanmax([np.nanmax(y_mod_10[r10]), np.nanmax(y_obs_10[r10])])
        )
        if np.isfinite(max_plot_10):
            ax1.set_xlim(0, max_plot_10 * 1.02)
            ax1.set_ylim(0, max_plot_10)

    # Basic statistics and regression line
    if r10.size > 1:
        # r^2
        with np.errstate(invalid="ignore"):
            Rcor = np.corrcoef(y_mod_10[r10], y_obs_10[r10])
            r_sq_pm10 = float(Rcor[0, 1] ** 2) if Rcor.shape == (2, 2) else np.nan
        # RMSE
        rmse_pm10 = float(rmse(y_mod_10[r10], y_obs_10[r10]))
        # Means
        mean_obs_pm10 = float(np.nanmean(y_obs_10[r10]))
        mean_mod_pm10 = float(np.nanmean(y_mod_10[r10]))
        # Regression y = a1*x + a0
        a1, a0 = np.polyfit(y_mod_10[r10], y_obs_10[r10], 1)

        ax1.text(0.05, 0.95, f"r²  = {r_sq_pm10:4.2f}", transform=ax1.transAxes)
        ax1.text(
            0.05, 0.88, f"RMSE = {rmse_pm10:4.1f} (µg/m³)", transform=ax1.transAxes
        )
        ax1.text(
            0.05, 0.81, f"OBS  = {mean_obs_pm10:4.1f} (µg/m³)", transform=ax1.transAxes
        )
        ax1.text(
            0.05, 0.74, f"MOD  = {mean_mod_pm10:4.1f} (µg/m³)", transform=ax1.transAxes
        )

        ax1.text(0.55, 0.10, f"a_0  = {a0:4.1f} (µg/m³)", transform=ax1.transAxes)
        ax1.text(0.55, 0.03, f"a_1  = {a1:4.2f}", transform=ax1.transAxes)

        xmin = float(np.nanmin(y_mod_10[r10]))
        xmax = float(np.nanmax(y_mod_10[r10]))
        ax1.plot(
            [xmin, xmax], [a0 + a1 * xmin, a0 + a1 * xmax], "-", color=(0.5, 0.5, 0.5)
        )

    # --- Panel 2: QQ PM10 ---
    ax2 = axes[0, 1]
    if int(shared.EP_emis_available) == 1:
        ax2.set_title(f"{title_str}: QQ plot total net PM10", fontsize=7)
    else:
        ax2.set_title(f"{title_str}: QQ plot net non-exhaust PM10", fontsize=7)

    r10b = np.where(np.isfinite(y_mod_10) & np.isfinite(y_obs_10))[0]
    if r10b.size > 0:
        y1_sort = np.sort(y_mod_10[r10b])
        y2_sort = np.sort(y_obs_10[r10b])
        ax2.scatter(y1_sort, y2_sort, facecolor="none", edgecolor="red", s=10)
        max_plot = float(np.nanmax([np.nanmax(y1_sort), np.nanmax(y2_sort)]))
        if np.isfinite(max_plot):
            ax2.set_xlim(0, max_plot * 1.02)
            ax2.set_ylim(0, max_plot)
    ax2.set_ylabel("PM10 observed concentration (µg/m³)")
    ax2.set_xlabel("PM10 modelled concentration (µg/m³)")
    ax2.grid(True)

    # Daily averaging extras
    if av[0] == 2 and r10b.size > 0:
        # Recompute sorted arrays in this scope to satisfy static analysis
        y1_sort = np.sort(y_mod_10[r10b])
        y2_sort = np.sort(y_obs_10[r10b])
        high36_mod = float(y1_sort[-36]) if y1_sort.size > 36 else 0.0
        high36_obs = float(y2_sort[-36]) if y2_sort.size > 36 else 0.0
        ex50_mod = int(np.sum(y_mod_10 > 50))
        ex50_obs = int(np.sum(y_obs_10 > 50))
        ax2.text(
            0.05,
            0.95,
            f"36th highest MOD  = {high36_mod:4.1f} (µg/m³)",
            transform=ax2.transAxes,
        )
        ax2.text(
            0.05,
            0.88,
            f"36th highest OBS  = {high36_obs:4.1f} (µg/m³)",
            transform=ax2.transAxes,
        )
        ax2.text(
            0.05, 0.81, f"Days > 50 µg/m³ MOD= {ex50_mod:4.0f}", transform=ax2.transAxes
        )
        ax2.text(
            0.05,
            0.74,
            f"Days > 50 µg/m³ OBS  = {ex50_obs:4.0f}",
            transform=ax2.transAxes,
        )

    # --- Panel 3: Scatter PM2.5 ---
    ax3 = axes[1, 0]
    if int(shared.EP_emis_available) == 1:
        ax3.set_title(f"{title_str}: Scatter plot total net PM2.5", fontsize=7)
    else:
        ax3.set_title(f"{title_str}: Scatter plot net non-exhaust PM2.5", fontsize=7)

    r25 = np.where(np.isfinite(y_mod_25) & np.isfinite(y_obs_25))[0]
    if r25.size > 1:
        ax3.scatter(
            y_mod_25[r25], y_obs_25[r25], facecolor="none", edgecolor="blue", s=10
        )

    ax3.set_ylabel("PM2.5 observed concentration (µg/m³)")
    ax3.set_xlabel("PM2.5 modelled concentration (µg/m³)")
    ax3.grid(True)

    if r25.size > 0:
        max_plot_25 = float(
            np.nanmax([np.nanmax(y_mod_25[r25]), np.nanmax(y_obs_25[r25])])
        )
        if np.isfinite(max_plot_25):
            ax3.set_xlim(0, max_plot_25 * 1.02)
            ax3.set_ylim(0, max_plot_25)

    if r25.size > 1:
        with np.errstate(invalid="ignore"):
            Rcor25 = np.corrcoef(y_mod_25[r25], y_obs_25[r25])
            r_sq_pm25 = float(Rcor25[0, 1] ** 2) if Rcor25.shape == (2, 2) else np.nan
        rmse_pm25 = float(rmse(y_mod_25[r25], y_obs_25[r25]))
        mean_obs_pm25 = float(np.nanmean(y_obs_25[r25]))
        mean_mod_pm25 = float(np.nanmean(y_mod_25[r25]))
        a1_25, a0_25 = np.polyfit(y_mod_25[r25], y_obs_25[r25], 1)

        ax3.text(0.05, 0.95, f"r²  = {r_sq_pm25:4.2f}", transform=ax3.transAxes)
        ax3.text(
            0.05, 0.88, f"RMSE = {rmse_pm25:4.1f} (µg/m³)", transform=ax3.transAxes
        )
        ax3.text(
            0.05, 0.81, f"OBS  = {mean_obs_pm25:4.1f} (µg/m³)", transform=ax3.transAxes
        )
        ax3.text(
            0.05, 0.74, f"MOD  = {mean_mod_pm25:4.1f} (µg/m³)", transform=ax3.transAxes
        )

        ax3.text(0.55, 0.10, f"a_0  = {a0_25:4.1f} (µg/m³)", transform=ax3.transAxes)
        ax3.text(0.55, 0.03, f"a_1  = {a1_25:4.2f}", transform=ax3.transAxes)

        xmin25 = float(np.nanmin(y_mod_25[r25]))
        xmax25 = float(np.nanmax(y_mod_25[r25]))
        ax3.plot(
            [xmin25, xmax25],
            [a0_25 + a1_25 * xmin25, a0_25 + a1_25 * xmax25],
            "-",
            color=(0.5, 0.5, 0.5),
        )

    # --- Panel 4: QQ PM2.5 ---
    ax4 = axes[1, 1]
    if int(shared.EP_emis_available) == 1:
        ax4.set_title(f"{title_str}: QQ plot PM2.5 + EP", fontsize=7)
    else:
        ax4.set_title(f"{title_str}: QQ plot PM2.5", fontsize=7)

    r25b = np.where(np.isfinite(y_mod_25) & np.isfinite(y_obs_25))[0]
    if r25b.size > 0:
        y1_25_sort = np.sort(y_mod_25[r25b])
        y2_25_sort = np.sort(y_obs_25[r25b])
        ax4.scatter(y1_25_sort, y2_25_sort, facecolor="none", edgecolor="red", s=10)
        max_plot_25 = float(np.nanmax([np.nanmax(y1_25_sort), np.nanmax(y2_25_sort)]))
        if np.isfinite(max_plot_25):
            ax4.set_xlim(0, max_plot_25 * 1.02)
            ax4.set_ylim(0, max_plot_25)
    ax4.set_ylabel("PM2.5 observed concentration (µg/m³)")
    ax4.set_xlabel("PM2.5 modelled concentration (µg/m³)")
    ax4.grid(True)

    if shared.save_plots:
        plot_file_name = generate_plot_filename(
            title_str=paths.title_str,
            plot_type_flag=shared.av[0],
            figure_number=9,  # Scatter/QQ plots is figure 9
            plot_name="Scatter_QQ",
            date_num=shared.date_num,
            min_time=shared.i_min,
            max_time=shared.i_max,
        )
        plt.savefig(os.path.join(paths.path_outputfig, plot_file_name))
