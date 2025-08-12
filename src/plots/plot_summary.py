from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import constants
from config_classes.model_parameters import model_parameters as ModelParameters
from config_classes.model_flags import model_flags as ModelFlags
from config_classes.model_file_paths import model_file_paths as ModelFilePaths
from input_classes.converted_data import converted_data as ConvertedData
from input_classes.input_metadata import input_metadata as InputMetadata
from input_classes.input_airquality import input_airquality as InputAirquality
from initialise.road_dust_initialise_time import time_config as TimeConfig
from initialise.road_dust_initialise_variables import (
    model_variables as ModelVariables,
)
from functions.average_data_func import average_data_func as Average_data_func


def _track_weighted_sum(array: np.ndarray, f_track: np.ndarray) -> np.ndarray:
    """
    Weighted sum over tracks for arrays with track dimension at index -2.
    array shape examples:
      - C_data: [source, size, process, time, track, road]
      - E_road_data: [source, size, process, time, track, road]
      - M_road_data: [source, size, time, track, road]
    """
    # Ensure f_track is broadcastable over a trailing track dimension
    # After slicing for a specific road, the track dimension is the last axis
    # Examples after ro-slice:
    #   C_data: [source, size, process, time, track]
    #   E_road_data: [source, size, process, time, track]
    #   M_road_data: [source, size, time, track]
    #   g_road_data: [moisture, time, track]
    weights_last = np.asarray(f_track, dtype=float)
    # Build a shape of ones except last axis equals num_track
    shape = [1] * array.ndim
    shape[-1] = weights_last.shape[0]
    weights_last = weights_last.reshape(shape)
    return np.nansum(array * weights_last, axis=-1)


def _prepare_series(
    *,
    date_num: np.ndarray,
    series: np.ndarray,
    time_config: TimeConfig,
    av: Tuple[int, ...],
) -> Tuple[list, np.ndarray, np.ndarray]:
    i_min = time_config.min_time
    i_max = time_config.max_time
    return Average_data_func(date_num, series, i_min, i_max, list(av))


def _matlab_datenum_to_datetime_array(nums: np.ndarray) -> list:
    """Convert MATLAB datenums to Python datetimes for axis plotting."""
    # MATLAB datenum 1 = year 0000-01-01, Python has no year 0; Average_data_func uses year 1 base.
    # We'll mirror that logic.
    from datetime import datetime, timedelta

    def convert_one(d: float):
        matlab_epoch = datetime(1, 1, 1)
        return matlab_epoch + timedelta(days=float(d) - 1.0)

    return [convert_one(d) for d in np.asarray(nums).ravel()]


def _format_time_axis(
    ax: plt.Axes, dt_x: list, av_index: int, day_tick_limit: int = 150
):
    if av_index in (3, 5):
        return  # handled separately using string ticks
    if not dt_x:
        return
    # Ensure sensible limits
    ax.set_xlim(dt_x[0], dt_x[-1])
    span_days = (dt_x[-1] - dt_x[0]).days if len(dt_x) >= 2 else 0
    if span_days > day_tick_limit:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=1))
    else:
        step = max(1, span_days // 12 or 1)
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=step))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    for label in ax.get_xticklabels():
        label.set_rotation(0)
        label.set_ha("center")


def plot_summary(
    *,
    time_config: TimeConfig,
    converted_data: ConvertedData,
    metadata: InputMetadata,
    airquality_data: InputAirquality,
    model_parameters: ModelParameters,
    model_flags: ModelFlags,
    model_variables: ModelVariables,
    paths: Optional[ModelFilePaths] = None,
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

    M_road_data_temp = _track_weighted_sum(
        model_variables.M_road_data[:, :, :, :, ro], np.array(model_parameters.f_track)
    )
    # Other temporary arrays can be computed similarly if/when needed by additional panels
    # E_road_data_temp = _track_weighted_sum(model_variables.E_road_data[:, :, :, :, :, ro], np.array(model_parameters.f_track))
    # g_road_data_temp = _track_weighted_sum(model_variables.g_road_data[:, :, :, ro], np.array(model_parameters.f_track))
    # activity_data_temp = converted_data.activity_data[:, :, ro]

    # Observations
    PM_obs_net = airquality_data.PM_obs_net

    # Date axis for this road
    date_num = converted_data.date_data[constants.datenum_index, :, ro]

    # Subplot 1: Concentrations
    # Handle NaNs analogous to MATLAB nodata masking
    mask_invalid = (PM_obs_net[x, :]) != metadata.nodata
    C_data_temp2 = C_data_temp.copy()
    C_data_temp2[:, x, :, mask_invalid] = np.nan
    PM_obs_net_temp = PM_obs_net.copy()
    PM_obs_net_temp[x, mask_invalid] = np.nan

    # Build series
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
            C_data_temp2[constants.wear_index, x, constants.C_total_index, :], axis=0
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

    fig, axes = plt.subplots(4, 1, figsize=(10, 8))
    ax1 = axes[0]
    title_str = paths.title_str if paths and paths.title_str else "NORTRIP"
    pm_text = (
        "PM_1_0"
        if x == constants.pm_10
        else ("PM_2_._5" if x == constants.pm_25 else "PM")
    )
    ax1.set_title(f"{title_str}: {pm_text}")
    av_index = model_flags.plot_type_flag
    if av_index in (3, 5):
        # Use string ticks for daily cycle or weekday plots
        ax1.plot(xplot, y_obs, "k--", linewidth=1, label="Observed")
        ax1.plot(xplot, y_salt_na, "g-", linewidth=1, label="Modelled salt(na)")
        ax1.plot(xplot, y_salt_2, "g--", linewidth=1, label="Modelled salt(2)")
        ax1.plot(xplot, y_wear, "r:", linewidth=1, label="Modelled wear")
        ax1.plot(xplot, y_sand, "m--", linewidth=1, label="Modelled sand")
        ax1.plot(xplot, y_total, "b-", linewidth=1, label="Modelled total")
        ax1.set_xticks(xplot)
        ax1.set_xticklabels(x_str1)
    else:
        dt_x = _matlab_datenum_to_datetime_array(xplot)
        ax1.plot(dt_x, y_obs, "k--", linewidth=1, label="Observed")
        ax1.plot(dt_x, y_salt_na, "g-", linewidth=1, label="Modelled salt(na)")
        ax1.plot(dt_x, y_salt_2, "g--", linewidth=1, label="Modelled salt(2)")
        ax1.plot(dt_x, y_wear, "r:", linewidth=1, label="Modelled wear")
        ax1.plot(dt_x, y_sand, "m--", linewidth=1, label="Modelled sand")
        ax1.plot(dt_x, y_total, "b-", linewidth=1, label="Modelled total")
        _format_time_axis(ax1, dt_x, av_index)
    ax1.set_ylabel(f"{pm_text} concentration (µg/m^3)")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.2)

    # Subplot 2: Mass loading
    # Convert linear mass per road width (mg/m) to areal mass (g/m^2)
    # MATLAB uses: b_factor = 1/1000/b_road_lanes
    road_width_m = 0.0
    if (
        hasattr(metadata, "b_road_lanes")
        and metadata.b_road_lanes
        and metadata.b_road_lanes > 0
    ):
        road_width_m = metadata.b_road_lanes
    elif (
        hasattr(metadata, "n_lanes")
        and hasattr(metadata, "b_lane")
        and metadata.n_lanes > 0
        and metadata.b_lane > 0
    ):
        road_width_m = metadata.n_lanes * metadata.b_lane
    elif hasattr(metadata, "b_road") and metadata.b_road > 0:
        road_width_m = metadata.b_road
    else:
        road_width_m = 1.0
    b_factor = 1.0 / 1000.0 / max(road_width_m, 1e-9)
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
    ax2 = axes[1]
    if av_index in (3, 5):
        ax2.plot(xplot2, y_dust, "k-", linewidth=1, label="Suspendable dust")
        ax2.plot(xplot2, y_salt_na_m, "g-", linewidth=1, label="Salt(na)")
        ax2.plot(xplot2, y_salt_2_m, "g--", linewidth=1, label="Salt(2)")
        ax2.plot(xplot2, y_sand_m, "r--", linewidth=1, label="Suspendable sand")
        ax2.set_xticks(xplot2)
        ax2.set_xticklabels(x_str2)
    else:
        dt_x2 = _matlab_datenum_to_datetime_array(xplot2)
        ax2.plot(dt_x2, y_dust, "k-", linewidth=1, label="Suspendable dust")
        ax2.plot(dt_x2, y_salt_na_m, "g-", linewidth=1, label="Salt(na)")
        ax2.plot(dt_x2, y_salt_2_m, "g--", linewidth=1, label="Salt(2)")
        ax2.plot(dt_x2, y_sand_m, "r--", linewidth=1, label="Suspendable sand")
        _format_time_axis(ax2, dt_x2, av_index)
    ax2.set_ylabel("Mass loading (g/m^2)")
    ax2.legend(loc="upper left", fontsize=8)
    ax2.grid(True, alpha=0.2)

    # Keep placeholders for other panels to match 4x1 grid used later
    axes[2].axis("off")
    axes[3].axis("off")
    axes[3].set_xlabel("Date")

    fig.tight_layout()
    plt.show()
