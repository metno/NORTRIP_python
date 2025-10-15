from __future__ import annotations

from typing import Tuple, Any
from datetime import datetime, timedelta

import numpy as np
import matplotlib.dates as mdates
from matplotlib.axes import Axes

import constants
from functions import average_data_func, datenum_to_datetime


def unix_timestamp_to_datetime_array(nums: np.ndarray) -> np.ndarray:
    """Convert Unix timestamps to Python datetimes for axis plotting."""
    return np.array([datenum_to_datetime(d) for d in np.asarray(nums).ravel()])


def prepare_series(
    *,
    date_num: np.ndarray,
    series: np.ndarray,
    time_config: Any,
    av: Tuple[int, ...],
) -> Tuple[list, np.ndarray, np.ndarray]:
    i_min = time_config.min_time
    i_max = time_config.max_time
    return average_data_func(date_num, series, i_min, i_max, list(av))


def format_time_axis(
    ax: Axes, dt_x: np.ndarray, av_index: int, day_tick_limit: int = 150
) -> None:
    if not dt_x.any():
        return

    if av_index in (3, 5):
        ax.set_xticks(dt_x)
        # Set appropriate x-limits for these modes
        ax.set_xlim(dt_x[0], dt_x[-1])
        if av_index == 5:  # Weekly cycle: set weekday labels
            weekday_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            ax.set_xticklabels(weekday_names)
        return

    ax.set_xlim(dt_x[0], dt_x[-1])
    span_days = (dt_x[-1] - dt_x[0]).days if dt_x.size >= 2 else 0
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
        label.set_horizontalalignment("center")


def mask_nodata(arr: np.ndarray, nodata: float) -> np.ndarray:
    """Return a float copy of arr with nodata values replaced by NaN.

    Centralized helper for plotting modules.
    """
    a = np.asarray(arr, dtype=float).copy()
    a[a == nodata] = np.nan
    return a


def generate_plot_filename(
    title_str: str,
    plot_type_flag: int,
    figure_number: int,
    plot_name: str,
    date_num: np.ndarray,
    min_time: int,
    max_time: int,
) -> str:
    """
    Generate filename for plot saving.

    Format: title_av_fig_i_plotname_startdate-enddate.png
    """
    av_str = constants.av_str[plot_type_flag - 1]

    # Convert MATLAB datenums to Python datetimes and format as yyyymmdd
    start_date_datenum = date_num[min_time]
    end_date_datenum = date_num[max_time]

    start_datetime = datenum_to_datetime(start_date_datenum)
    end_datetime = datenum_to_datetime(end_date_datenum)

    file_start_date = start_datetime.strftime("%Y%m%d")
    file_end_date = end_datetime.strftime("%Y%m%d")

    # Construct filename following MATLAB pattern
    filename = f"{title_str}_{av_str}_fig_{figure_number}_{plot_name}_{file_start_date}-{file_end_date}.png"

    return filename
