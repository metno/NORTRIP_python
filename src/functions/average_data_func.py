import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, List, Optional, Union


def average_data_func(
    date_num: np.ndarray,
    val: np.ndarray,
    i_min: int,
    i_max: int,
    index_in: List[int],
    use_max: bool = False,
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """
    Average input data along various time scales.

    Args:
        date_num: Date numbers (MATLAB format)
        val: Values to average
        i_min: Minimum index
        i_max: Maximum index
        index_in: Averaging specification [index, optional_param]
            1: No averaging
            2: Daily means
            3: Daily cycles
            4: 12 hourly means starting at av_start_hour=[6,18]
            5: Weekly cycles
            6: Daily running means
            7: Weekly means
            8: Monthly means
            9: Hourly means
        use_max: If True, use max instead of mean for averaging

    Returns:
        tuple: (av_date_str, av_date_num, av_val) where:
            av_date_str: List of formatted date strings
            av_date_num: Array of averaged date numbers
            av_val: Array of averaged values
    """
    # Treats NaNs as non-valid data
    index = index_in[0]
    index2 = index_in[1] if len(index_in) >= 2 else np.nan

    av_date_str = []
    av_date_num = np.array([])
    av_val = np.array([])

    # Convert MATLAB datenum to Python datetime for processing
    def matlab_datenum_to_datetime(datenum):
        """Convert MATLAB datenum to Python datetime."""
        # MATLAB datenum 1 = January 1, year 0000
        # Python datetime reference is January 1, year 1
        matlab_epoch = datetime(1, 1, 1)
        days_since_epoch = datenum - 1
        return matlab_epoch + timedelta(days=days_since_epoch)

    def datetime_to_matlab_datenum(dt):
        """Convert Python datetime to MATLAB datenum."""
        matlab_epoch = datetime(1, 1, 1)
        delta = dt - matlab_epoch
        return delta.total_seconds() / 86400.0 + 1

    # No averaging
    if index == 1:
        av_date_num = date_num[i_min : i_max + 1]
        av_val = val[i_min : i_max + 1].reshape(-1, 1)
        av_date_str = [
            matlab_datenum_to_datetime(d).strftime("%H:%M %d %b") for d in av_date_num
        ]

    # Hourly means
    elif index == 9:
        min_num = index2 if not np.isnan(index2) else 1

        # Create hourly time grid
        start_date = matlab_datenum_to_datetime(date_num[i_min])
        end_date = matlab_datenum_to_datetime(date_num[i_max])

        current_hour = start_date.replace(minute=0, second=0, microsecond=0)
        hourly_dates = []
        hourly_vals = []

        while current_hour <= end_date + timedelta(hours=1):
            next_hour = current_hour + timedelta(hours=1)

            # Find data within this hour
            hour_mask = np.array(
                [
                    current_hour <= matlab_datenum_to_datetime(d) < next_hour
                    for d in date_num[i_min : i_max + 1]
                ]
            )

            if np.any(hour_mask):
                hour_vals = val[i_min : i_max + 1][hour_mask]
                valid_vals = hour_vals[~np.isnan(hour_vals)]

                if len(valid_vals) >= min_num:
                    if use_max:
                        hourly_vals.append(np.max(valid_vals))
                    else:
                        hourly_vals.append(np.mean(valid_vals))
                else:
                    hourly_vals.append(np.nan)
            else:
                hourly_vals.append(np.nan)

            hourly_dates.append(datetime_to_matlab_datenum(current_hour))
            current_hour = next_hour

        av_date_num = np.array(hourly_dates)
        av_val = np.array(hourly_vals).reshape(-1, 1)
        av_date_str = [
            matlab_datenum_to_datetime(d).strftime("%d %b") for d in av_date_num
        ]

    # Daily means
    elif index == 2:
        min_num = index2 if not np.isnan(index2) else 6

        start_day = int(np.floor(date_num[i_min]))
        end_day = int(np.floor(date_num[i_max]))

        daily_dates = []
        daily_vals = []

        for i_day in range(start_day, end_day + 1):
            # Find indices for this day
            day_mask = np.floor(date_num) == i_day
            day_indices = np.where(day_mask)[0]
            day_indices = day_indices[(day_indices > i_min) & (day_indices < i_max)]

            if len(day_indices) >= 12:
                day_vals = val[day_indices]
                valid_vals = day_vals[~np.isnan(day_vals)]

                if len(valid_vals) > min_num:
                    if use_max:
                        daily_vals.append(np.max(valid_vals))
                    else:
                        daily_vals.append(np.mean(valid_vals))
                else:
                    daily_vals.append(np.nan)
            else:
                daily_vals.append(np.nan)

            daily_dates.append(float(i_day))

        av_date_num = np.array(daily_dates)
        av_val = np.array(daily_vals).reshape(-1, 1)
        av_date_str = [
            matlab_datenum_to_datetime(d).strftime("%d %b") for d in av_date_num
        ]

    # Daily cycle
    elif index == 3:
        date_subset = date_num[i_min : i_max + 1]
        val_subset = val[i_min : i_max + 1]

        hourly_vals = []
        hourly_dates = []

        for j in range(24):
            # Find all data for this hour
            hour_mask = np.array(
                [matlab_datenum_to_datetime(d).hour == j for d in date_subset]
            )

            if np.any(hour_mask):
                hour_vals = val_subset[hour_mask]
                valid_vals = hour_vals[~np.isnan(hour_vals)]

                if len(valid_vals) > 0:
                    hourly_vals.append(np.mean(valid_vals))
                else:
                    hourly_vals.append(np.nan)
            else:
                hourly_vals.append(np.nan)

            hourly_dates.append(j)

        av_date_num = np.array(hourly_dates)
        av_val = np.array(hourly_vals).reshape(-1, 1)
        av_date_str = [f"{j:02d}" for j in range(24)]

    # 12 hourly means
    elif index == 4:
        av_start_hour = [6, 18]
        if len(index_in) >= 3:
            av_start_hour = [index_in[1], index_in[2]]

        date_subset = date_num[i_min : i_max + 1]
        val_subset = val[i_min : i_max + 1]

        # Find start and end indices for 12-hour periods
        hour_mask_1 = np.array(
            [
                matlab_datenum_to_datetime(d).hour == av_start_hour[0]
                for d in date_subset
            ]
        )
        hour_mask_2 = np.array(
            [
                matlab_datenum_to_datetime(d).hour == av_start_hour[1]
                for d in date_subset
            ]
        )

        start_indices_1 = np.where(hour_mask_1)[0]
        start_indices_2 = np.where(hour_mask_2)[0]

        if len(start_indices_1) > 0 and len(start_indices_2) > 0:
            i_halfday_min = min(start_indices_1[0], start_indices_2[0])
            i_halfday_max = max(start_indices_1[-1], start_indices_2[-1])

            halfday_indices = list(range(i_halfday_min, i_halfday_max - 12, 12))
            halfday_dates = date_subset[halfday_indices]
            halfday_vals = []

            for i, idx in enumerate(halfday_indices):
                period_vals = val_subset[idx : idx + 13]  # 12 hours + 1
                valid_vals = period_vals[~np.isnan(period_vals)]

                if len(valid_vals) > 6:
                    if use_max:
                        halfday_vals.append(np.max(valid_vals))
                    else:
                        halfday_vals.append(np.mean(valid_vals))
                else:
                    halfday_vals.append(np.nan)

            av_date_num = halfday_dates
            av_val = np.array(halfday_vals).reshape(-1, 1)
            av_date_str = [
                matlab_datenum_to_datetime(d).strftime("%H:%M %d %b")
                for d in av_date_num
            ]

    # Week days
    elif index == 5:
        date_subset = date_num[i_min : i_max + 1]
        val_subset = val[i_min : i_max + 1]

        weekday_vals = []
        weekday_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

        for j in range(7):
            # Python weekday: Monday=0, Sunday=6
            # MATLAB weekday: Sunday=1, Saturday=7
            matlab_weekday = j + 2 if j < 6 else 1

            weekday_mask = np.array(
                [matlab_datenum_to_datetime(d).weekday() == j for d in date_subset]
            )

            if np.any(weekday_mask):
                day_vals = val_subset[weekday_mask]
                valid_vals = day_vals[~np.isnan(day_vals)]

                if len(valid_vals) > 0:
                    weekday_vals.append(np.mean(valid_vals))
                else:
                    weekday_vals.append(np.nan)
            else:
                weekday_vals.append(np.nan)

        av_date_num = np.arange(1, 8)
        av_val = np.array(weekday_vals).reshape(-1, 1)
        av_date_str = weekday_names

    # Daily running means
    elif index == 6:
        av_date_num = date_num[i_min : i_max + 1]
        di = 11
        running_vals = []

        for i in range(len(av_date_num)):
            i1 = max(i - di, 0)
            i2 = min(i + di, len(av_date_num) - 1)
            window_vals = val[i_min + i1 : i_min + i2 + 1]
            running_vals.append(np.mean(window_vals))

        av_val = np.array(running_vals).reshape(-1, 1)
        av_date_str = [
            matlab_datenum_to_datetime(d).strftime("%H:%M %d %b") for d in av_date_num
        ]

    # Weekly means starting on Monday
    elif index == 7:
        date_subset = date_num[i_min : i_max + 1]
        val_subset = val[i_min : i_max + 1]

        # Find Mondays at hour 0
        monday_mask = np.array(
            [
                (
                    matlab_datenum_to_datetime(d).weekday() == 0
                    and matlab_datenum_to_datetime(d).hour == 0
                )
                for d in date_subset
            ]
        )

        monday_indices = np.where(monday_mask)[0]
        n_av = 24 * 7  # Week in hours

        if len(monday_indices) > 0:
            weekly_dates = []
            weekly_vals = []

            for j in range(len(monday_indices) + 1):
                if j == 0:
                    i_start = 0
                    i_end = (
                        monday_indices[0]
                        if len(monday_indices) > 0
                        else len(date_subset)
                    )
                else:
                    i_start = monday_indices[j - 1]
                    i_end = min(i_start + n_av, len(date_subset))

                week_vals = val_subset[i_start:i_end]
                valid_vals = week_vals[~np.isnan(week_vals)]

                if len(valid_vals) > n_av // 4:
                    weekly_vals.append(np.mean(valid_vals))
                else:
                    weekly_vals.append(np.nan)

                weekly_dates.append(date_subset[i_start])

            av_date_num = np.array(weekly_dates)
            av_val = np.array(weekly_vals).reshape(-1, 1)
        else:
            av_date_num = np.array([date_subset[0]])
            av_val = np.array([np.nan]).reshape(-1, 1)

        av_date_str = [
            matlab_datenum_to_datetime(d).strftime("%d %b") for d in av_date_num
        ]

    # Monthly means
    elif index == 8:
        date_subset = date_num[i_min : i_max + 1]
        val_subset = val[i_min : i_max + 1]

        start_dt = matlab_datenum_to_datetime(date_subset[0])
        end_dt = matlab_datenum_to_datetime(date_subset[-1])

        n_av = 24 * 30  # Month in hours (approximate)
        monthly_dates = []
        monthly_vals = []

        for year in range(start_dt.year, end_dt.year + 1):
            for month in range(1, 13):
                if (
                    (year == start_dt.year and month >= start_dt.month)
                    and (year == end_dt.year and month <= end_dt.month)
                ) or (start_dt.year < year < end_dt.year):
                    month_mask = np.array(
                        [
                            (
                                matlab_datenum_to_datetime(d).year == year
                                and matlab_datenum_to_datetime(d).month == month
                            )
                            for d in date_subset
                        ]
                    )

                    if np.any(month_mask):
                        month_vals = val_subset[month_mask]
                        valid_vals = month_vals[~np.isnan(month_vals)]

                        if len(valid_vals) > n_av // 4:
                            if index2 == 2:
                                monthly_vals.append(np.median(valid_vals))
                            else:
                                monthly_vals.append(np.mean(valid_vals))
                        else:
                            monthly_vals.append(np.nan)

                        first_day_month = datetime(year, month, 1)
                        monthly_dates.append(
                            datetime_to_matlab_datenum(first_day_month)
                        )

        av_date_num = np.array(monthly_dates)
        av_val = np.array(monthly_vals).reshape(-1, 1)

        # Filter out NaN dates
        valid_mask = ~np.isnan(av_date_num)
        av_date_str = [
            matlab_datenum_to_datetime(d).strftime("%b %Y")
            for d in av_date_num[valid_mask]
        ]

    return av_date_str, av_date_num, av_val
