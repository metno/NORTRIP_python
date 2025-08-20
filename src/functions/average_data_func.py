import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, List


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

    av_date_str: List[str] = []
    av_date_num = np.array([])
    av_val = np.array([])

    # Convert MATLAB datenum to Python datetime for processing
    def matlab_datenum_to_datetime(datenum: float) -> datetime:
        """Convert MATLAB datenum to Python datetime."""
        # MATLAB datenum 1 = January 1, year 0000
        # Python datetime reference is January 1, year 1
        matlab_epoch = datetime(1, 1, 1)
        days_since_epoch = datenum - 1
        return matlab_epoch + timedelta(days=days_since_epoch)

    def datetime_to_matlab_datenum(dt: datetime) -> float:
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
        # Hourly means - match MATLAB behavior
        min_num = int(index2) if not np.isnan(index2) else 1

        start_dt = matlab_datenum_to_datetime(date_num[i_min])
        end_dt = matlab_datenum_to_datetime(date_num[i_max])

        # Build hourly grid from start to end inclusive of the last hour
        current_hour = start_dt.replace(minute=0, second=0, microsecond=0)
        hourly_dates = []
        hourly_vals = []

        # Precompute vectors
        all_dt = [matlab_datenum_to_datetime(d) for d in date_num]

        while current_hour <= end_dt + timedelta(hours=1):
            # Find indices exactly matching this Y-M-D-H
            matches = [
                idx
                for idx, dt in enumerate(all_dt)
                if dt.year == current_hour.year
                and dt.month == current_hour.month
                and dt.day == current_hour.day
                and dt.hour == current_hour.hour
            ]
            # Apply strict window bounds (r > i_min & r < i_max)
            r2 = [idx for idx in matches if (idx > i_min and idx < i_max)]

            if len(r2) >= 1:
                vals = val[r2]
                valid = vals[~np.isnan(vals)]
                if len(valid) > min_num:
                    hourly_vals.append(np.max(valid) if use_max else np.mean(valid))
                else:
                    hourly_vals.append(np.nan)
            else:
                hourly_vals.append(np.nan)

            hourly_dates.append(datetime_to_matlab_datenum(current_hour))
            current_hour += timedelta(hours=1)

        av_date_num = np.array(hourly_dates)
        av_val = np.array(hourly_vals).reshape(-1, 1)
        av_date_str = [
            matlab_datenum_to_datetime(d).strftime("%d %b") for d in av_date_num
        ]

    # Daily means
    elif index == 2:
        # Daily means - match MATLAB behavior
        min_num = int(index2) if not np.isnan(index2) else 6

        start_day = int(np.floor(date_num[i_min]))
        end_day = int(np.floor(date_num[i_max]))

        daily_dates = []
        daily_vals = []

        for i_day in range(start_day, end_day + 1):
            r = np.where(np.floor(date_num) == i_day)[0]
            # Strict bounds (r > i_min & r < i_max)
            r2 = r[(r > i_min) & (r < i_max)]
            if len(r2) >= 12:
                vals = val[r2]
                valid = vals[~np.isnan(vals)]
                if len(valid) > min_num:
                    daily_vals.append(np.max(valid) if use_max else np.mean(valid))
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

            for idx in halfday_indices:
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
        # Daily running means (MATLAB: no NaN traps, so NaN in window yields NaN)
        av_date_num = date_num[i_min : i_max + 1]
        di = 11
        running_vals = []

        for i in range(len(av_date_num)):
            i1 = max(i - di, 0)
            i2 = min(i + di, len(av_date_num) - 1)
            window_vals = val[i_min + i1 : i_min + i2 + 1]
            # Use plain mean (propagates NaN like MATLAB)
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
        # Monthly means - match MATLAB behavior
        date_subset = date_num[i_min : i_max + 1]
        val_subset = val[i_min : i_max + 1]

        # Year-month arrays
        Y = np.array([matlab_datenum_to_datetime(d).year for d in date_subset])
        M = np.array([matlab_datenum_to_datetime(d).month for d in date_subset])

        n_av = 24 * 30
        monthly_dates = []
        monthly_vals = []

        start_year = Y[0]
        end_year = Y[-1]
        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                r_month = np.where((M == month) & (Y == year))[0]
                if r_month.size > 0:
                    val_temp2 = val_subset[r_month]
                    valid = val_temp2[~np.isnan(val_temp2)]
                    if len(valid) > n_av // 4:
                        if index2 == 2:
                            monthly_vals.append(np.median(valid))
                        else:
                            monthly_vals.append(np.mean(valid))
                    else:
                        monthly_vals.append(np.nan)
                    # Use the first sample time in that month
                    monthly_dates.append(date_subset[r_month[0]])

        av_date_num = np.array(monthly_dates)
        av_val = np.array(monthly_vals).reshape(-1, 1)
        av_date_str = [
            matlab_datenum_to_datetime(d).strftime("%b %Y") for d in av_date_num
        ]

    return av_date_str, av_date_num, av_val
