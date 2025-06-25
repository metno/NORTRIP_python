import pandas as pd
import numpy as np
import datetime
import logging
from input_classes import input_traffic
from pd_util import find_column_index
import constants

logger = logging.getLogger(__name__)


def _calculate_daily_averages(
    date_num: np.ndarray, data: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate hourly averages for daily cycle (equivalent to MATLAB Average_data_func with type 3)."""
    hours = np.array([int(((dn - int(dn)) * 24) % 24) for dn in date_num])

    hourly_averages = np.full(24, np.nan)  # Initialize with NaN instead of zeros
    for hour in range(24):
        hour_mask = hours == hour
        if np.any(hour_mask):
            hourly_averages[hour] = np.mean(data[hour_mask])

    return np.arange(24), hourly_averages


def read_input_traffic(
    traffic_df: pd.DataFrame, nodata: float = -99.0, print_results: bool = False
) -> input_traffic:
    """
    Read traffic data from DataFrame into input_traffic dataclass.

    Args:
        traffic_df (pd.DataFrame): DataFrame containing the traffic data
        nodata (float): Nodata value to use for missing data
        print_results (bool): Whether to print the results to the console

    Returns:
        input_traffic: Dataclass containing the traffic data
    """
    loaded_traffic = input_traffic()

    # Clean and set headers
    header_text = traffic_df.iloc[0, :].astype(str).str.replace(r"\s+", "", regex=True)
    traffic_df.columns = header_text
    traffic_df = traffic_df.iloc[1:].reset_index(drop=True)

    # Read date/time data
    loaded_traffic.year = traffic_df["Year"].values.astype(int)
    loaded_traffic.month = traffic_df["Month"].values.astype(int)
    loaded_traffic.day = traffic_df["Day"].values.astype(int)
    loaded_traffic.hour = traffic_df["Hour"].values.astype(int)
    try:
        loaded_traffic.minute = traffic_df["Minute"].values.astype(int)
    except KeyError:
        loaded_traffic.minute = loaded_traffic.year * 0

    n_traffic = len(loaded_traffic.year)
    loaded_traffic.n_traffic = n_traffic

    # Convert to datetime objects for processing
    datetime_objects = [
        datetime.datetime(year, month, day, hour, minute, 0)
        for year, month, day, hour, minute in zip(
            loaded_traffic.year,
            loaded_traffic.month,
            loaded_traffic.day,
            loaded_traffic.hour,
            loaded_traffic.minute,
            strict=True,
        )
    ]

    # Create date_num similar to MATLAB datenum (days since year 1 + fractional day)
    loaded_traffic.date_num = np.array(
        [
            dt.toordinal() + 366 + dt.hour / 24.0 + dt.minute / (24.0 * 60.0)
            for dt in datetime_objects
        ]
    )

    # Create date_str arrays matching MATLAB datestr format
    date_str_format1 = np.array([dt.strftime("%Y.%m.%d %H") for dt in datetime_objects])
    date_str_format2 = np.array(
        [dt.strftime("%H:%M %d %b ") for dt in datetime_objects]
    )

    # Combine into 2D array as expected by the class
    loaded_traffic.date_str = np.array(
        [date_str_format1, date_str_format2], dtype=object
    )

    # Read traffic volumes
    # Total traffic
    col_idx = find_column_index(
        "N(total)", header_text, print_results, exact_match=True
    )
    if col_idx == -1:
        logger.error("No traffic data found - N(total) column missing")
        return loaded_traffic

    loaded_traffic.N_total = traffic_df.iloc[:, col_idx].values.astype(float)

    # Initialize arrays with correct dimensions
    loaded_traffic.N_v = np.zeros((constants.num_veh, n_traffic))
    loaded_traffic.N = np.zeros((constants.num_tyre, constants.num_veh, n_traffic))
    loaded_traffic.V_veh = np.zeros((constants.num_veh, n_traffic))

    # Vehicle type traffic volumes
    for v, veh_type in enumerate(["he", "li"]):
        col_idx = find_column_index(
            f"N({veh_type})", header_text, print_results, exact_match=True
        )
        if col_idx != -1:
            loaded_traffic.N_v[v, :] = traffic_df.iloc[:, col_idx].values.astype(float)

    # Tyre type traffic volumes
    for t, tyre_type in enumerate(["st", "wi", "su"]):
        for v, veh_type in enumerate(["he", "li"]):
            col_idx = find_column_index(
                f"N({tyre_type},{veh_type})",
                header_text,
                print_results,
                exact_match=True,
            )
            if col_idx != -1:
                loaded_traffic.N[t, v, :] = traffic_df.iloc[:, col_idx].values.astype(
                    float
                )

    # Vehicle speeds
    for v, veh_type in enumerate(["he", "li"]):
        col_idx = find_column_index(
            f"V_veh({veh_type})", header_text, print_results, exact_match=True
        )
        if col_idx != -1:
            loaded_traffic.V_veh[v, :] = traffic_df.iloc[:, col_idx].values.astype(
                float
            )

    # Check data and find missing values
    # Find good data indices (all traffic data present and valid)
    temp = np.zeros((6, n_traffic))
    temp[0, :] = loaded_traffic.N[constants.st, constants.he, :]
    temp[1, :] = loaded_traffic.N[constants.wi, constants.he, :]
    temp[2, :] = loaded_traffic.N[constants.su, constants.he, :]
    temp[3, :] = loaded_traffic.N[constants.st, constants.li, :]
    temp[4, :] = loaded_traffic.N[constants.wi, constants.li, :]
    temp[5, :] = loaded_traffic.N[constants.su, constants.li, :]

    N_good_data = np.where(
        (loaded_traffic.N_total != nodata)
        & (~np.isnan(loaded_traffic.N_total))
        & (loaded_traffic.N_v[constants.li, :] != nodata)
        & (~np.isnan(loaded_traffic.N_v[constants.li, :]))
        & (loaded_traffic.N_v[constants.he, :] != nodata)
        & (~np.isnan(loaded_traffic.N_v[constants.he, :]))
        & (temp[0, :] != nodata)
        & (~np.isnan(temp[0, :]))
        & (temp[1, :] != nodata)
        & (~np.isnan(temp[1, :]))
        & (temp[2, :] != nodata)
        & (~np.isnan(temp[2, :]))
        & (temp[3, :] != nodata)
        & (~np.isnan(temp[3, :]))
        & (temp[4, :] != nodata)
        & (~np.isnan(temp[4, :]))
        & (temp[5, :] != nodata)
        & (~np.isnan(temp[5, :]))
    )[0]
    # Handle missing N_total data
    N_total_nodata = np.where(
        (loaded_traffic.N_total == nodata) | np.isnan(loaded_traffic.N_total)
    )[0]
    loaded_traffic.N_total_nodata = N_total_nodata.tolist()

    # Handle missing N_v data
    N_v_nodata = []
    for v in range(constants.num_veh):
        missing_indices = np.where(
            (loaded_traffic.N_v[v, :] == nodata) | np.isnan(loaded_traffic.N_v[v, :])
        )[0]
        N_v_nodata.append(missing_indices.tolist())

        if len(missing_indices) > 0 and len(missing_indices) != n_traffic:
            for i in missing_indices:
                if i > 0:
                    loaded_traffic.N_v[v, i] = loaded_traffic.N_v[v, i - 1]

    loaded_traffic.N_v_nodata = N_v_nodata

    # Handle missing N data
    N_nodata = []
    for t in range(constants.num_tyre):
        tyre_nodata = []
        for v in range(constants.num_veh):
            missing_indices = np.where(
                (loaded_traffic.N[t, v, :] == nodata)
                | np.isnan(loaded_traffic.N[t, v, :])
            )[0]
            tyre_nodata.append(missing_indices.tolist())

            if len(missing_indices) > 0 and len(missing_indices) != n_traffic:
                for i in missing_indices:
                    if i > 0:
                        loaded_traffic.N[t, v, i] = loaded_traffic.N[t, v, i - 1]
        N_nodata.append(tyre_nodata)

    loaded_traffic.N_nodata = N_nodata

    # Handle missing V_veh data (including zeros)
    V_veh_nodata = []
    for v in range(constants.num_veh):
        missing_indices = np.where(
            (loaded_traffic.V_veh[v, :] == nodata)
            | (loaded_traffic.V_veh[v, :] == 0)
            | np.isnan(loaded_traffic.V_veh[v, :])
        )[0]
        V_veh_nodata.append(missing_indices.tolist())

        if len(missing_indices) > 0 and len(missing_indices) != n_traffic:
            for i in missing_indices:
                if i > 0:
                    loaded_traffic.V_veh[v, i] = loaded_traffic.V_veh[v, i - 1]

    loaded_traffic.V_veh_nodata = V_veh_nodata

    # Create ratios for imputation
    loaded_traffic.N_ratio = np.zeros(
        (constants.num_tyre, constants.num_veh, n_traffic)
    )

    for v in range(constants.num_veh):
        non_zero_indices = np.where(loaded_traffic.N_v[v, :] != 0)[0]
        for t in range(constants.num_tyre):
            loaded_traffic.N_ratio[t, v, non_zero_indices] = (
                loaded_traffic.N[t, v, non_zero_indices]
                / loaded_traffic.N_v[v, non_zero_indices]
            )

    # Create missing traffic data using average daily cycles
    if len(N_total_nodata) > 0 and len(N_good_data) > 0:
        xplot, yplot = _calculate_daily_averages(
            loaded_traffic.date_num[N_good_data], loaded_traffic.N_total[N_good_data]
        )

        for i in N_total_nodata:
            hour_val = loaded_traffic.hour[i]
            if hour_val < 24 and not np.isnan(yplot[hour_val]):
                loaded_traffic.N_total[i] = yplot[hour_val]

    # Forward fill as fallback for any remaining missing data
    if len(N_total_nodata) > 0 and len(N_total_nodata) != n_traffic:
        for i in N_total_nodata:
            if i > 0 and (
                loaded_traffic.N_total[i] == nodata
                or np.isnan(loaded_traffic.N_total[i])
            ):
                loaded_traffic.N_total[i] = loaded_traffic.N_total[i - 1]

    # Handle N_v missing data using daily averages
    if any(len(sublist) > 0 for sublist in N_v_nodata) and len(N_good_data) > 0:
        for v in range(constants.num_veh):
            if len(N_v_nodata[v]) > 0:
                xplot, yplot = _calculate_daily_averages(
                    loaded_traffic.date_num[N_good_data],
                    loaded_traffic.N_v[v, N_good_data],
                )

                for i in N_v_nodata[v]:
                    hour_val = loaded_traffic.hour[i]
                    if hour_val < 24:
                        loaded_traffic.N_v[v, i] = yplot[hour_val]

    # Update N data based on corrected N_v and ratios
    if any(len(sublist) > 0 for sublist in N_v_nodata):
        for t in range(constants.num_tyre):
            for v in range(constants.num_veh):
                for i in N_v_nodata[v]:
                    loaded_traffic.N[t, v, i] = (
                        loaded_traffic.N_ratio[t, v, i] * loaded_traffic.N_v[v, i]
                    )

    logger.info(
        f"Successfully loaded traffic data with {loaded_traffic.n_traffic} records"
    )
    return loaded_traffic
