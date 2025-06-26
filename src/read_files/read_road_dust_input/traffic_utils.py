import numpy as np
import constants


def calculate_daily_averages(
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


def calculate_good_traffic_data_indices(traffic_data, nodata: float) -> np.ndarray:
    """
    Calculate indices of good traffic data (all traffic data present and valid).

    This replicates the logic from the traffic reading function to identify
    time indices where all traffic data is available and valid.

    Args:
        traffic_data: Traffic data object with N_total, N_v, and N arrays
        nodata: Missing data value

    Returns:
        np.ndarray: Array of indices where all traffic data is valid
    """
    # Find good data indices (all traffic data present and valid)
    temp = np.zeros((6, traffic_data.n_traffic))
    temp[0, :] = traffic_data.N[constants.st, constants.he, :]
    temp[1, :] = traffic_data.N[constants.wi, constants.he, :]
    temp[2, :] = traffic_data.N[constants.su, constants.he, :]
    temp[3, :] = traffic_data.N[constants.st, constants.li, :]
    temp[4, :] = traffic_data.N[constants.wi, constants.li, :]
    temp[5, :] = traffic_data.N[constants.su, constants.li, :]

    N_good_data = np.where(
        (traffic_data.N_total != nodata)
        & (~np.isnan(traffic_data.N_total))
        & (traffic_data.N_v[constants.li, :] != nodata)
        & (~np.isnan(traffic_data.N_v[constants.li, :]))
        & (traffic_data.N_v[constants.he, :] != nodata)
        & (~np.isnan(traffic_data.N_v[constants.he, :]))
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

    return N_good_data
