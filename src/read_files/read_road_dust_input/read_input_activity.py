import pandas as pd
import numpy as np
import logging
from input_classes import input_activity
from pd_util import find_column_index, safe_float
import constants

logger = logging.getLogger(__name__)


def read_input_activity(
    activity_df: pd.DataFrame,
    traffic_year: np.ndarray | None = None,
    traffic_month: np.ndarray | None = None,
    traffic_day: np.ndarray | None = None,
    traffic_hour: np.ndarray | None = None,
    traffic_minute: np.ndarray | None = None,
    print_results: bool = False,
    nodata: float = -99.0,
) -> input_activity:
    """
    Read activity input data from a pandas DataFrame.

    Args:
        activity_df: pandas DataFrame containing activity input data
        traffic_year: Year array from traffic data for redistribution (optional)
        traffic_month: Month array from traffic data for redistribution (optional)
        traffic_day: Day array from traffic data for redistribution (optional)
        traffic_hour: Hour array from traffic data for redistribution (optional)
        traffic_minute: Minute array from traffic data for redistribution (optional)
        print_results: Whether to print results to console

    Returns:
        input_activity: input_activity object containing parsed data
    """

    loaded_activity = input_activity()

    # Check if activity data exists
    if activity_df is None or activity_df.empty:
        logger.warning("No activity data provided - initializing with default values")
        # Initialize with default values for traffic length if available
        if traffic_year is not None:
            n_date = len(traffic_year)
            loaded_activity.M_salting = np.zeros((constants.num_salt, n_date))
            loaded_activity.g_road_wetting = np.zeros(n_date)
            loaded_activity.M_sanding = np.zeros(n_date)
            loaded_activity.t_ploughing = np.zeros(n_date)
            loaded_activity.t_cleaning = np.zeros(n_date)
            loaded_activity.M_fugitive = np.zeros(n_date)

            # Set salt type defaults
            loaded_activity.salt_type = np.array(
                [constants.na, constants.mg], dtype=np.int32
            )
            loaded_activity.second_salt_type = constants.mg
            loaded_activity.salt2_str = "mg"

            # Copy input arrays
            loaded_activity.M_salting_input = loaded_activity.M_salting.copy()
            loaded_activity.g_road_wetting_input = loaded_activity.g_road_wetting.copy()
            loaded_activity.M_sanding_input = loaded_activity.M_sanding.copy()
            loaded_activity.t_ploughing_input = loaded_activity.t_ploughing.copy()
            loaded_activity.t_cleaning_input = loaded_activity.t_cleaning.copy()
            loaded_activity.M_fugitive_input = loaded_activity.M_fugitive.copy()

        return loaded_activity

    # Clean and set headers
    header_text = activity_df.iloc[0, :].astype(str).str.replace(r"\s+", "", regex=True)
    activity_df.columns = header_text
    activity_df = activity_df.iloc[1:].reset_index(drop=True)

    # Read date/time data
    loaded_activity.year = activity_df["Year"].values.astype(int)
    loaded_activity.month = activity_df["Month"].values.astype(int)
    loaded_activity.day = activity_df["Day"].values.astype(int)
    loaded_activity.hour = activity_df["Hour"].values.astype(int)

    # Handle optional minute column
    col_idx = find_column_index("Minute", header_text, print_results)
    if col_idx != -1:
        loaded_activity.minute = (
            activity_df.iloc[:, col_idx].apply(lambda x: safe_float(x, nodata)).values.astype(int)
        )
    else:
        loaded_activity.minute = np.zeros_like(loaded_activity.year)

    n_act = len(loaded_activity.year)
    loaded_activity.n_act = n_act

    if n_act == 0:
        logger.warning("No activity records found")
        return loaded_activity

    # Initialize activity data arrays
    M_sanding = np.zeros(n_act)
    M_salting = np.zeros((constants.num_salt, n_act))
    t_ploughing = np.zeros(n_act)
    t_cleaning = np.zeros(n_act)
    g_road_wetting = np.zeros(n_act)
    M_fugitive = np.zeros(n_act)

    # Read M_sanding (optional)
    col_idx = find_column_index("M_sanding", header_text, print_results)
    if col_idx != -1:
        M_sanding = activity_df.iloc[:, col_idx].apply(lambda x: safe_float(x, nodata)).values
    else:
        if print_results:
            logger.info("M_sanding column not found - using zeros")

    # Read primary salt (na) - always first salt type
    col_idx = find_column_index("M_salting(na)", header_text, print_results)
    if col_idx != -1:
        M_salting[0, :] = activity_df.iloc[:, col_idx].apply(lambda x: safe_float(x, nodata)).values
    else:
        if print_results:
            logger.info("M_salting(na) column not found - using zeros")

    # Read secondary salt with priority: mg > cma > ca
    second_salt_available = 0
    second_salt_type = constants.mg
    salt2_str = ""

    # Try mg first
    col_idx = find_column_index("M_salting(mg)", header_text, print_results)
    if col_idx != -1:
        M_salting[1, :] = activity_df.iloc[:, col_idx].apply(lambda x: safe_float(x, nodata)).values
        second_salt_type = constants.mg
        second_salt_available = 1
        salt2_str = "mg"
    else:
        # Try cma if mg not available
        col_idx = find_column_index("M_salting(cma)", header_text, print_results)
        if col_idx != -1:
            M_salting[1, :] = activity_df.iloc[:, col_idx].apply(lambda x: safe_float(x, nodata)).values
            second_salt_type = constants.cma
            second_salt_available = 1
            salt2_str = "cma"
        else:
            # Try ca if neither mg nor cma available
            col_idx = find_column_index("M_salting(ca)", header_text, print_results)
            if col_idx != -1:
                M_salting[1, :] = activity_df.iloc[:, col_idx].apply(lambda x: safe_float(x, nodata)).values
                second_salt_type = constants.ca
                second_salt_available = 1
                salt2_str = "ca"

    if second_salt_available == 0:
        second_salt_type = constants.mg
        salt2_str = "mg"
        if print_results:
            logger.info("No secondary salt data found - defaulting to mg")

    # Read Wetting (optional)
    col_idx = find_column_index("Wetting", header_text, print_results)
    g_road_wetting_available = 0
    if col_idx != -1:
        g_road_wetting = activity_df.iloc[:, col_idx].apply(lambda x: safe_float(x, nodata)).values
        g_road_wetting_available = 1
    else:
        if print_results:
            logger.info("Wetting column not found - using zeros")

    # Read Ploughing_road (required for MATLAB compatibility)
    col_idx = find_column_index("Ploughing_road", header_text, print_results)
    if col_idx != -1:
        t_ploughing = activity_df.iloc[:, col_idx].apply(lambda x: safe_float(x, nodata)).values
    else:
        if print_results:
            logger.warning("Ploughing_road column not found - using zeros")

    # Read Cleaning_road (required for MATLAB compatibility)
    col_idx = find_column_index("Cleaning_road", header_text, print_results)
    if col_idx != -1:
        t_cleaning = activity_df.iloc[:, col_idx].apply(lambda x: safe_float(x, nodata)).values
    else:
        if print_results:
            logger.warning("Cleaning_road column not found - using zeros")

    # Read Fugitive (optional)
    col_idx = find_column_index("Fugitive", header_text, print_results)
    if col_idx != -1:
        M_fugitive = activity_df.iloc[:, col_idx].apply(lambda x: safe_float(x, nodata)).values
    else:
        if print_results:
            logger.info("Fugitive column not found - using zeros")

    # Set salt type information
    loaded_activity.salt_type = np.array(
        [constants.na, second_salt_type], dtype=np.int32
    )
    loaded_activity.second_salt_type = second_salt_type
    loaded_activity.second_salt_available = second_salt_available
    loaded_activity.salt2_str = salt2_str
    loaded_activity.g_road_wetting_available = g_road_wetting_available

    # If activity dates don't correspond to traffic dates, redistribute data
    if traffic_year is not None and len(loaded_activity.year) != len(traffic_year):
        if print_results:
            logger.info("Redistributing activity input data to match traffic timeline")

        n_traf = len(traffic_year)

        # Store original activity data
        M_sanding_act = M_sanding.copy()
        M_fugitive_act = M_fugitive.copy()
        M_salting_act = M_salting.copy()
        g_road_wetting_act = g_road_wetting.copy()
        t_ploughing_act = t_ploughing.copy()
        t_cleaning_act = t_cleaning.copy()

        # Initialize arrays for traffic timeline
        M_sanding = np.zeros(n_traf)
        M_fugitive = np.zeros(n_traf)
        M_salting = np.zeros((constants.num_salt, n_traf))
        g_road_wetting = np.zeros(n_traf)
        t_ploughing = np.zeros(n_traf)
        t_cleaning = np.zeros(n_traf)

        # Redistribute activity data to traffic timeline
        for i in range(n_act):
            # Find matching time in traffic data
            matches = np.where(
                (loaded_activity.year[i] == traffic_year)
                & (loaded_activity.month[i] == traffic_month)
                & (loaded_activity.day[i] == traffic_day)
                & (loaded_activity.hour[i] == traffic_hour)
                & (loaded_activity.minute[i] == traffic_minute)
            )[0]

            if len(matches) == 1:
                r = matches[0]
                M_sanding[r] += M_sanding_act[i]
                M_fugitive[r] += M_fugitive_act[i]
                M_salting[0, r] += M_salting_act[0, i]
                M_salting[1, r] += M_salting_act[1, i]
                g_road_wetting[r] += g_road_wetting_act[i]
                t_ploughing[r] += t_ploughing_act[i]
                t_cleaning[r] += t_cleaning_act[i]
            elif len(matches) > 1 and print_results:
                logger.warning(
                    "Problem with activity input data - multiple time matches found"
                )

    # Store final arrays in the dataclass
    loaded_activity.M_sanding = M_sanding
    loaded_activity.M_salting = M_salting
    loaded_activity.t_ploughing = t_ploughing
    loaded_activity.t_cleaning = t_cleaning
    loaded_activity.g_road_wetting = g_road_wetting
    loaded_activity.M_fugitive = M_fugitive

    # Store input arrays (original data before redistribution)
    loaded_activity.M_sanding_input = loaded_activity.M_sanding.copy()
    loaded_activity.M_salting_input = loaded_activity.M_salting.copy()
    loaded_activity.t_ploughing_input = loaded_activity.t_ploughing.copy()
    loaded_activity.t_cleaning_input = loaded_activity.t_cleaning.copy()
    loaded_activity.g_road_wetting_input = loaded_activity.g_road_wetting.copy()
    loaded_activity.M_fugitive_input = loaded_activity.M_fugitive.copy()

    logger.info(
        f"Successfully loaded activity data with {loaded_activity.n_act} activity records"
    )

    return loaded_activity
