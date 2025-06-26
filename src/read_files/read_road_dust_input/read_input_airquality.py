import pandas as pd
import numpy as np
import logging
from input_classes import input_airquality
from pd_util import (
    find_column_index,
    safe_float,
)
from .traffic_utils import calculate_daily_averages
import constants

logger = logging.getLogger(__name__)


def read_input_airquality(
    airquality_df: pd.DataFrame,
    ospm_df: pd.DataFrame | None = None,
    nodata: float = -99.0,
    traffic_date_num: np.ndarray | None = None,
    traffic_hour: np.ndarray | None = None,
    N_total_nodata: list | None = None,
    N_good_data: np.ndarray | None = None,
    print_results: bool = False,
) -> input_airquality:
    """
    Read air quality input data from a pandas DataFrame.

    Args:
        airquality_df (pd.DataFrame): DataFrame containing the air quality data
        ospm_df (pd.DataFrame | None): Optional DataFrame containing OSPM data
        nodata (float): Nodata value to use for missing data
        traffic_date_num (np.ndarray | None): Date numbers from traffic data for missing data filling
        traffic_hour (np.ndarray | None): Hour array from traffic data for missing data filling
        N_total_nodata (list | None): Indices of missing traffic data for emission data filling
        N_good_data (np.ndarray | None): Indices of good traffic data for daily average calculation
        print_results (bool): Whether to print the results to the console

    Returns:
        input_airquality: Dataclass containing the air quality data
    """
    loaded_airquality = input_airquality()

    # Check if airquality data exists
    if airquality_df is None or airquality_df.empty:
        logger.warning(
            "No air quality data provided - initializing with default values"
        )
        return loaded_airquality

    # Clean and set headers - first row contains headers
    header_text = (
        airquality_df.iloc[0, :].astype(str).str.replace(r"\s+", "", regex=True)
    )
    airquality_df.columns = header_text
    airquality_df = airquality_df.iloc[1:].reset_index(drop=True)

    n_date = len(airquality_df)
    loaded_airquality.n_date = n_date

    if n_date == 0:
        logger.warning("No air quality records found")
        return loaded_airquality

    # Initialize data arrays with correct dimensions
    PM_obs = np.full((constants.num_size, n_date), nodata)
    PM_background = np.full((constants.num_size, n_date), nodata)
    NOX_obs = np.full(n_date, nodata)
    NOX_background = np.full(n_date, nodata)
    NOX_emis = np.full(n_date, nodata)
    EP_emis = np.full(n_date, nodata)
    Salt_obs = np.full((constants.num_salt, n_date), nodata)
    f_dis_input = np.full(n_date, nodata)

    # Read PM10 observations
    col_idx = find_column_index("PM10_obs", header_text, print_results)
    if col_idx != -1:
        PM_obs[constants.pm_10, :] = (
            airquality_df.iloc[:, col_idx].apply(safe_float).values
        )

    # Read PM10 background
    col_idx = find_column_index("PM10_background", header_text, print_results)
    if col_idx != -1:
        PM_background[constants.pm_10, :] = (
            airquality_df.iloc[:, col_idx].apply(safe_float).values
        )

    # Read PM25 observations
    col_idx = find_column_index("PM25_obs", header_text, print_results)
    if col_idx != -1:
        PM_obs[constants.pm_25, :] = (
            airquality_df.iloc[:, col_idx].apply(safe_float).values
        )

    # Read PM25 background
    col_idx = find_column_index("PM25_background", header_text, print_results)
    if col_idx != -1:
        PM_background[constants.pm_25, :] = (
            airquality_df.iloc[:, col_idx].apply(safe_float).values
        )

    # Read NOX observations
    col_idx = find_column_index("NOX_obs", header_text, print_results)
    if col_idx != -1:
        NOX_obs[:] = airquality_df.iloc[:, col_idx].apply(safe_float).values

    # Read NOX background
    col_idx = find_column_index("NOX_background", header_text, print_results)
    if col_idx != -1:
        NOX_background[:] = airquality_df.iloc[:, col_idx].apply(safe_float).values

    # Read NOX emissions (optional)
    col_idx = find_column_index("NOX_emis", header_text, print_results)
    NOX_emis_available = 0
    if col_idx != -1:
        NOX_emis[:] = airquality_df.iloc[:, col_idx].apply(safe_float).values
        NOX_emis_available = 1

    # Read EP emissions (optional)
    col_idx = find_column_index("EP_emis", header_text, print_results)
    EP_emis_available = 0
    if col_idx != -1:
        EP_emis[:] = airquality_df.iloc[:, col_idx].apply(safe_float).values
        EP_emis_available = 1

    # Read Salt observations for sodium (optional)
    col_idx = find_column_index("Salt_obs(na)", header_text, print_results)
    Salt_obs_available = np.zeros(constants.num_salt, dtype=int)
    if col_idx != -1:
        Salt_obs[constants.na, :] = (
            airquality_df.iloc[:, col_idx].apply(safe_float).values
        )
        Salt_obs_available[constants.na] = 1

    # Read dispersion factor (optional)
    col_idx = find_column_index("Disp_fac", header_text, print_results)
    f_dis_available = 0
    if col_idx != -1:
        f_dis_input[:] = airquality_df.iloc[:, col_idx].apply(safe_float).values
        f_dis_available = 1

    # Handle NaN values by converting to nodata
    PM_obs[np.isnan(PM_obs)] = nodata
    PM_background[np.isnan(PM_background)] = nodata
    NOX_obs[np.isnan(NOX_obs)] = nodata
    NOX_background[np.isnan(NOX_background)] = nodata
    NOX_emis[np.isnan(NOX_emis)] = nodata
    EP_emis[np.isnan(EP_emis)] = nodata

    # Check availability after NaN handling
    if np.all(NOX_emis == nodata) or np.all(np.isnan(NOX_emis)):
        NOX_emis_available = 0
    if np.all(EP_emis == nodata) or np.all(np.isnan(EP_emis)):
        EP_emis_available = 0

    # Handle salt observations NaN values
    if Salt_obs_available[constants.na]:
        Salt_obs[constants.na, np.isnan(Salt_obs[constants.na, :])] = nodata

    # Replace emission data when there is no traffic data (as this is usually coupled)
    if (
        N_total_nodata is not None
        and len(N_total_nodata) > 0
        and N_good_data is not None
        and len(N_good_data) > 0
        and traffic_date_num is not None
        and traffic_hour is not None
    ):
        # Fill NOX emissions using daily averages
        if NOX_emis_available:
            xplot, yplot = calculate_daily_averages(
                traffic_date_num[N_good_data], NOX_emis[N_good_data]
            )

            for i in N_total_nodata:
                hour_val = traffic_hour[i]
                if hour_val < 24 and not np.isnan(yplot[hour_val]):
                    NOX_emis[i] = yplot[hour_val]

        # Fill EP emissions using daily averages
        if EP_emis_available:
            xplot, yplot = calculate_daily_averages(
                traffic_date_num[N_good_data], EP_emis[N_good_data]
            )

            for i in N_total_nodata:
                hour_val = traffic_hour[i]
                if hour_val < 24 and not np.isnan(yplot[hour_val]):
                    EP_emis[i] = yplot[hour_val]

    # Calculate net concentrations (PM_obs - PM_background)
    PM_obs_net = np.full((constants.num_size, n_date), np.nan)
    PM_obs_bg = np.full((constants.num_size, n_date), np.nan)
    NOX_obs_net = np.full(n_date, np.nan)

    for ti in range(n_date):
        # Calculate PM net concentrations for pm_10 and pm_25
        for x in [constants.pm_10, constants.pm_25]:
            if PM_obs[x, ti] != nodata and PM_background[x, ti] != nodata:
                PM_obs_net[x, ti] = PM_obs[x, ti] - PM_background[x, ti]
                if PM_obs_net[x, ti] <= 0:
                    PM_obs_net[x, ti] = nodata
            else:
                PM_obs_net[x, ti] = nodata

            # Rewrite background concentrations for consistency
            PM_obs_bg[x, ti] = PM_background[x, ti]

        # Calculate NOX net concentrations
        if NOX_obs[ti] != nodata and NOX_background[ti] != nodata:
            NOX_obs_net[ti] = NOX_obs[ti] - NOX_background[ti]
            if NOX_obs_net[ti] <= 0:
                NOX_obs_net[ti] = nodata
        else:
            NOX_obs_net[ti] = nodata

    # Read OSPM data if it exists
    OSPM_data_exists = 0
    U_mast_ospm_orig = np.array([])
    wind_dir_ospm_orig = np.array([])
    TK_ospm_orig = np.array([])
    GlobalRad_ospm_orig = np.array([])
    cNOx_b_ospm_orig = np.array([])
    qNOX_ospm_orig = np.array([])
    NNp_ospm_orig = np.array([])
    NNt_ospm_orig = np.array([])
    Vp_ospm_orig = np.array([])
    Vt_ospm_orig = np.array([])

    if ospm_df is not None and not ospm_df.empty:
        OSPM_data_exists = 1

        # Create a working copy to avoid type checking issues
        ospm_working_df = ospm_df.copy()

        # Clean and set headers for OSPM data
        ospm_header_text = (
            ospm_working_df.iloc[0, :].astype(str).str.replace(r"\s+", "", regex=True)
        )
        ospm_working_df.columns = ospm_header_text
        ospm_working_df = ospm_working_df.iloc[1:].reset_index(drop=True)

        n_ospm = len(ospm_working_df)

        # Initialize OSPM arrays
        U_mast_ospm_orig = np.full(n_ospm, nodata)
        wind_dir_ospm_orig = np.full(n_ospm, nodata)
        TK_ospm_orig = np.full(n_ospm, nodata)
        GlobalRad_ospm_orig = np.full(n_ospm, nodata)
        cNOx_b_ospm_orig = np.full(n_ospm, nodata)
        qNOX_ospm_orig = np.full(n_ospm, nodata)
        NNp_ospm_orig = np.full(n_ospm, nodata)
        NNt_ospm_orig = np.full(n_ospm, nodata)
        Vp_ospm_orig = np.full(n_ospm, nodata)
        Vt_ospm_orig = np.full(n_ospm, nodata)

        # Read OSPM meteorological data
        col_idx = find_column_index("FFospm(m/s)", ospm_header_text, print_results)
        if col_idx != -1:
            U_mast_ospm_orig[:] = (
                ospm_working_df.iloc[:, col_idx].apply(safe_float).values
            )

        col_idx = find_column_index("DDospm(deg)", ospm_header_text, print_results)
        if col_idx != -1:
            wind_dir_ospm_orig[:] = (
                ospm_working_df.iloc[:, col_idx].apply(safe_float).values
            )

        col_idx = find_column_index("TKospm(degK)", ospm_header_text, print_results)
        if col_idx != -1:
            TK_ospm_orig[:] = ospm_working_df.iloc[:, col_idx].apply(safe_float).values

        col_idx = find_column_index(
            "Globalradiationospm(W/m^2)", ospm_header_text, print_results
        )
        if col_idx != -1:
            GlobalRad_ospm_orig[:] = (
                ospm_working_df.iloc[:, col_idx].apply(safe_float).values
            )

        col_idx = find_column_index(
            "Cbackgroundospm(ug/m^3)", ospm_header_text, print_results
        )
        if col_idx != -1:
            cNOx_b_ospm_orig[:] = (
                ospm_working_df.iloc[:, col_idx].apply(safe_float).values
            )

        # Read OSPM traffic data
        col_idx = find_column_index("N(li)ospm", ospm_header_text, print_results)
        if col_idx != -1:
            NNp_ospm_orig[:] = ospm_working_df.iloc[:, col_idx].apply(safe_float).values

        col_idx = find_column_index("N(he)ospm", ospm_header_text, print_results)
        if col_idx != -1:
            NNt_ospm_orig[:] = ospm_working_df.iloc[:, col_idx].apply(safe_float).values

        col_idx = find_column_index(
            "V_veh(li)ospm(km/hr)", ospm_header_text, print_results
        )
        if col_idx != -1:
            Vp_ospm_orig[:] = ospm_working_df.iloc[:, col_idx].apply(safe_float).values

        col_idx = find_column_index(
            "V_veh(he)ospm(km/hr)", ospm_header_text, print_results
        )
        if col_idx != -1:
            Vt_ospm_orig[:] = ospm_working_df.iloc[:, col_idx].apply(safe_float).values

        col_idx = find_column_index(
            "Cemisospm(ug/m/s)", ospm_header_text, print_results
        )
        if col_idx != -1:
            qNOX_ospm_orig[:] = (
                ospm_working_df.iloc[:, col_idx].apply(safe_float).values
            )

        # Clean OSPM data - replace negative values and NaN with nodata
        for data_array in [
            U_mast_ospm_orig,
            wind_dir_ospm_orig,
            TK_ospm_orig,
            GlobalRad_ospm_orig,
            cNOx_b_ospm_orig,
            qNOX_ospm_orig,
            NNp_ospm_orig,
            NNt_ospm_orig,
            Vp_ospm_orig,
            Vt_ospm_orig,
        ]:
            data_array[(data_array < 0) | np.isnan(data_array)] = nodata

    # Store all data in the dataclass
    loaded_airquality.PM_obs = PM_obs
    loaded_airquality.PM_background = PM_background
    loaded_airquality.NOX_obs = NOX_obs
    loaded_airquality.NOX_background = NOX_background
    loaded_airquality.NOX_emis = NOX_emis
    loaded_airquality.EP_emis = EP_emis
    loaded_airquality.Salt_obs = Salt_obs
    loaded_airquality.f_dis_input = f_dis_input

    # Store availability flags
    loaded_airquality.NOX_emis_available = NOX_emis_available
    loaded_airquality.EP_emis_available = EP_emis_available
    loaded_airquality.Salt_obs_available = Salt_obs_available
    loaded_airquality.f_dis_available = f_dis_available

    # Store calculated net concentrations
    loaded_airquality.PM_obs_net = PM_obs_net
    loaded_airquality.PM_obs_bg = PM_obs_bg
    loaded_airquality.NOX_obs_net = NOX_obs_net

    # Store OSPM data
    loaded_airquality.OSPM_data_exists = OSPM_data_exists
    loaded_airquality.U_mast_ospm_orig = U_mast_ospm_orig
    loaded_airquality.wind_dir_ospm_orig = wind_dir_ospm_orig
    loaded_airquality.TK_ospm_orig = TK_ospm_orig
    loaded_airquality.GlobalRad_ospm_orig = GlobalRad_ospm_orig
    loaded_airquality.cNOx_b_ospm_orig = cNOx_b_ospm_orig
    loaded_airquality.qNOX_ospm_orig = qNOX_ospm_orig
    loaded_airquality.NNp_ospm_orig = NNp_ospm_orig
    loaded_airquality.NNt_ospm_orig = NNt_ospm_orig
    loaded_airquality.Vp_ospm_orig = Vp_ospm_orig
    loaded_airquality.Vt_ospm_orig = Vt_ospm_orig

    logger.info(
        f"Successfully loaded air quality data with {loaded_airquality.n_date} records"
    )
    if OSPM_data_exists:
        logger.info("OSPM data also loaded successfully")

    return loaded_airquality
