import pandas as pd
import numpy as np
import logging
from input_classes import input_meteorology
from pd_util import find_column_index, safe_float, check_data_availability

logger = logging.getLogger(__name__)


def _forward_fill_missing(data: np.ndarray, nodata: float) -> tuple[np.ndarray, list]:
    """
    Forward fill missing data and return indices of originally missing values.

    Args:
        data: Data array to fill
        nodata: Nodata value

    Returns:
        tuple: (filled_data, missing_indices)
    """
    missing_indices = np.where((data == nodata) | np.isnan(data))[0]

    for i in missing_indices:
        if i > 0:
            data[i] = data[i - 1]

    return data, missing_indices.tolist()


def _dewpoint_from_rh(temperature: np.ndarray, rh: np.ndarray) -> np.ndarray:
    """
    Calculate dewpoint temperature from air temperature and relative humidity.

    Args:
        temperature: Air temperature in Celsius
        rh: Relative humidity in %

    Returns:
        np.ndarray: Dewpoint temperature in Celsius
    """
    # Magnus formula approximation
    a = 17.27
    b = 237.7

    # Ensure RH is between 0 and 100
    rh = np.clip(rh, 0.01, 100)

    alpha = (a * temperature) / (b + temperature) + np.log(rh / 100.0)
    dewpoint = (b * alpha) / (a - alpha)

    return dewpoint


def _rh_from_dewpoint(temperature: np.ndarray, dewpoint: np.ndarray) -> np.ndarray:
    """
    Calculate relative humidity from air temperature and dewpoint temperature.

    Args:
        temperature: Air temperature in Celsius
        dewpoint: Dewpoint temperature in Celsius

    Returns:
        np.ndarray: Relative humidity in %
    """
    # Magnus formula approximation
    a = 17.27
    b = 237.7

    alpha_t = (a * temperature) / (b + temperature)
    alpha_d = (a * dewpoint) / (b + dewpoint)

    rh = 100.0 * np.exp(alpha_d - alpha_t)

    return np.clip(rh, 0, 100)


def read_input_meteorology(
    meteorology_df: pd.DataFrame,
    nodata: float = -99.0,
    wind_speed_correction: float = 1.0,
    pressure_default: float = 101325.0,
    print_results: bool = False,
) -> input_meteorology:
    """
    Read meteorological input data from a pandas DataFrame.

    Args:
        meteorology_df (pd.DataFrame): DataFrame containing the meteorological data
        nodata (float): Nodata value to use for missing data
        wind_speed_correction (float): Correction factor for wind speed
        pressure_default (float): Default pressure value if not available
        print_results (bool): Whether to print the results to the console

    Returns:
        input_meteorology: Dataclass containing the meteorological data
    """
    loaded_meteo = input_meteorology()

    # Clean and set headers - first row contains headers
    header_text = (
        meteorology_df.iloc[0, :].astype(str).str.replace(r"\s+", "", regex=True)
    )
    meteorology_df.columns = header_text
    meteorology_df = meteorology_df.iloc[1:].reset_index(drop=True)

    n_meteo = len(meteorology_df)
    loaded_meteo.n_meteo = n_meteo

    # Read T2m (required)
    col_idx = find_column_index("T2m", header_text, print_results)
    if col_idx == -1:
        logger.error("No T2m temperature data found - T2m column missing")
        loaded_meteo.n_meteo = 0
        return loaded_meteo

    T_a_data = meteorology_df.iloc[:, col_idx].apply(safe_float).values

    # Read T_a_alt (optional)
    col_idx = find_column_index("T_a_alt", header_text, print_results)
    if col_idx != -1:
        T2_a_data = meteorology_df.iloc[:, col_idx].apply(safe_float).values
        loaded_meteo.T2_a_available = 1
    else:
        T2_a_data = np.zeros_like(T_a_data)
        loaded_meteo.T2_a_available = 0

    # Read FF (required)
    col_idx = find_column_index("FF", header_text, print_results)
    if col_idx == -1:
        logger.error("No wind speed data found - FF column missing")
        loaded_meteo.n_meteo = 0
        return loaded_meteo

    FF_data = meteorology_df.iloc[:, col_idx].apply(safe_float).values

    # Read DD (optional)
    col_idx = find_column_index("DD", header_text, print_results)
    if col_idx != -1:
        DD_data = meteorology_df.iloc[:, col_idx].apply(safe_float).values
        loaded_meteo.DD_available = 1
    else:
        DD_data = np.zeros_like(FF_data)
        loaded_meteo.DD_available = 0

    # Read RH (optional)
    col_idx = find_column_index("RH", header_text, print_results)
    if col_idx != -1:
        RH_data = meteorology_df.iloc[:, col_idx].apply(safe_float).values
        loaded_meteo.RH_available = 1
    else:
        RH_data = np.full_like(T_a_data, nodata)
        loaded_meteo.RH_available = 0

    # Read T2m dewpoint (optional) - search for cleaned header
    col_idx = find_column_index("T2mdewpoint", header_text, print_results)
    if col_idx != -1:
        T_dewpoint_data = meteorology_df.iloc[:, col_idx].apply(safe_float).values
        loaded_meteo.T_dewpoint_available = 1
    else:
        T_dewpoint_data = np.full_like(T_a_data, nodata)
        loaded_meteo.T_dewpoint_available = 0

    # Read Rain (required)
    col_idx = find_column_index("Rain", header_text, print_results)
    if col_idx == -1:
        logger.error("No rain data found - Rain column missing")
        loaded_meteo.n_meteo = 0
        return loaded_meteo

    Rain_data = meteorology_df.iloc[:, col_idx].apply(safe_float).values

    # Read Snow (required)
    col_idx = find_column_index("Snow", header_text, print_results)
    if col_idx == -1:
        logger.error("No snow data found - Snow column missing")
        loaded_meteo.n_meteo = 0
        return loaded_meteo

    Snow_data = meteorology_df.iloc[:, col_idx].apply(safe_float).values

    # Read Global radiation (optional) - search for cleaned header
    col_idx = find_column_index("Globalradiation", header_text, print_results)
    if col_idx != -1:
        short_rad_in_data = meteorology_df.iloc[:, col_idx].apply(safe_float).values
        loaded_meteo.short_rad_in_available = 1
    else:
        short_rad_in_data = np.zeros_like(T_a_data)
        loaded_meteo.short_rad_in_available = 0

    # Read Longwave radiation (optional) - search for cleaned header
    col_idx = find_column_index("Longwaveradiation", header_text, print_results)
    if col_idx != -1:
        long_rad_in_data = meteorology_df.iloc[:, col_idx].apply(safe_float).values
        loaded_meteo.long_rad_in_available = 1
    else:
        long_rad_in_data = np.zeros_like(T_a_data)
        loaded_meteo.long_rad_in_available = 0

    # Read Cloud cover (optional) - search for cleaned header
    col_idx = find_column_index("Cloudcover", header_text, print_results)
    if col_idx != -1:
        cloud_cover_data = meteorology_df.iloc[:, col_idx].apply(safe_float).values
        loaded_meteo.cloud_cover_available = 1
    else:
        cloud_cover_data = np.zeros_like(T_a_data)
        loaded_meteo.cloud_cover_available = 0

    # Read Road wetness (optional) - search for cleaned header with substring match
    col_idx = find_column_index(
        "Roadwetness", header_text, print_results, exact_match=False
    )
    if col_idx != -1:
        road_wetness_obs_data = meteorology_df.iloc[:, col_idx].apply(safe_float).values
        loaded_meteo.road_wetness_obs_available = 1

        # Check if road wetness is in mm - check the cleaned header since we already found it
        header_str = str(header_text.iloc[col_idx])
        if "(mm)" in header_str:
            loaded_meteo.road_wetness_obs_in_mm = 1
        else:
            loaded_meteo.road_wetness_obs_in_mm = 0
    else:
        road_wetness_obs_data = np.full_like(T_a_data, np.nan)
        loaded_meteo.road_wetness_obs_available = 0
        loaded_meteo.road_wetness_obs_in_mm = 0

    # Read Road surface temperature (optional) - search for cleaned header
    col_idx = find_column_index("Roadsurfacetemperature", header_text, print_results)
    if col_idx != -1:
        road_temperature_obs_data = (
            meteorology_df.iloc[:, col_idx].apply(safe_float).values
        )
        loaded_meteo.road_temperature_obs_available = 1
    else:
        road_temperature_obs_data = np.full_like(T_a_data, np.nan)
        loaded_meteo.road_temperature_obs_available = 0

    # Read Pressure (optional)
    col_idx = find_column_index("Pressure", header_text, print_results)
    if col_idx != -1:
        Pressure_a_data = meteorology_df.iloc[:, col_idx].apply(safe_float).values
        loaded_meteo.pressure_obs_available = 1
    else:
        Pressure_a_data = np.full(n_meteo, pressure_default)
        loaded_meteo.pressure_obs_available = 1

    # Read T subsurface (optional) - search for cleaned header
    col_idx = find_column_index("Tsubsurface", header_text, print_results)
    if col_idx != -1:
        T_sub_data = meteorology_df.iloc[:, col_idx].apply(safe_float).values
        loaded_meteo.T_sub_available = 1
    else:
        T_sub_data = np.full_like(T_a_data, nodata)
        loaded_meteo.T_sub_available = 0

    # Process and fill missing data

    # Handle T_a
    T_a_data, loaded_meteo.T_a_nodata = _forward_fill_missing(T_a_data, nodata)

    # Handle FF and apply wind speed correction
    FF_data, loaded_meteo.FF_nodata = _forward_fill_missing(FF_data, nodata)
    FF_data = FF_data * wind_speed_correction

    # Handle RH
    if loaded_meteo.RH_available:
        RH_data, loaded_meteo.RH_nodata = _forward_fill_missing(RH_data, nodata)
        # Remove negative values
        RH_data = np.maximum(RH_data, 0)

    # Handle Rain
    Rain_data, loaded_meteo.Rain_nodata = _forward_fill_missing(Rain_data, nodata)
    # Remove negative values
    Rain_data = np.maximum(Rain_data, 0)

    # Handle Snow
    Snow_data, loaded_meteo.Snow_nodata = _forward_fill_missing(Snow_data, nodata)
    # Remove negative values
    Snow_data = np.maximum(Snow_data, 0)

    # Handle optional data with availability checks
    if loaded_meteo.DD_available:
        DD_data, loaded_meteo.DD_available, loaded_meteo.DD_nodata = (
            check_data_availability(DD_data, loaded_meteo.DD_available, nodata)
        )
        if loaded_meteo.DD_available:
            DD_data, _ = _forward_fill_missing(DD_data, nodata)

    if loaded_meteo.T2_a_available:
        T2_a_data, loaded_meteo.T2_a_available, loaded_meteo.T2_a_nodata = (
            check_data_availability(T2_a_data, loaded_meteo.T2_a_available, nodata)
        )
        if loaded_meteo.T2_a_available:
            T2_a_data, _ = _forward_fill_missing(T2_a_data, nodata)

    if loaded_meteo.T_sub_available:
        T_sub_data, loaded_meteo.T_sub_available, loaded_meteo.T_sub_nodata = (
            check_data_availability(T_sub_data, loaded_meteo.T_sub_available, nodata)
        )
        if loaded_meteo.T_sub_available:
            T_sub_data, _ = _forward_fill_missing(T_sub_data, nodata)

    # Check data availability for optional fields
    if loaded_meteo.short_rad_in_available:
        (
            short_rad_in_data,
            loaded_meteo.short_rad_in_available,
            loaded_meteo.short_rad_in_missing,
        ) = check_data_availability(
            short_rad_in_data, loaded_meteo.short_rad_in_available, nodata
        )

    if loaded_meteo.long_rad_in_available:
        (
            long_rad_in_data,
            loaded_meteo.long_rad_in_available,
            loaded_meteo.long_rad_in_missing,
        ) = check_data_availability(
            long_rad_in_data, loaded_meteo.long_rad_in_available, nodata
        )

    if loaded_meteo.cloud_cover_available:
        (
            cloud_cover_data,
            loaded_meteo.cloud_cover_available,
            loaded_meteo.cloud_cover_missing,
        ) = check_data_availability(
            cloud_cover_data, loaded_meteo.cloud_cover_available, nodata
        )

    if loaded_meteo.road_wetness_obs_available:
        (
            road_wetness_obs_data,
            loaded_meteo.road_wetness_obs_available,
            loaded_meteo.road_wetness_obs_missing,
        ) = check_data_availability(
            road_wetness_obs_data, loaded_meteo.road_wetness_obs_available, nodata
        )

    if loaded_meteo.road_temperature_obs_available:
        (
            road_temperature_obs_data,
            loaded_meteo.road_temperature_obs_available,
            loaded_meteo.road_temperature_obs_missing,
        ) = check_data_availability(
            road_temperature_obs_data,
            loaded_meteo.road_temperature_obs_available,
            nodata,
        )

    if loaded_meteo.pressure_obs_available:
        (
            Pressure_a_data,
            loaded_meteo.pressure_obs_available,
            loaded_meteo.pressure_obs_missing,
        ) = check_data_availability(
            Pressure_a_data, loaded_meteo.pressure_obs_available, nodata
        )

    # If pressure is all nodata, use default
    if not loaded_meteo.pressure_obs_available:
        Pressure_a_data[:] = pressure_default
        loaded_meteo.pressure_obs_available = 1

    # Calculate road wetness statistics
    if loaded_meteo.road_wetness_obs_available:
        valid_wetness = road_wetness_obs_data[
            ~np.isnan(road_wetness_obs_data) & (road_wetness_obs_data != nodata)
        ]
        if len(valid_wetness) > 0:
            loaded_meteo.max_road_wetness_obs = float(np.max(valid_wetness))
            loaded_meteo.min_road_wetness_obs = float(np.min(valid_wetness))
            loaded_meteo.mean_road_wetness_obs = float(np.mean(valid_wetness))
        else:
            loaded_meteo.max_road_wetness_obs = np.nan
            loaded_meteo.min_road_wetness_obs = np.nan
            loaded_meteo.mean_road_wetness_obs = np.nan
    else:
        loaded_meteo.max_road_wetness_obs = np.nan
        loaded_meteo.min_road_wetness_obs = np.nan
        loaded_meteo.mean_road_wetness_obs = np.nan

    # Calculate missing dewpoint or RH if one is available but not the other
    if loaded_meteo.RH_available and not loaded_meteo.T_dewpoint_available:
        T_dewpoint_data = _dewpoint_from_rh(T_a_data, RH_data)
        loaded_meteo.T_dewpoint_available = 1

    if loaded_meteo.T_dewpoint_available and not loaded_meteo.RH_available:
        RH_data = _rh_from_dewpoint(T_a_data, T_dewpoint_data)
        loaded_meteo.RH_available = 1

    # Store data in the correct format (2D arrays as expected by the dataclass)
    loaded_meteo.T_a = np.array([T_a_data, []], dtype=object)
    loaded_meteo.T2_a = np.array([T2_a_data, []], dtype=object)
    loaded_meteo.FF = np.array([FF_data, []], dtype=object)
    loaded_meteo.DD = np.array([DD_data, []], dtype=object)
    loaded_meteo.RH = np.array([RH_data, []], dtype=object)
    loaded_meteo.T_dewpoint = np.array([T_dewpoint_data, []], dtype=object)
    loaded_meteo.Rain = np.array([Rain_data, []], dtype=object)
    loaded_meteo.Snow = np.array([Snow_data, []], dtype=object)
    loaded_meteo.short_rad_in = np.array([short_rad_in_data, []], dtype=object)
    loaded_meteo.long_rad_in = np.array([long_rad_in_data, []], dtype=object)
    loaded_meteo.cloud_cover = np.array([cloud_cover_data, []], dtype=object)
    loaded_meteo.road_wetness_obs = np.array([road_wetness_obs_data, []], dtype=object)
    loaded_meteo.road_temperature_obs = np.array(
        [road_temperature_obs_data, []], dtype=object
    )
    loaded_meteo.Pressure_a = np.array([Pressure_a_data, []], dtype=object)
    loaded_meteo.T_sub = np.array([T_sub_data, []], dtype=object)

    logger.info(
        f"Successfully loaded meteorological data with {loaded_meteo.n_meteo} records"
    )
    return loaded_meteo
