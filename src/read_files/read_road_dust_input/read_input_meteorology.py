import pandas as pd
import numpy as np
import logging
from input_classes import input_meteorology

logger = logging.getLogger(__name__)


def _find_column_index(
    search_text: str, header_text: pd.Series, print_results: bool = False
) -> int:
    """Find column index for given search text, handling duplicates."""
    matches = header_text.str.contains(search_text, case=False, na=False, regex=False)
    if matches.sum() > 1 and print_results:
        logger.warning(
            f"Double occurrence of input data header '{search_text}': USING THE FIRST"
        )
    if matches.any():
        return matches.argmax()
    return -1


def _check_data_availability(data: np.ndarray, nodata: float) -> tuple[bool, list]:
    """Check if data is available (not all nodata/NaN) and return missing indices."""
    missing_indices = np.where((data == nodata) | np.isnan(data))[0]
    is_available = len(missing_indices) < len(data)
    return is_available, missing_indices.tolist()


def _forward_fill_missing(
    data: np.ndarray, missing_indices: list, nodata: float
) -> np.ndarray:
    """Forward fill missing values in data array."""
    data_copy = data.copy()
    for i in missing_indices:
        if i > 0:
            data_copy[i] = data_copy[i - 1]
    return data_copy


def _dewpoint_from_RH(T_a: np.ndarray, RH: np.ndarray) -> np.ndarray:
    """Calculate dewpoint temperature from air temperature and relative humidity."""
    # Using Magnus formula approximation
    # T_dewpoint = T_a - ((100 - RH) / 5)  # Simple approximation
    # More accurate Magnus formula:
    # Avoid divide by zero and negative/zero RH values
    RH_safe = np.clip(RH, 0.1, 100.0)  # Clip to avoid log(0) and ensure positive values
    gamma = np.log(RH_safe / 100.0) + (17.625 * T_a) / (243.04 + T_a)
    T_dewpoint = (243.04 * gamma) / (17.625 - gamma)
    return T_dewpoint


def _RH_from_dewpoint(T_a: np.ndarray, T_dewpoint: np.ndarray) -> np.ndarray:
    """Calculate relative humidity from air temperature and dewpoint temperature."""
    # Using Magnus formula
    gamma_a = (17.625 * T_a) / (243.04 + T_a)
    gamma_d = (17.625 * T_dewpoint) / (243.04 + T_dewpoint)
    RH = 100.0 * np.exp(gamma_d - gamma_a)
    return np.clip(RH, 0, 100)  # Clip to valid range


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
    header_text = meteorology_df.iloc[0, :].astype(str)

    # Skip header row and get data
    data_df = meteorology_df.iloc[1:].reset_index(drop=True)
    n_meteo = len(data_df)
    loaded_meteo.n_meteo = n_meteo

    if n_meteo == 0:
        logger.warning("No meteorological data found")
        return loaded_meteo

    # Read core required data
    # Air temperature (T2m)
    col_idx = _find_column_index("T2m", header_text, print_results)
    if col_idx == -1:
        logger.error("Required field 'T2m' not found in meteorology data")
        return loaded_meteo
    loaded_meteo.T_a = data_df.iloc[:, col_idx].values.astype(float)

    # Wind speed (FF)
    col_idx = _find_column_index("FF", header_text, print_results)
    if col_idx == -1:
        logger.error("Required field 'FF' not found in meteorology data")
        return loaded_meteo
    loaded_meteo.FF = data_df.iloc[:, col_idx].values.astype(float)

    # Rain precipitation
    col_idx = _find_column_index("Rain", header_text, print_results)
    if col_idx == -1:
        logger.error("Required field 'Rain' not found in meteorology data")
        return loaded_meteo
    loaded_meteo.Rain = data_df.iloc[:, col_idx].values.astype(float)

    # Snow precipitation
    col_idx = _find_column_index("Snow", header_text, print_results)
    if col_idx == -1:
        logger.error("Required field 'Snow' not found in meteorology data")
        return loaded_meteo
    loaded_meteo.Snow = data_df.iloc[:, col_idx].values.astype(float)

    # Read optional data with availability flags
    # Alternative air temperature (T_a_alt)
    col_idx = _find_column_index("T_a_alt", header_text, print_results)
    if col_idx != -1:
        loaded_meteo.T2_a = data_df.iloc[:, col_idx].values.astype(float)
        loaded_meteo.T2_a_available = 1
    else:
        loaded_meteo.T2_a = np.zeros(n_meteo)
        loaded_meteo.T2_a_available = 0

    # Wind direction (DD)
    col_idx = _find_column_index("DD", header_text, print_results)
    if col_idx != -1:
        loaded_meteo.DD = data_df.iloc[:, col_idx].values.astype(float)
        loaded_meteo.DD_available = 1
    else:
        loaded_meteo.DD = np.zeros(n_meteo)
        loaded_meteo.DD_available = 0

    # Relative humidity (RH)
    col_idx = _find_column_index("RH", header_text, print_results)
    if col_idx != -1:
        loaded_meteo.RH = data_df.iloc[:, col_idx].values.astype(float)
        loaded_meteo.RH_available = 1
    else:
        loaded_meteo.RH = np.zeros(n_meteo)
        loaded_meteo.RH_available = 0

    # Dewpoint temperature (T2m dewpoint)
    col_idx = _find_column_index("T2m dewpoint", header_text, print_results)
    if col_idx != -1:
        loaded_meteo.T_dewpoint = data_df.iloc[:, col_idx].values.astype(float)
        loaded_meteo.T_dewpoint_available = 1
    else:
        loaded_meteo.T_dewpoint = np.zeros(n_meteo)
        loaded_meteo.T_dewpoint_available = 0

    # Global radiation
    col_idx = _find_column_index("Global radiation", header_text, print_results)
    if col_idx != -1:
        loaded_meteo.short_rad_in = data_df.iloc[:, col_idx].values.astype(float)
        loaded_meteo.short_rad_in_available = 1
    else:
        loaded_meteo.short_rad_in = np.zeros(n_meteo)
        loaded_meteo.short_rad_in_available = 0

    # Longwave radiation
    col_idx = _find_column_index("Longwave radiation", header_text, print_results)
    if col_idx != -1:
        loaded_meteo.long_rad_in = data_df.iloc[:, col_idx].values.astype(float)
        loaded_meteo.long_rad_in_available = 1
    else:
        loaded_meteo.long_rad_in = np.zeros(n_meteo)
        loaded_meteo.long_rad_in_available = 0

    # Cloud cover
    col_idx = _find_column_index("Cloud cover", header_text, print_results)
    if col_idx != -1:
        loaded_meteo.cloud_cover = data_df.iloc[:, col_idx].values.astype(float)
        loaded_meteo.cloud_cover_available = 1
    else:
        loaded_meteo.cloud_cover = np.zeros(n_meteo)
        loaded_meteo.cloud_cover_available = 0

    # Road wetness
    col_idx = _find_column_index("Road wetness", header_text, print_results)
    if col_idx != -1:
        loaded_meteo.road_wetness_obs = data_df.iloc[:, col_idx].values.astype(float)
        loaded_meteo.road_wetness_obs_available = 1

        # Check if road wetness is in mm units
        header_str = str(header_text.iloc[col_idx])
        if "(mm)" in header_str:
            loaded_meteo.road_wetness_obs_in_mm = 1
    else:
        loaded_meteo.road_wetness_obs = np.full(n_meteo, np.nan)
        loaded_meteo.road_wetness_obs_available = 0

    # Road surface temperature
    col_idx = _find_column_index("Road surface temperature", header_text, print_results)
    if col_idx != -1:
        loaded_meteo.road_temperature_obs = data_df.iloc[:, col_idx].values.astype(
            float
        )
        loaded_meteo.road_temperature_obs_available = 1
    else:
        loaded_meteo.road_temperature_obs = np.full(n_meteo, np.nan)
        loaded_meteo.road_temperature_obs_available = 0

    # Atmospheric pressure
    col_idx = _find_column_index("Pressure", header_text, print_results)
    if col_idx != -1:
        loaded_meteo.Pressure_a = data_df.iloc[:, col_idx].values.astype(float)
        loaded_meteo.pressure_obs_available = 1
    else:
        loaded_meteo.Pressure_a = np.full(n_meteo, pressure_default)
        loaded_meteo.pressure_obs_available = 1

    # Subsurface temperature
    col_idx = _find_column_index("T subsurface", header_text, print_results)
    if col_idx != -1:
        loaded_meteo.T_sub = data_df.iloc[:, col_idx].values.astype(float)
        loaded_meteo.T_sub_available = 1
    else:
        loaded_meteo.T_sub = np.full(n_meteo, nodata)
        loaded_meteo.T_sub_available = 0

    # Handle missing data and forward fill for core variables
    # Air temperature
    missing_indices = np.where(
        (loaded_meteo.T_a == nodata) | np.isnan(loaded_meteo.T_a)
    )[0]
    loaded_meteo.T_a_nodata = missing_indices.tolist()
    if len(missing_indices) > 0:
        loaded_meteo.T_a = _forward_fill_missing(
            loaded_meteo.T_a, missing_indices.tolist(), nodata
        )

    # Wind speed
    missing_indices = np.where((loaded_meteo.FF == nodata) | np.isnan(loaded_meteo.FF))[
        0
    ]
    loaded_meteo.FF_nodata = missing_indices.tolist()
    if len(missing_indices) > 0:
        loaded_meteo.FF = _forward_fill_missing(
            loaded_meteo.FF, missing_indices.tolist(), nodata
        )

    # Apply wind speed correction
    loaded_meteo.FF = loaded_meteo.FF * wind_speed_correction

    # Rain precipitation - remove negative values
    missing_indices = np.where(
        (loaded_meteo.Rain == nodata) | np.isnan(loaded_meteo.Rain)
    )[0]
    loaded_meteo.Rain_nodata = missing_indices.tolist()
    if len(missing_indices) > 0:
        loaded_meteo.Rain = _forward_fill_missing(
            loaded_meteo.Rain, missing_indices.tolist(), nodata
        )
    loaded_meteo.Rain = np.maximum(loaded_meteo.Rain, 0)

    # Snow precipitation - remove negative values
    missing_indices = np.where(
        (loaded_meteo.Snow == nodata) | np.isnan(loaded_meteo.Snow)
    )[0]
    loaded_meteo.Snow_nodata = missing_indices.tolist()
    if len(missing_indices) > 0:
        loaded_meteo.Snow = _forward_fill_missing(
            loaded_meteo.Snow, missing_indices.tolist(), nodata
        )
    loaded_meteo.Snow = np.maximum(loaded_meteo.Snow, 0)

    # Handle optional data availability and missing values
    # Check data availability and handle missing values for optional fields
    optional_fields = [
        ("DD", "DD_available", "DD_nodata"),
        ("T2_a", "T2_a_available", "T2_a_nodata"),
        ("T_sub", "T_sub_available", "T_sub_nodata"),
        ("short_rad_in", "short_rad_in_available", "short_rad_in_missing"),
        ("long_rad_in", "long_rad_in_available", "long_rad_in_missing"),
        ("cloud_cover", "cloud_cover_available", "cloud_cover_missing"),
        ("road_wetness_obs", "road_wetness_obs_available", "road_wetness_obs_missing"),
        (
            "road_temperature_obs",
            "road_temperature_obs_available",
            "road_temperature_obs_missing",
        ),
        ("Pressure_a", "pressure_obs_available", "pressure_obs_missing"),
    ]

    for data_field, avail_field, missing_field in optional_fields:
        if getattr(loaded_meteo, avail_field):
            data_array = getattr(loaded_meteo, data_field)
            is_available, missing_indices = _check_data_availability(data_array, nodata)

            if not is_available:
                setattr(loaded_meteo, avail_field, 0)

            setattr(loaded_meteo, missing_field, missing_indices)

            # Forward fill if data is available but has some missing values
            if is_available and len(missing_indices) > 0:
                filled_data = _forward_fill_missing(data_array, missing_indices, nodata)
                setattr(loaded_meteo, data_field, filled_data)

    # Handle RH - remove negative values and forward fill
    if loaded_meteo.RH_available:
        missing_indices = np.where(
            (loaded_meteo.RH == nodata) | np.isnan(loaded_meteo.RH)
        )[0]
        loaded_meteo.RH_nodata = missing_indices.tolist()
        if len(missing_indices) > 0:
            loaded_meteo.RH = _forward_fill_missing(
                loaded_meteo.RH, missing_indices.tolist(), nodata
            )
        loaded_meteo.RH = np.maximum(loaded_meteo.RH, 0)

    # Handle special case for pressure - if all nodata, use default
    if getattr(loaded_meteo, "pressure_obs_available") == 0:
        loaded_meteo.Pressure_a[:] = pressure_default

    # Calculate road wetness statistics if available
    if loaded_meteo.road_wetness_obs_available:
        valid_data = loaded_meteo.road_wetness_obs[
            (loaded_meteo.road_wetness_obs != nodata)
            & (~np.isnan(loaded_meteo.road_wetness_obs))
        ]
        if len(valid_data) > 0:
            loaded_meteo.max_road_wetness_obs = float(np.max(valid_data))
            loaded_meteo.min_road_wetness_obs = float(np.min(valid_data))
            loaded_meteo.mean_road_wetness_obs = float(np.mean(valid_data))

    # Handle RH and dewpoint temperature cross-calculation
    if loaded_meteo.RH_available and not loaded_meteo.T_dewpoint_available:
        loaded_meteo.T_dewpoint = _dewpoint_from_RH(loaded_meteo.T_a, loaded_meteo.RH)

    if loaded_meteo.T_dewpoint_available and not loaded_meteo.RH_available:
        loaded_meteo.RH = _RH_from_dewpoint(loaded_meteo.T_a, loaded_meteo.T_dewpoint)

    logger.info(
        f"Successfully loaded meteorological data with {loaded_meteo.n_meteo} records"
    )
    return loaded_meteo
