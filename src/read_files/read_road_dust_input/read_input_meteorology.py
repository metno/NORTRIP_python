import pandas as pd
import numpy as np
import logging
from input_classes import input_meteorology
from pd_util import find_column_index, safe_float

logger = logging.getLogger(__name__)


def _safe_convert_to_float(data_series: pd.Series) -> np.ndarray:
    """Safely convert a pandas Series to float array, handling whitespace and invalid values."""
    return np.array([safe_float(val) for val in data_series])


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

    logger.info(
        f"Successfully loaded meteorological data with {loaded_meteo.n_meteo} records"
    )
    return loaded_meteo
