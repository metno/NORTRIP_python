import pandas as pd
import numpy as np
import logging
from input_classes import input_meteorology
from pd_util import find_column_index, safe_float

logger = logging.getLogger(__name__)


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
