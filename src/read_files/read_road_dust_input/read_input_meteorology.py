import pandas as pd
import numpy as np
from datetime import datetime
from input_classes import input_meteorology
from pd_util import find_float_or_default, find_str_or_default


def read_input_meteorology(
    meteorology_df: pd.DataFrame, nodata: float = -99, print_results=False
) -> input_meteorology:
    """
    Read meteorological input data from a pandas DataFrame.

    Args:
        meteorology_df (pd.DataFrame): DataFrame containing the meteorological data
        nodata (float): Nodata value to use for missing data
        print_results (bool): Whether to print the results to the console

    Returns:
        input_meteorology: Dataclass containing the meteorological data
    """

    loaded = input_meteorology()

    return loaded
