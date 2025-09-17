import pandas as pd
import numpy as np
from .safe_float import safe_float


def apply_safe_float(
    df: pd.DataFrame,
    col_idx: int,
    nodata: float = -99.0
) -> np.ndarray:
    """
    Apply safe_float conversion to a DataFrame column and return values as numpy array.

    This function encapsulates the common pattern of:
    df.iloc[:, col_idx].apply(lambda x: safe_float(x, nodata)).values

    Args:
        df: DataFrame containing the data
        col_idx: Column index to process
        nodata: Nodata value to use for missing/invalid values

    Returns:
        np.ndarray: Array of float values with safe_float conversion applied
    """
    return df.iloc[:, col_idx].apply(lambda x: safe_float(x, nodata)).values
