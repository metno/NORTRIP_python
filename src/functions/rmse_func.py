import numpy as np


def rmse_func(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calculate Root Mean Square Error (RMSE) between two arrays.

    Args:
        a: First array
        b: Second array

    Returns:
        float: RMSE value, or NaN if arrays are incompatible or empty
    """
    # Check if arrays have compatible dimensions and are not empty
    if len(a) == len(b) and len(a) > 0 and len(b) > 0:
        # Find indices where both arrays have valid (non-NaN) values
        valid_indices = ~np.isnan(a) & ~np.isnan(b)

        if np.any(valid_indices):
            # Calculate RMSE using only valid data points
            valid_a = a[valid_indices]
            valid_b = b[valid_indices]
            result = np.sqrt(np.sum((valid_a - valid_b) ** 2) / len(valid_a))
        else:
            result = np.nan
    else:
        result = np.nan

    return result
