import numpy as np
from typing import Tuple


def check_data_func(
    val: np.ndarray, available: int, nodata: float
) -> Tuple[np.ndarray, int, list]:
    """
    Check data for missing values and fill gaps using forward fill method.

    Args:
        val: Data array to check and potentially modify
        available: Flag indicating if data is available (1=available, 0=not available)
        nodata: Value representing missing data

    Returns:
        tuple: (val, available, missing_flag) where:
            val: Modified data array with gaps filled
            available: Updated availability flag
            missing_flag: List of indices where data was missing
    """
    missing_flag = []
    val = val.copy()  # Create a copy to avoid modifying the original

    if available == 1:
        # Find indices where data is missing (nodata or NaN)
        missing_indices = np.where((val == nodata) | np.isnan(val))[0]

        if len(missing_indices) > 0:
            if len(missing_indices) == len(val):
                # All data is missing
                available = 0
            else:
                # Find indices where data is valid
                valid_indices = np.where((val != nodata) & ~np.isnan(val))[0]

                if len(valid_indices) > 0:
                    # Fill gaps backwards if the first values are missing
                    if valid_indices[0] != 0:
                        val[0 : valid_indices[0]] = val[valid_indices[0]]

                    # Forward fill missing values
                    for i, missing_idx in enumerate(missing_indices):
                        if missing_idx > 0:
                            val[missing_idx] = val[missing_idx - 1]
                            missing_flag.append(missing_idx)

                    # Handle nodata in the first position
                    if len(missing_indices) > 0 and missing_indices[0] == 0:
                        if len(val) > 1:
                            val[0] = val[1]
                            if 0 not in missing_flag:
                                missing_flag.insert(0, 0)

    return val, available, missing_flag
