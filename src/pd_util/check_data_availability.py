import numpy as np


def check_data_availability(
    data: np.ndarray, available: int, nodata: float
) -> tuple[np.ndarray, int, list]:
    """
    Check if data is available (not all nodata values).

    Args:
        data: Data array to check
        available: Current availability flag
        nodata: Nodata value

    Returns:
        tuple: (data, updated_availability_flag, missing_indices)
    """
    if not available:
        return data, 0, []

    missing_indices = np.where((data == nodata) | np.isnan(data))[0]

    if len(missing_indices) == len(data):
        # All data is missing
        return data, 0, missing_indices.tolist()

    return data, available, missing_indices.tolist()
