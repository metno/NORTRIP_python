import numpy as np


def forward_fill_missing(data: np.ndarray, nodata: float) -> tuple[np.ndarray, list]:
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
