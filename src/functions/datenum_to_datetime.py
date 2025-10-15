from datetime import datetime
import numpy as np


def datenum_to_datetime(datenum: np.float64) -> datetime:
    """Convert datenum to Python datetime."""

    dt = datetime.fromtimestamp(datenum)

    return dt
