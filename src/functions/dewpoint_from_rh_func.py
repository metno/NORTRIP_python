import numpy as np
import logging

logger = logging.getLogger(__name__)


def dewpoint_from_rh_func(TC: float, RH: float) -> float:
    """
    Calculate dewpoint temperature from air temperature and relative humidity.

    Args:
        TC: Air temperature in Celsius
        RH: Relative humidity in %

    Returns:
        Dewpoint temperature in Celsius
    """
    # Constants for Magnus formula
    a = 6.1121
    b = 17.67
    c = 243.5

    # Saturation vapor pressure at air temperature
    esat = a * np.exp(b * TC / (c + TC))

    # Actual vapor pressure
    eair = RH / 100.0 * esat

    if eair <= 0.0:
        eair = 0.0001

    # Dewpoint temperature
    TC_dewpoint = c * np.log(eair / a) / (b - np.log(eair / a))
    return TC_dewpoint
