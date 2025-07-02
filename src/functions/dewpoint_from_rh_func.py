import numpy as np
from typing import Union


def dewpoint_from_rh_func(
    TC: Union[float, np.ndarray], RH: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
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

    # Dewpoint temperature
    TC_dewpoint = c * np.log(eair / a) / (b - np.log(eair / a))

    return TC_dewpoint
