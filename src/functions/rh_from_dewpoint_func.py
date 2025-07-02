import numpy as np
from typing import Union


def rh_from_dewpoint_func(
    TC: Union[float, np.ndarray], TC_dewpoint: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    Calculate relative humidity from air temperature and dewpoint temperature.

    Args:
        TC: Air temperature in Celsius
        TC_dewpoint: Dewpoint temperature in Celsius

    Returns:
        Relative humidity in %
    """
    # Constants for Magnus formula
    a = 6.1121
    b = 17.67
    c = 243.5

    # Saturation vapor pressure at air temperature
    esat = a * np.exp(b * TC / (c + TC))

    # Actual vapor pressure at dewpoint temperature
    eair = a * np.exp(b * TC_dewpoint / (c + TC_dewpoint))

    # Relative humidity
    RH = 100 * eair / esat

    return RH
