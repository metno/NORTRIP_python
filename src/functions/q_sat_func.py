import numpy as np


def q_sat_func(TC: float, P: float) -> tuple[float, float, float]:
    """
    Calculate saturation vapor pressure and related quantities.

    Args:
        TC: Temperature in Celsius
        P: Pressure in Pa

    Returns:
        tuple: (esat, qsat, s) where:
            esat: Saturation vapor pressure in Pa
            qsat: Saturation specific humidity in kg/kg
            s: Slope of saturation vapor pressure curve in Pa/K
    """
    # Constants for Magnus formula
    a = 6.1121
    b = 17.67
    c = 243.5

    # Saturation vapor pressure (Magnus formula) in Pa
    esat = a * np.exp(b * TC / (c + TC))

    # Saturation specific humidity (Clausius-Clapeyron)
    # q = 0.622 * e / (P - 0.378 * e)
    qsat = 0.622 * esat / (P - 0.378 * esat)

    # Slope of saturation vapor pressure curve (Pa/K)
    s = esat * b * c / ((c + TC) ** 2)

    return esat, qsat, s
