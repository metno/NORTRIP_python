import numpy as np


def q_sat_func(TC: float, P: float) -> tuple[float, float, float]:
    """
    Calculate saturation vapor pressure and related quantities.

    Args:
        TC: Temperature in Celsius
        P: Pressure in hPa/mbar (same as MATLAB)

    Returns:
        tuple: (esat, qsat, d_qsat_dT) where:
            esat: Saturation vapor pressure in hPa
            qsat: Saturation specific humidity in kg/kg
            d_qsat_dT: Temperature derivative of qsat in kg/kg/K
    """
    # Constants for Magnus formula
    a = 6.1121
    b = 17.67
    c = 243.5

    # Pressure is already in hPa/mbar (same as MATLAB)
    P_hPa = P

    # Saturation vapor pressure (Magnus formula) in hPa
    esat = a * np.exp(b * TC / (c + TC))

    # Saturation specific humidity (Clausius-Clapeyron)
    qsat = 0.622 * esat / (P_hPa - 0.378 * esat)

    # Temperature derivative of saturation vapor pressure (hPa/K)
    d_esat_dT = esat * b * c / ((c + TC) ** 2)

    # Temperature derivative of saturation specific humidity (kg/kg/K)
    d_qsat_dT = 0.622 * P_hPa / ((P_hPa - 0.378 * esat) ** 2) * d_esat_dT

    return esat, qsat, d_qsat_dT
