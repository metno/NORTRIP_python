import numpy as np


def q_sat_ice_func(TC: float, P: float) -> tuple[float, float, float]:
    """
    Calculate saturation vapor pressure and related quantities over ice.

    This function uses the formulation from the CIMO guide (WMO, 2008).
    Note: d_qsat_dT is valid only for ice, not for water vapour.

    Args:
        TC: Temperature in Celsius
        P: Pressure in hPa/mbar (same as MATLAB)

    Returns:
        tuple: (esat, qsat, d_qsat_dT) where:
            esat: Saturation vapor pressure over ice in hPa
            qsat: Saturation specific humidity in kg/kg
            d_qsat_dT: Temperature derivative of qsat in kg/kg/K
    """
    # Constants for Magnus formula over ice
    a = 6.1121
    b = 22.46
    c = 272.62

    # Pressure is already in hPa/mbar (same as MATLAB)
    P_hPa = P

    # Saturation vapor pressure over ice (Magnus formula) in hPa
    esat = a * np.exp(b * TC / (c + TC))

    # Saturation specific humidity (Clausius-Clapeyron)
    qsat = 0.622 * esat / (P_hPa - 0.378 * esat)

    # Temperature derivative of saturation vapor pressure (hPa/K)
    d_esat_dT = esat * b * c / ((c + TC) ** 2)

    # Temperature derivative of saturation specific humidity (kg/kg/K)
    d_qsat_dT = 0.622 * P_hPa / ((P_hPa - 0.378 * esat) ** 2) * d_esat_dT

    return esat, qsat, d_qsat_dT
