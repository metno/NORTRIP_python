import numpy as np


def q_sat_ice_func(TC: float, P: float) -> tuple[float, float, float]:
    """
    Calculate saturation vapor pressure and related quantities over ice.

    This function uses the formulation from the CIMO guide (WMO, 2008).
    Note: d_qsat_dT is valid only for ice, not for water vapour.

    Args:
        TC: Temperature in Celsius
        P: Pressure in Pa (converted from mbar/hPa internally)

    Returns:
        tuple: (esat, qsat, d_qsat_dT) where:
            esat: Saturation vapor pressure over ice in Pa
            qsat: Saturation specific humidity in kg/kg
            d_qsat_dT: Temperature derivative of qsat in kg/kg/K
    """
    # Constants for Magnus formula over ice
    a = 6.1121
    b = 22.46
    c = 272.62

    # Convert pressure from Pa to hPa for calculations
    P_hPa = P / 100.0

    # Saturation vapor pressure over ice (Magnus formula) in hPa
    esat_hPa = a * np.exp(b * TC / (c + TC))

    # Convert back to Pa
    esat = esat_hPa * 100.0

    # Saturation specific humidity (Clausius-Clapeyron)
    # q = 0.622 * e / (P - 0.378 * e)
    qsat = 0.622 * esat / (P - 0.378 * esat)

    # Temperature derivative of saturation vapor pressure (Pa/K)
    d_esat_dT = esat * b * c / ((c + TC) ** 2)

    # Temperature derivative of saturation specific humidity (kg/kg/K)
    d_qsat_dT = 0.622 * P / ((P - 0.378 * esat) ** 2) * d_esat_dT

    return esat, qsat, d_qsat_dT
