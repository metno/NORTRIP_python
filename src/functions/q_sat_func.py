import numpy as np


def q_sat_func(TC: float, P: float) -> tuple[float, float, float]:
    """
    Calculate saturation vapor pressure and related quantities.

    Args:
        TC: Temperature in Celsius
        P: Pressure in Pa (converted from hPa/mbar internally)

    Returns:
        tuple: (esat, qsat, d_qsat_dT) where:
            esat: Saturation vapor pressure in Pa
            qsat: Saturation specific humidity in kg/kg
            d_qsat_dT: Temperature derivative of qsat in kg/kg/K
    """
    # Constants for Magnus formula
    a = 6.1121
    b = 17.67
    c = 243.5

    # Convert pressure from Pa to hPa for calculations (MATLAB uses hPa/mbar)
    P_hPa = P / 100.0

    # Saturation vapor pressure (Magnus formula) in hPa
    esat_hPa = a * np.exp(b * TC / (c + TC))

    # Convert back to Pa for consistency
    esat = esat_hPa * 100.0

    # Saturation specific humidity (Clausius-Clapeyron)
    # Using hPa for pressure in this calculation as in MATLAB
    qsat = 0.622 * esat_hPa / (P_hPa - 0.378 * esat_hPa)

    # Temperature derivative of saturation vapor pressure (hPa/K)
    d_esat_dT_hPa = esat_hPa * b * c / ((c + TC) ** 2)

    # Temperature derivative of saturation specific humidity (kg/kg/K)
    d_qsat_dT = 0.622 * P_hPa / ((P_hPa - 0.378 * esat_hPa) ** 2) * d_esat_dT_hPa

    return esat, qsat, d_qsat_dT
