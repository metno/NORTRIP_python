from .q_sat_func import q_sat_func


def longwave_in_radiation_func(TC: float, RH: float, n_c: float, P: float) -> float:
    """
    Calculate incoming longwave radiation.

    Returns the incoming longwave radiation based on Konzelman et al. (1994)
    and other related articles. See Sedlar and Hock (2008) "On the use of
    incoming radiation parameterisations in a glacier environment".

    Args:
        TC: Air temperature in Celsius
        RH: Relative humidity in %
        n_c: Cloud cover fraction (0-1)
        P: Pressure in Pa

    Returns:
        float: Incoming longwave radiation in W/m²
    """
    # Set constants
    T0C = 273.15
    sigma = 5.67e-8

    # Calculate saturation vapor pressure and related quantities
    esat, qsat, d_qsat_dT = q_sat_func(TC, P)

    # Actual vapor pressure
    # The Python q_sat_func returns esat in Pa, but we need to work in hPa like MATLAB
    # In MATLAB: e_a = esat * RH / 100, where esat is in hPa
    esat_hPa = esat / 100.0  # Convert from Pa to hPa
    e_a = esat_hPa * RH / 100.0  # e_a in hPa

    # Air temperature in Kelvin
    TK_a = T0C + TC

    # Clear sky emissivity (Konzelman et al. 1994)
    # MATLAB: eps_cs=0.23+0.48*(e_a*100/TK_a)^(1/8)
    # where e_a is in hPa, so e_a*100 converts it to Pa for the formula
    eps_cs = 0.23 + 0.48 * (e_a * 100.0 / TK_a) ** (1.0 / 8.0)

    # Cloud emissivity
    eps_cl = 0.97

    # Effective emissivity
    eps_eff = eps_cs * (1 - n_c**2) + eps_cl * n_c**2

    # Incoming longwave radiation
    RL_in = eps_eff * sigma * TK_a**4

    return RL_in
