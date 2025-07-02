import numpy as np
from .q_sat_func import q_sat_func


def penman_modified_func(
    rad_net: float,
    G: float,
    r_aero: float,
    TC: float,
    RH: float,
    RHs: float,
    P: float,
    dt_h: float,
    g_surf: float,
    g_min: float,
    surface_humidity_flag: int,
) -> tuple[float, float, float, float, float, float]:
    """
    Calculate evaporation using modified Penman equation.

    Output units are mm/hr.

    Args:
        rad_net: Net radiation
        G: Ground heat flux
        r_aero: Aerodynamic resistance
        TC: Air temperature in Celsius
        RH: Relative humidity in %
        RHs: Surface relative humidity in %
        P: Pressure in Pa
        dt_h: Time step in hours
        g_surf: Surface moisture
        g_min: Minimum surface moisture
        surface_humidity_flag: Surface humidity calculation flag

    Returns:
        tuple: (evap_m, evap_pot, TCs, H, L, G) where:
            evap_m: Modified evaporation
            evap_pot: Potential evaporation
            TCs: Surface temperature
            H: Sensible heat flux
            L: Latent heat flux
            G: Ground heat flux
    """
    set_limit_noevap = 1

    # Set constants
    Cp = 1006.0
    lambda_val = 2.5e6
    RD = 287.0
    T0C = 273.15
    gam = Cp / lambda_val
    dt_sec = dt_h * 3600

    # Lower boundary to the surface relative humidity at 1%
    RHs = max(RHs, 1.0)

    # Set modified gam
    gam_m = gam * 100.0 / RHs

    # Determine density
    rho = P * 100.0 / (RD * (T0C + TC))

    # Determine saturated vapour pressure in air
    esat, qsat, d_qsat_dT = q_sat_func(TC, P)

    # Set the water vapour deficit
    q_def = qsat * (1 - RH / 100.0)
    q_def_m = qsat * (RHs / 100.0 - RH / 100.0)
    q = qsat * (RH / 100.0)

    # Set gamma's
    gamma = d_qsat_dT / (d_qsat_dT + gam)
    gamma_m = d_qsat_dT / (d_qsat_dT + gam_m)

    # Potential evaporation
    evap_pot = (
        3600
        / lambda_val
        * (gamma * (rad_net - G) + (1 - gamma) * rho * lambda_val * q_def / r_aero)
    )

    # Modified evaporation
    evap_m = (
        3600
        / lambda_val
        * (
            gamma_m * (rad_net - G)
            + (1 - gamma_m) * rho * lambda_val * q_def_m / r_aero
        )
    )

    # Limit evaporation (L) to the amount of water available
    if set_limit_noevap == 1:
        L = lambda_val * evap_m / 3600
        dT = (rad_net - G - L) * r_aero / Cp / rho
        TCs = TC + dT
        esat_s, qsats, _ = q_sat_func(TCs, P)

        if surface_humidity_flag == 1:
            g_noevap = q / qsats * g_min
        elif surface_humidity_flag == 2:
            g_noevap = -np.log(1 - q / qsats) * g_min / 2
        else:
            g_noevap = 0.0
    else:
        g_noevap = 0.0

    if evap_m >= (g_surf - g_noevap) / dt_h and evap_m >= 0:
        evap_m = max(0.0, (g_surf - g_noevap) / dt_h)
        L = evap_m * lambda_val * dt_h / dt_sec

    # Calculate surface temperature and fluxes
    L = lambda_val * evap_m / 3600
    dT = (rad_net - G - L) * r_aero / Cp / rho
    TCs = TC + dT
    H = rad_net - G - L

    return evap_m, evap_pot, TCs, H, L, G
