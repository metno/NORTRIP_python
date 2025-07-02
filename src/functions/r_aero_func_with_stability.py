import numpy as np
from typing import List


def r_aero_func_with_stability(
    FF: float,
    Tc: float,
    Ts: float,
    z_FF: float,
    z_T: float,
    z0: float,
    z0t: float,
    V_veh: List[float],
    N_v: List[float],
    num_veh: int,
    a_traffic: List[float],
) -> float:
    """
    Calculate aerodynamic resistance with atmospheric stability corrections.

    Args:
        FF: Wind speed
        Tc: Air temperature
        Ts: Surface temperature
        z_FF: Wind measurement height
        z_T: Temperature measurement height
        z0: Roughness length for momentum
        z0t: Roughness length for heat
        V_veh: Vehicle velocities for each vehicle type
        N_v: Number of vehicles for each vehicle type
        num_veh: Number of vehicle types
        a_traffic: Traffic coefficients for each vehicle type

    Returns:
        float: Aerodynamic resistance with stability corrections
    """
    # Constants
    kappa = 0.4
    g = 9.8
    T0K = 273.15
    a_stab = 16.0
    b_stab = 5.0
    p = -0.25
    q = -0.5
    pi = np.pi
    iterations = 2
    phi_h = 0.0
    phi_m = 0.0
    eps = 0.0

    for i in range(iterations):
        FF_temp = max(
            0.2, FF * (np.log(z_T / z0) - phi_m) / (np.log(z_FF / z0) - phi_m)
        )

        # Richardson number
        Rib = g / (Tc + T0K) * z_T * (Tc - Ts) / (FF_temp * FF_temp)

        # Stability parameter
        eps = (
            Rib
            * (np.log(z_T / z0) - phi_m)
            * (np.log(z_T / z0) - phi_m)
            / (np.log(z_T / z0t) - phi_h)
        )

        if eps >= 0:
            # Stable conditions
            phim = 1 + b_stab * eps
            phih = 1 + b_stab * eps
            phi_m = -b_stab * eps
            phi_h = -b_stab * eps
        else:
            # Unstable conditions
            phim = (1.0 - a_stab * eps) ** p
            phih = (1.0 - a_stab * eps) ** q
            phi_m = (
                2.0 * np.log((1.0 + 1.0 / phim) / 2.0)
                + np.log((1.0 + 1.0 / (phim * phim)) / 2.0)
                - 2.0 * np.arctan(1.0 / phim)
                + pi / 2.0
            )
            phi_h = 2.0 * np.log((1.0 + 1.0 / phih) / 2.0)

    # Wind component of resistance
    inv_r_wind = (
        FF_temp
        * kappa
        * kappa
        / ((np.log(z_T / z0) - phi_m) * (np.log(z_T / z0t) - phi_h))
    )

    # Traffic component of resistance
    inv_r_traffic = 0.0
    for v in range(num_veh):
        inv_r_traffic += N_v[v] * V_veh[v] * a_traffic[v]

    inv_r_traffic = max(1e-6, inv_r_traffic / 3600 / 3.6)

    # Combined aerodynamic resistance
    inv_r_aero = inv_r_traffic + inv_r_wind
    r_aero = 1 / inv_r_aero

    return r_aero
