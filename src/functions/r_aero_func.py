import numpy as np
from typing import List


def r_aero_func(
    FF: float,
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
    Calculate aerodynamic resistance.

    Args:
        FF: Wind speed
        z_FF: Wind measurement height
        z_T: Temperature measurement height
        z0: Roughness length for momentum
        z0t: Roughness length for heat
        V_veh: Vehicle velocities for each vehicle type
        N_v: Number of vehicles for each vehicle type
        num_veh: Number of vehicle types
        a_traffic: Traffic coefficients for each vehicle type

    Returns:
        float: Aerodynamic resistance
    """
    kappa = 0.4

    inv_r_wind = max(FF, 0.2) * kappa**2 / (np.log(z_FF / z0) * np.log(z_T / z0t))

    inv_r_traffic = 0.0
    for v in range(num_veh):
        inv_r_traffic += N_v[v] * V_veh[v] * a_traffic[v]

    inv_r_traffic = max(1e-6, inv_r_traffic / 3600 / 3.6)

    inv_r_aero = inv_r_traffic + inv_r_wind

    r_aero = 1 / inv_r_aero

    return r_aero
