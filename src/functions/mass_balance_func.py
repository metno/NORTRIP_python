import numpy as np


def mass_balance_func(M_0: float, P: float, R: float, dt: float) -> float:
    """
    Calculate temporal mass balance changes.

    Args:
        M_0: Initial mass
        P: Production term
        R: Removal rate
        dt: Time step

    Returns:
        float: New mass value
    """
    if P < R * 1e8:
        M = P / R * (1 - np.exp(-R * dt)) + M_0 * np.exp(-R * dt)
    else:
        M = M_0 + P * dt

    return M
