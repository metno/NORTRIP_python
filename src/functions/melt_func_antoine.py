import numpy as np


def melt_func_antoine(
    solution_salt: float,
    saturated: float,
    melt_temperature_saturated: float,
    ai: float,
    bi: float,
    ci: float,
    a: float,
    b: float,
    c: float,
    afactor: float,
) -> float:
    """
    Calculate melting temperature using the Antoine equation.

    This function finds the melt temperature from:
    as - bs/(cs - T_melt) = ai - bi/(ci - T_melt)
    Which gives: AA(T_melt)^2 + BB*T + CC = 0
    Solution: T_melt = (-BB +- sqrt(BB^2 - 4*AA*CC)) / (2*AA)

    Args:
        solution_salt: Salt solution concentration
        saturated: Saturated concentration
        melt_temperature_saturated: Saturated melting temperature
        ai: Antoine coefficient A for ice
        bi: Antoine coefficient B for ice
        ci: Antoine coefficient C for ice
        a: Antoine coefficient A for solution
        b: Antoine coefficient B for solution
        c: Antoine coefficient C for solution
        afactor: Activity factor

    Returns:
        float: Melting temperature in Celsius
    """
    # Adjust Antoine coefficients for solution
    as_val = a + np.log10(afactor)
    bs_val = b
    cs_val = c

    # Calculate quadratic equation coefficients
    AA = ai - as_val
    BB = (ai - as_val) * (ci + cs_val) - bi + bs_val
    CC = (ai - as_val) * cs_val * ci - bi * cs_val + bs_val * ci

    # Calculate discriminant
    discriminant = BB**2 - 4 * AA * CC

    # Determine melting temperature
    if solution_salt == 0.0:
        # Minimum solution
        melt_temp = 0.0
    elif discriminant < 0.0:
        # Imaginary roots
        melt_temp = melt_temperature_saturated
    else:
        # Real roots - use the negative root
        melt_temp = (-BB - np.sqrt(BB**2 - 4 * AA * CC)) / (2 * AA)

    # Apply constraints
    if melt_temp < melt_temperature_saturated:
        melt_temp = melt_temperature_saturated

    if melt_temp > 0:
        melt_temp = 0.0

    return melt_temp
