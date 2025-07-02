def f_susroad_func(f_0_susroad: float, V_veh: float, a_sus: list) -> float:
    """
    Vehicle speed dependence function for suspension.

    Depends on:
    - source (s)
    - tire type (t)
    - vehicle category (v)
    - vehicle speed (V_veh and V_ref)
    - power law dependence (a_sus)

    Args:
        f_0_susroad: Base suspension factor
        V_veh: Vehicle velocity
        a_sus: Suspension coefficients [a1, a2, a3, a4, a5]

    Returns:
        float: Suspension factor
    """
    h_V = max(0.0, a_sus[0] + a_sus[1] * (max(V_veh, a_sus[4]) / a_sus[3]) ** a_sus[2])
    f = f_0_susroad * h_V

    return f
