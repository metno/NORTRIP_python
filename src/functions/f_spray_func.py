def f_spray_func(
    R_0_spray: float,
    V_veh: float,
    V_ref_spray: float,
    V_thresh_spray: float,
    a_spray: float,
    water_spray_flag: int,
) -> float:
    """
    Calculate spray factor for water spray from vehicles.

    Args:
        R_0_spray: Base spray factor
        V_veh: Vehicle velocity
        V_ref_spray: Reference spray velocity
        V_thresh_spray: Threshold spray velocity
        a_spray: Spray exponent
        water_spray_flag: Flag indicating if water spray is active

    Returns:
        float: Spray factor
    """
    f_spray = 0.0
    if water_spray_flag and V_ref_spray > V_thresh_spray:
        f_spray = (
            R_0_spray
            * (max(0.0, (V_veh - V_thresh_spray)) / (V_ref_spray - V_thresh_spray))
            ** a_spray
        )

    return f_spray
