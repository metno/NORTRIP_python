def f_crushing_func(
    f_crushing_0: float,
    V_veh: float,
    s_road: float,
    V_ref: float,
    s_roadwear_thresh: float,
) -> float:
    """
    Crushing function for calculating crushing factor.

    Args:
        f_crushing_0: Base crushing factor
        V_veh: Vehicle velocity
        s_road: Snow/water on road surface
        V_ref: Reference velocity
        s_roadwear_thresh: Threshold for snow/water affecting wear

    Returns:
        float: Crushing factor
    """
    if V_ref == 0:
        f_V = 1.0
    else:
        f_V = V_veh / V_ref

    # No wear production due to snow on the surface
    f_snow = 1.0
    if s_road > s_roadwear_thresh:
        f_snow = 0.0

    f = f_crushing_0 * f_V * f_snow

    return f
