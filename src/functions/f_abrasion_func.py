def f_abrasion_func(
    f_sandpaper_0: float,
    h_pave: float,
    V_veh: float,
    s_road: float,
    V_ref: float,
    s_roadwear_thresh: float,
) -> float:
    """
    Sandpaper function for abrasion calculation.

    Args:
        f_sandpaper_0: Base sandpaper factor
        h_pave: Pavement factor
        V_veh: Vehicle velocity
        s_road: Snow/water on road surface
        V_ref: Reference velocity
        s_roadwear_thresh: Threshold for snow/water affecting wear

    Returns:
        float: Abrasion factor
    """
    f_V = V_veh / V_ref

    # No wear production due to snow on the surface
    f_snow = 1.0
    if s_road > s_roadwear_thresh:
        f_snow = 0.0

    f = f_sandpaper_0 * h_pave * f_V * f_snow

    return f
