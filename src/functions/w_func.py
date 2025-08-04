import numpy as np


def w_func(
    W_0: float,
    h_pave: float,
    h_dc: float,
    V_veh: float,
    a_wear: np.ndarray,
    s_road: float,
    s_roadwear_thresh: float,
    s: int,
    road_index: int,
    tyre_index: int,
    brake_index: int,
) -> float:
    """
    Wear function that calculates wear production based on various factors.

    Args:
        W_0: Base wear factor
        h_pave: Pavement factor
        h_dc: Driving cycle factor
        V_veh: Vehicle velocity
        a_wear: Wear coefficients [a1, a2, a3, a4, a5]
        s_road: Snow/water on road surface
        s_roadwear_thresh: Threshold for snow/water affecting wear
        s: Source type index
        road_index: Index for road dust source
        tyre_index: Index for tyre wear source
        brake_index: Index for brake wear source

    Returns:
        float: Wear factor
    """
    # No wear production due to snow on the surface
    f_snow = 1.0
    if s_road > s_roadwear_thresh:
        f_snow = 0.0

    # Velocity factor
    f_V = max(
        0.0, a_wear[0] + a_wear[1] * (max(V_veh, a_wear[4]) / a_wear[3]) ** a_wear[2]
    )

    # Source-specific adjustments
    h_pave_adj = h_pave
    h_dc_adj = h_dc
    f_snow_adj = f_snow

    if s == road_index:
        h_dc_adj = 1.0
    if s == tyre_index:
        h_dc_adj = 1.0
        h_pave_adj = 1.0
    if s == brake_index:
        h_pave_adj = 1.0
        f_snow_adj = 1.0

    # Calculate final wear factor
    f = W_0 * h_pave_adj * h_dc_adj * f_V * f_snow_adj


    return f
