def r_0_wind_func(FF: float, tau_wind: float, FF_thresh: float) -> float:
    """
    Wind blown dust wind speed dependency function.

    Args:
        FF: Wind speed
        tau_wind: Wind blown dust time scale
        FF_thresh: Threshold wind speed

    Returns:
        float: Wind factor
    """
    if FF > FF_thresh:
        h_FF = (FF / FF_thresh - 1) ** 3
    else:
        h_FF = 0.0

    f = (1 / tau_wind) * h_FF

    return f
