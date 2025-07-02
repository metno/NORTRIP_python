def energy_correction_func(dE1: float, dE2: float) -> float:
    """
    Energy correction function used for correcting energy balance based on observed road temperature.

    Args:
        dE1: First energy term
        dE2: Second energy term

    Returns:
        float: Energy correction value
    """
    f = 1.0
    E_correction = f * dE1 + (1 - f) * dE2

    return E_correction
