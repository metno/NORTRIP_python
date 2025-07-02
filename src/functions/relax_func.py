import numpy as np


def relax_func(dt: float, hour_of_forecast: int) -> float:
    """
    Calculate relaxation term for modifying the energy correction term during a forecast.

    Returns a fraction between 0 and 1 for modifying the energy correction term.

    Args:
        dt: Time step in hours
        hour_of_forecast: Current forecast hour

    Returns:
        float: Relaxation term (0-1)
    """
    # Create modifier that linearly decreases from 1 to 0 over 3 hours
    modifier_length = round(3.0 / dt)
    modifier = np.linspace(1.0, 0.0, modifier_length)

    if hour_of_forecast > len(modifier):
        relaxation_term = 0.0
    else:
        # MATLAB uses 1-based indexing, Python uses 0-based
        relaxation_term = modifier[hour_of_forecast - 1]

    return relaxation_term
