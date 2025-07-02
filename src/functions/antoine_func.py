import numpy as np


def antoine_func(a: float, b: float, c: float, TC: float) -> float:
    """
    Calculate vapor pressure using the Antoine equation.

    The Antoine equation is: log10(P) = a - b/(c + T)
    where P is vapor pressure and T is temperature.

    Args:
        a: Antoine coefficient A
        b: Antoine coefficient B
        c: Antoine coefficient C
        TC: Temperature in Celsius

    Returns:
        float: Vapor pressure
    """
    vp = 10 ** (a - (b / (c + TC)))
    return vp
