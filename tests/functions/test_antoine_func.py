import numpy as np
from src.functions.antoine_func import antoine_func


def test_antoine_func():
    """Test Antoine equation vapor pressure calculation."""
    # Test with typical water vapor parameters
    # Antoine coefficients for water (common values)
    a = 8.07131
    b = 1730.63
    c = 233.426

    # Test at 20°C
    TC = 20.0
    result = antoine_func(a, b, c, TC)
    expected = 10 ** (a - (b / (c + TC)))
    assert abs(result - expected) < 1e-10

    # Test at freezing point
    TC = 0.0
    result = antoine_func(a, b, c, TC)
    expected = 10 ** (a - (b / (c + TC)))
    assert abs(result - expected) < 1e-10

    # Test with different coefficients (ice)
    a_ice = 9.5504
    b_ice = 2348.8
    c_ice = 273.2
    TC = -10.0
    result = antoine_func(a_ice, b_ice, c_ice, TC)
    expected = 10 ** (a_ice - (b_ice / (c_ice + TC)))
    assert abs(result - expected) < 1e-10

    # Test positive result
    assert result > 0

    # Test that higher temperature gives higher vapor pressure
    result_higher_temp = antoine_func(a, b, c, 30.0)
    result_lower_temp = antoine_func(a, b, c, 10.0)
    assert result_higher_temp > result_lower_temp
