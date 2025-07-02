import numpy as np
from src.functions.antoine_func import antoine_func


def test_antoine_func_basic():
    """Test basic Antoine equation calculation."""
    # Test with known values
    a, b, c = 8.07131, 1730.63, 233.426  # Water Antoine constants
    TC = 25.0  # 25°C

    result = antoine_func(a, b, c, TC)

    # Should return vapor pressure around 23.7 hPa for water at 25°C
    assert result > 0
    assert 23 < result < 25  # Approximate range check


def test_antoine_func_zero_temperature():
    """Test Antoine function at 0°C."""
    a, b, c = 8.07131, 1730.63, 233.426
    TC = 0.0

    result = antoine_func(a, b, c, TC)

    assert isinstance(result, float)
    assert result > 0
    assert 4 < result < 5  # Approximate range for water at 0°C


def test_antoine_func_negative_temperature():
    """Test Antoine function at negative temperature."""
    a, b, c = 8.07131, 1730.63, 233.426
    TC = -10.0

    result = antoine_func(a, b, c, TC)

    assert isinstance(result, float)
    assert result > 0


def test_antoine_func_different_coefficients():
    """Test with different Antoine coefficients."""
    # Different substance coefficients
    a, b, c = 7.96681, 1668.21, 228.0
    TC = 20.0

    result = antoine_func(a, b, c, TC)

    assert isinstance(result, float)
    assert result > 0


def test_antoine_func_high_temperature():
    """Test Antoine function at high temperature."""
    a, b, c = 8.07131, 1730.63, 233.426
    TC = 100.0

    result = antoine_func(a, b, c, TC)

    assert isinstance(result, float)
    assert result > 0
    # Should be much higher at 100°C
    assert result > 100
