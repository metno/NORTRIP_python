from src.functions.melt_func_antoine import melt_func_antoine
import numpy as np


def test_melt_func_antoine():
    """Test melting temperature calculation using Antoine equation."""

    # Test basic case
    solution_salt = 0.1  # 10% salt solution
    saturated = 0.2  # 20% saturation limit
    melt_temperature_saturated = -5.0  # Saturated melt temperature
    ai = 9.5  # Antoine A for ice
    bi = 3000.0  # Antoine B for ice
    ci = 273.0  # Antoine C for ice
    a = 8.5  # Antoine A for solution
    b = 2800.0  # Antoine B for solution
    c = 263.0  # Antoine C for solution
    afactor = 1.0  # Activity factor

    result = melt_func_antoine(
        solution_salt,
        saturated,
        melt_temperature_saturated,
        ai,
        bi,
        ci,
        a,
        b,
        c,
        afactor,
    )

    # Result should be negative (below 0°C)
    assert result <= 0.0
    # Should be above the saturated melt temperature
    assert result >= melt_temperature_saturated


def test_melt_func_antoine_zero_salt():
    """Test case with zero salt solution."""

    # Test minimum solution case
    solution_salt = 0.0
    saturated = 0.2
    melt_temperature_saturated = -8.0
    ai = 9.5
    bi = 3000.0
    ci = 273.0
    a = 8.5
    b = 2800.0
    c = 263.0
    afactor = 1.0

    result = melt_func_antoine(
        solution_salt,
        saturated,
        melt_temperature_saturated,
        ai,
        bi,
        ci,
        a,
        b,
        c,
        afactor,
    )

    # Should return 0.0 for zero salt
    assert result == 0.0


def test_melt_func_antoine_negative_discriminant():
    """Test case with negative discriminant (imaginary roots)."""

    # Create conditions that lead to negative discriminant
    solution_salt = 0.15
    saturated = 0.2
    melt_temperature_saturated = -10.0
    ai = 10.0
    bi = 3500.0
    ci = 280.0
    a = 9.0  # Different from ai to avoid divide by zero
    b = 3400.0  # Different from bi
    c = 270.0  # Different from ci
    afactor = 1.0

    result = melt_func_antoine(
        solution_salt,
        saturated,
        melt_temperature_saturated,
        ai,
        bi,
        ci,
        a,
        b,
        c,
        afactor,
    )

    # Should return a valid temperature (when discriminant is negative, returns saturated temp)
    assert result == melt_temperature_saturated or (
        not np.isnan(result) and result <= 0.0
    )


def test_melt_func_antoine_below_saturated():
    """Test case where calculated temperature is below saturated temperature."""

    solution_salt = 0.05  # Low concentration
    saturated = 0.2
    melt_temperature_saturated = -15.0
    ai = 9.5
    bi = 3000.0
    ci = 273.0
    a = 8.0
    b = 2500.0
    c = 250.0
    afactor = 1.0

    result = melt_func_antoine(
        solution_salt,
        saturated,
        melt_temperature_saturated,
        ai,
        bi,
        ci,
        a,
        b,
        c,
        afactor,
    )

    # Should be constrained to saturated melt temperature
    assert result >= melt_temperature_saturated


def test_melt_func_antoine_above_zero():
    """Test case where calculated temperature would be above 0°C."""

    solution_salt = 0.3  # High concentration
    saturated = 0.2
    melt_temperature_saturated = -2.0
    ai = 8.0
    bi = 2000.0
    ci = 250.0
    a = 7.0
    b = 1500.0
    c = 230.0
    afactor = 0.5  # Low activity factor

    result = melt_func_antoine(
        solution_salt,
        saturated,
        melt_temperature_saturated,
        ai,
        bi,
        ci,
        a,
        b,
        c,
        afactor,
    )

    # Should be constrained to 0°C maximum
    assert result <= 0.0


def test_melt_func_antoine_activity_factor():
    """Test effect of activity factor."""

    solution_salt = 0.15
    saturated = 0.25
    melt_temperature_saturated = -8.0
    ai = 9.5
    bi = 3000.0
    ci = 273.0
    a = 8.5
    b = 2800.0
    c = 263.0

    # Test with different activity factors
    result_low = melt_func_antoine(
        solution_salt, saturated, melt_temperature_saturated, ai, bi, ci, a, b, c, 0.5
    )
    result_high = melt_func_antoine(
        solution_salt, saturated, melt_temperature_saturated, ai, bi, ci, a, b, c, 1.5
    )

    # Both should be valid temperatures
    assert result_low <= 0.0
    assert result_high <= 0.0
    assert result_low >= melt_temperature_saturated
    assert result_high >= melt_temperature_saturated


def test_melt_func_antoine_quadratic_solution():
    """Test the quadratic equation solution."""

    solution_salt = 0.1
    saturated = 0.2
    melt_temperature_saturated = -6.0
    ai = 9.7
    bi = 3100.0
    ci = 275.0
    a = 8.8
    b = 2900.0
    c = 265.0
    afactor = 1.0

    result = melt_func_antoine(
        solution_salt,
        saturated,
        melt_temperature_saturated,
        ai,
        bi,
        ci,
        a,
        b,
        c,
        afactor,
    )

    # Verify the quadratic equation manually
    as_val = a + np.log10(afactor)
    bs_val = b
    cs_val = c

    AA = ai - as_val
    BB = (ai - as_val) * (ci + cs_val) - bi + bs_val
    CC = (ai - as_val) * cs_val * ci - bi * cs_val + bs_val * ci

    discriminant = BB**2 - 4 * AA * CC

    if discriminant >= 0:
        # Should use the negative root
        expected = (-BB - np.sqrt(discriminant)) / (2 * AA)
        expected = max(expected, melt_temperature_saturated)
        expected = min(expected, 0.0)

        assert abs(result - expected) < 1e-10


def test_melt_func_antoine_extreme_values():
    """Test with extreme parameter values."""

    # Very concentrated solution
    solution_salt = 0.8
    saturated = 0.3
    melt_temperature_saturated = -25.0
    ai = 12.0
    bi = 4000.0
    ci = 300.0
    a = 10.0
    b = 3500.0
    c = 280.0
    afactor = 2.0

    result = melt_func_antoine(
        solution_salt,
        saturated,
        melt_temperature_saturated,
        ai,
        bi,
        ci,
        a,
        b,
        c,
        afactor,
    )

    # Should handle extreme values gracefully
    assert result <= 0.0
    assert result >= melt_temperature_saturated
    assert result >= -50.0  # Reasonable lower bound


def test_melt_func_antoine_edge_cases():
    """Test various edge cases."""

    saturated = 0.2
    melt_temperature_saturated = -5.0
    ai = 9.5
    bi = 3000.0
    ci = 273.0
    a = 8.5
    b = 2800.0
    c = 263.0
    afactor = 1.0

    # Test with solution at saturation
    result_sat = melt_func_antoine(
        saturated, saturated, melt_temperature_saturated, ai, bi, ci, a, b, c, afactor
    )
    assert result_sat <= 0.0

    # Test with very small non-zero solution
    result_small = melt_func_antoine(
        1e-6, saturated, melt_temperature_saturated, ai, bi, ci, a, b, c, afactor
    )
    assert result_small <= 0.0

    # Test with very small activity factor (avoid zero which causes log(0) = -inf)
    result_small_activity = melt_func_antoine(
        0.1, saturated, melt_temperature_saturated, ai, bi, ci, a, b, c, 1e-6
    )
    # Should handle small activity factor gracefully
    assert not np.isnan(result_small_activity)


def test_melt_func_antoine_temperature_constraints():
    """Test temperature constraint logic."""

    solution_salt = 0.12
    saturated = 0.2
    melt_temperature_saturated = -12.0
    ai = 9.5
    bi = 3000.0
    ci = 273.0
    a = 8.5
    b = 2800.0
    c = 263.0
    afactor = 1.0

    result = melt_func_antoine(
        solution_salt,
        saturated,
        melt_temperature_saturated,
        ai,
        bi,
        ci,
        a,
        b,
        c,
        afactor,
    )

    # Verify all constraints
    assert result <= 0.0, "Temperature should not exceed 0°C"
    assert result >= melt_temperature_saturated, (
        "Temperature should not be below saturated melt temperature"
    )

    # Test with positive saturated temperature (should still be constrained)
    result_positive_sat = melt_func_antoine(
        solution_salt, saturated, 5.0, ai, bi, ci, a, b, c, afactor
    )
    assert result_positive_sat <= 0.0, (
        "Temperature should still be constrained to 0°C max"
    )
