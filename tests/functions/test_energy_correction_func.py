from src.functions.energy_correction_func import energy_correction_func


def test_energy_correction_func():
    """Test energy correction function."""

    # Test with basic values
    dE1 = 100.0
    dE2 = 50.0
    result = energy_correction_func(dE1, dE2)

    # Since f = 1.0, result should equal dE1
    assert result == dE1

    # Test with zero values
    result = energy_correction_func(0.0, 0.0)
    assert result == 0.0

    # Test with negative values
    dE1 = -50.0
    dE2 = -25.0
    result = energy_correction_func(dE1, dE2)
    assert result == dE1

    # Test with mixed positive/negative
    dE1 = 75.0
    dE2 = -30.0
    result = energy_correction_func(dE1, dE2)
    assert result == dE1

    # Test with very large values
    dE1 = 1e6
    dE2 = 1e5
    result = energy_correction_func(dE1, dE2)
    assert result == dE1

    # Test with very small values
    dE1 = 1e-6
    dE2 = 1e-7
    result = energy_correction_func(dE1, dE2)
    assert result == dE1


def test_energy_correction_func_formula():
    """Test that the energy correction formula works as expected."""

    # The function uses: E_correction = f * dE1 + (1 - f) * dE2
    # where f = 1.0, so E_correction = dE1

    test_cases = [
        (100.0, 200.0),
        (-50.0, 75.0),
        (0.0, 100.0),
        (123.456, 789.012),
        (-999.9, -111.1),
    ]

    for dE1, dE2 in test_cases:
        result = energy_correction_func(dE1, dE2)
        expected = 1.0 * dE1 + (1.0 - 1.0) * dE2  # f = 1.0
        assert abs(result - expected) < 1e-10
        # Since f = 1.0, result should always equal dE1
        assert result == dE1
