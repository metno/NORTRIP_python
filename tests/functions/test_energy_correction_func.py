from src.functions.energy_correction_func import energy_correction_func


def test_energy_correction_func_basic():
    """Test basic energy correction calculation."""
    dE1 = 100.0
    dE2 = 50.0

    result = energy_correction_func(dE1, dE2)

    # With f=1.0, should equal dE1
    assert isinstance(result, float)
    assert result == dE1


def test_energy_correction_func_zero_values():
    """Test energy correction with zero values."""
    dE1 = 0.0
    dE2 = 0.0

    result = energy_correction_func(dE1, dE2)

    assert isinstance(result, float)
    assert result == 0.0


def test_energy_correction_func_negative_values():
    """Test energy correction with negative values."""
    dE1 = -50.0
    dE2 = -25.0

    result = energy_correction_func(dE1, dE2)

    assert isinstance(result, float)
    assert result == dE1  # Should equal dE1 since f=1.0


def test_energy_correction_func_mixed_signs():
    """Test energy correction with mixed sign values."""
    dE1 = 100.0
    dE2 = -50.0

    result = energy_correction_func(dE1, dE2)

    assert isinstance(result, float)
    assert result == dE1


def test_energy_correction_func_large_values():
    """Test energy correction with large values."""
    dE1 = 1000000.0
    dE2 = 500000.0

    result = energy_correction_func(dE1, dE2)

    assert isinstance(result, float)
    assert result == dE1
