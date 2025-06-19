import pandas as pd
from src.pd_util import safe_float


def test_safe_float():
    """Test the safe_float function with various inputs."""
    # Test normal float conversion
    assert safe_float("10.5") == 10.5
    assert safe_float(10.5) == 10.5
    assert safe_float("10") == 10.0

    # Test European decimal format (comma as decimal separator)
    assert safe_float("10,5") == 10.5
    assert safe_float("1,234") == 1.234

    # Test edge cases
    assert safe_float(pd.NA) == 0.0
    assert safe_float(None) == 0.0
    assert safe_float("") == 0.0
    assert safe_float("nan") == 0.0
    assert safe_float("NaN") == 0.0
    assert safe_float("  ") == 0.0

    # Test invalid values
    assert safe_float("invalid") == 0.0
    assert safe_float("10.5.6") == 0.0


def test_safe_float_whitespace():
    """Test safe_float with various whitespace scenarios."""
    assert safe_float("  10.5  ") == 10.5
    assert safe_float("\t5.2\n") == 5.2
    assert safe_float("   ") == 0.0


def test_safe_float_edge_cases():
    """Test safe_float with edge cases and boundary values."""
    # Test very large numbers
    assert safe_float("1e10") == 1e10
    assert safe_float("1.5e-5") == 1.5e-5

    # Test negative numbers
    assert safe_float("-10.5") == -10.5
    assert safe_float("-5,7") == -5.7

    # Test zero values
    assert safe_float("0") == 0.0
    assert safe_float("0.0") == 0.0
    assert safe_float("0,0") == 0.0


def test_safe_float_type_variations():
    """Test safe_float with different input types."""
    # Test integer input
    assert safe_float(42) == 42.0

    # Test boolean input (converts to string "True"/"False" which are invalid, so returns 0.0)
    assert safe_float(True) == 0.0
    assert safe_float(False) == 0.0
