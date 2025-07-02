import pandas as pd
from src.pd_util import find_float_or_default


def test_find_str_or_default_found_value():
    header = pd.Series(["Header1"])
    data = pd.Series([12.34])
    result = find_float_or_default("Header1", header, data, 0.0)
    assert result == 12.34


def test_find_str_or_default_default_value():
    header = pd.Series(["Header1"])
    data = pd.Series([12.34])
    result = find_float_or_default("Hello", header, data, 0.0)
    assert result == 0.0


def test_find_str_or_default_empty_string():
    header = pd.Series()
    data = pd.Series([12.34])
    result = find_float_or_default("Header1", header, data, 0.0)
    assert result == 0.0


def test_find_float_or_default_comma_decimal():
    header = pd.Series(["with_comma"])
    data = pd.Series(["12,34"])
    result = find_float_or_default("with_comma", header, data, 0.0)
    assert result == 12.34
