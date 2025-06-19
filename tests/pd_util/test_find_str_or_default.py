import pandas as pd

from src.pd_util.find_str_or_default import find_str_or_default


def test_find_str_or_default_found_value():
    header = pd.Series(["Header1"])
    data = pd.Series(["Data1"])
    result = find_str_or_default("Header1", header, data, "default")
    assert result == "Data1"


def test_find_str_or_default_default_value():
    header = pd.Series(["Header1"])
    data = pd.Series(["Data1"])
    result = find_str_or_default("Hello", header, data, "default")
    assert result == "default"


def test_find_str_or_default_empty_string():
    header = pd.Series()
    data = pd.Series()
    result = find_str_or_default("Header1", header, data, "default")
    assert result == "default"
