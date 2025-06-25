import pandas as pd
from src.pd_util.find_column_index import find_column_index


def test_find_column_index_exact_match():
    """Test exact string matching functionality."""
    headers = pd.Series(["Year", "Month", "Day", "N(total)", "Temperature"])

    # Test exact matches
    assert find_column_index("Year", headers, exact_match=True) == 0
    assert find_column_index("N(total)", headers, exact_match=True) == 3
    assert find_column_index("Temperature", headers, exact_match=True) == 4

    # Test case insensitive exact matches
    assert find_column_index("year", headers, exact_match=True) == 0
    assert find_column_index("MONTH", headers, exact_match=True) == 1

    # Test non-matches
    assert find_column_index("Temp", headers, exact_match=True) == -1
    assert find_column_index("N(", headers, exact_match=True) == -1


def test_find_column_index_substring_match():
    """Test substring matching functionality."""
    headers = pd.Series(
        ["T2m", "FF", "Global radiation", "Road wetness", "Temperature"]
    )

    # Test substring matches (default behavior)
    assert find_column_index("T2m", headers) == 0
    assert find_column_index("Global", headers) == 2
    assert find_column_index("radiation", headers) == 2
    assert find_column_index("Road", headers) == 3
    assert find_column_index("wetness", headers) == 3

    # Test case insensitive
    assert find_column_index("global", headers) == 2
    assert find_column_index("ROAD", headers) == 3

    # Test non-matches
    assert find_column_index("xyz", headers) == -1
    assert find_column_index("Solar", headers) == -1


def test_find_column_index_exact_vs_substring():
    """Test difference between exact and substring matching."""
    headers = pd.Series(["Temp", "Temperature", "Temp_avg"])

    # Exact match should only match exact strings
    assert find_column_index("Temp", headers, exact_match=True) == 0
    assert find_column_index("Temperature", headers, exact_match=True) == 1
    assert find_column_index("Temp_avg", headers, exact_match=True) == 2

    # Substring match should match the first occurrence containing the substring
    assert find_column_index("Temp", headers, exact_match=False) == 0
    assert find_column_index("Temperature", headers, exact_match=False) == 1


def test_find_column_index_duplicate_handling():
    """Test handling of duplicate headers."""
    headers = pd.Series(["Year", "Month", "Year", "Day", "Year"])

    # Should return the first occurrence
    assert find_column_index("Year", headers, exact_match=True) == 0
    assert find_column_index("Month", headers, exact_match=True) == 1
    assert find_column_index("Day", headers, exact_match=True) == 3


def test_find_column_index_special_characters():
    """Test handling of headers with special characters."""
    headers = pd.Series(["N(total)", "V_veh(he)", "Road wetness (mm)", "T2m dewpoint"])

    # Exact match with special characters
    assert find_column_index("N(total)", headers, exact_match=True) == 0
    assert find_column_index("V_veh(he)", headers, exact_match=True) == 1
    assert find_column_index("Road wetness (mm)", headers, exact_match=True) == 2

    # Substring match with special characters
    assert find_column_index("N(total)", headers, exact_match=False) == 0
    assert find_column_index("(he)", headers, exact_match=False) == 1
    assert find_column_index("(mm)", headers, exact_match=False) == 2
    assert find_column_index("dewpoint", headers, exact_match=False) == 3


def test_find_column_index_empty_and_na():
    """Test handling of empty strings and NaN values."""
    headers = pd.Series(["Year", "", None, "Month", pd.NA])

    # Empty string exact match should find empty string at index 1
    assert find_column_index("", headers, exact_match=True) == 1

    # Empty string substring match finds first string (since empty string is contained in all strings)
    assert find_column_index("", headers, exact_match=False) == 0

    # Should not match None values when searching for literal "None"
    assert find_column_index("None", headers, exact_match=True) == -1

    # Should still find valid headers
    assert find_column_index("Year", headers, exact_match=True) == 0
    assert find_column_index("Month", headers, exact_match=True) == 3


def test_find_column_index_print_results():
    """Test print_results parameter (should not raise errors)."""
    headers = pd.Series(["Year", "Month", "Year", "Day"])

    # This should work without errors even with duplicates
    result = find_column_index("Year", headers, print_results=True, exact_match=True)
    assert result == 0

    result = find_column_index("Year", headers, print_results=False, exact_match=True)
    assert result == 0
