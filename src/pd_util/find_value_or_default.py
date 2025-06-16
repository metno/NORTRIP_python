import pandas as pd
from .find_value import find_value


def find_value_or_default(
    search_text: str,
    header_series: pd.Series,
    data_series: pd.Series,
    default_val: float,
) -> float:
    result = find_value(search_text, header_series, data_series)
    if result == "" or pd.isna(result):
        return default_val

    # Handle case where result is already numeric
    if isinstance(result, (int, float)):
        return float(result)

    # Handle string case with comma replacement
    return float(result.replace(",", "."))
