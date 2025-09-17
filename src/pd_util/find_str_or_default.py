import pandas as pd
from .find_value import find_value


def find_str_or_default(
    search_text: str,
    header_series: pd.Series,
    data_series: pd.Series,
    default_val: str,
) -> str:
    result = find_value(search_text, header_series, data_series)
    if (result == "" or 
        pd.isna(result) or 
        str(result).strip() == "" or 
        result.lower() == "nan"):
        
        return default_val
    return str(result)
