import pandas as pd
import re


def find_value(search_text: str, header_list: pd.Series, file_list: pd.Series) -> str:
    escaped_text = re.escape(search_text)
    matches = header_list.str.contains(escaped_text, case=False, na=False)
    if matches.any():
        return str(file_list[matches].iloc[0])  # type: ignore
    return ""
