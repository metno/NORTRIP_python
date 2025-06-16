import pandas as pd


def find_value(search_text: str, header_list: pd.Series, file_list: pd.Series) -> str:
    matches = header_list.str.contains(search_text, case=False, na=False)
    if matches.any():
        return file_list[matches].iloc[0]
    return ""
