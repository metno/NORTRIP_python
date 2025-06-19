import pandas as pd
import numpy as np


def read_txt(filepath: str) -> pd.DataFrame:
    """
    Read a text file into a pandas DataFrame.

    Args:
        filepath (str): Path to the text file to read

    Returns:
        pd.DataFrame: DataFrame containing the data from the text file
    """

    rows = []
    max_cols = 0
    with open(filepath, "r", encoding="latin-1") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 1:
                parts = line.rstrip("\n").split()
            max_cols = max(max_cols, len(parts))
            rows.append(parts)
    # Pad all rows to max_cols
    for i in range(len(rows)):
        if len(rows[i]) < max_cols:
            rows[i] = rows[i] + [np.nan] * (max_cols - len(rows[i]))
        elif len(rows[i]) > max_cols:
            rows[i] = rows[i][:max_cols]
    columns = [f"{i}" for i in range(max_cols)]
    df = pd.DataFrame(rows, columns=pd.Index(columns))
    return df
