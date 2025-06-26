import pandas as pd
from input_classes import input_activity


def read_input_activity(activity_df: pd.DataFrame):
    """
    Read activity input data from a pandas DataFrame.

    Args:
        activity_df: pandas DataFrame containing activity input data

    Returns:
        input_activity: input_activity object containing parsed data
    """

    loaded_activity = input_activity()

    return loaded_activity
