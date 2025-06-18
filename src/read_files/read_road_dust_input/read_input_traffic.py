import pandas as pd
import numpy as np
from input_classes import input_traffic
import constants


def read_input_traffic(
    traffic_df: pd.DataFrame, nodata: float = -99.0, print_results: bool = False
) -> input_traffic:
    """
    Parse traffic data from DataFrame into input_traffic dataclass.
    """
    loaded_traffic = input_traffic()

    return loaded_traffic
