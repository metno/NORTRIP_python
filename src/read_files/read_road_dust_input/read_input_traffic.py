import pandas as pd
import numpy as np
import datetime
from input_classes import input_traffic
import constants


def read_input_traffic(
    traffic_df: pd.DataFrame, nodata: float = -99.0, print_results: bool = False
) -> input_traffic:
    """
    Parse traffic data from DataFrame into input_traffic dataclass.
    """
    loaded_traffic = input_traffic()

    header_text = traffic_df.iloc[0, :].str.replace(r"\s+", "", regex=True)

    traffic_df.columns = header_text
    traffic_df = traffic_df.iloc[1:].reset_index(drop=True)

    loaded_traffic.year = traffic_df["Year"].values.astype(int)
    loaded_traffic.month = traffic_df["Month"].values.astype(int)
    loaded_traffic.day = traffic_df["Day"].values.astype(int)
    loaded_traffic.hour = traffic_df["Hour"].values.astype(int)
    try:
        loaded_traffic.minute = traffic_df["Minute"].values.astype(int)
    except KeyError:
        loaded_traffic.minute = loaded_traffic.year * 0

    # Convert to datetime objects for processing
    datetime_objects = [
        datetime.datetime(year, month, day, hour, minute, 0)
        for year, month, day, hour, minute in zip(
            loaded_traffic.year,
            loaded_traffic.month,
            loaded_traffic.day,
            loaded_traffic.hour,
            loaded_traffic.minute,
            strict=True,
        )
    ]

    # Create date_num similar to MATLAB datenum (days since year 1 + fractional day)
    loaded_traffic.date_num = np.array([dt.timestamp() for dt in datetime_objects])

    # Create date_str arrays matching MATLAB datestr format
    date_str_format1 = np.array([dt.strftime("%Y.%m.%d %H") for dt in datetime_objects])
    date_str_format2 = np.array(
        [dt.strftime("%H:%M %d %b ") for dt in datetime_objects]
    )

    # Combine into 2D array as expected by the class
    loaded_traffic.date_str = np.array(
        [date_str_format1, date_str_format2], dtype=object
    )

    loaded_traffic.n_traffic = len(traffic_df)

    return loaded_traffic
