from __future__ import annotations

import pandas as pd

from initialise import time_config as time_config_cls
from initialise.road_dust_initialise_variables import model_variables as model_vars_cls
from input_classes import (
    converted_data as converted_data_cls,
    input_metadata as input_metadata_cls,
    input_airquality as input_airquality_cls,
    input_activity as input_activity_cls,
)
from config_classes import (
    model_parameters as model_parameters_cls,
    model_flags as model_flags_cls,
)
from .results_dataframe import build_results_dataframe


def save_road_dust_results_average(
    *,
    time_config: time_config_cls,
    converted_data: converted_data_cls,
    metadata: input_metadata_cls,
    airquality_data: input_airquality_cls,
    model_parameters: model_parameters_cls,
    model_flags: model_flags_cls,
    model_variables: model_vars_cls,
    input_activity: input_activity_cls,
    av: list[int] | None = None,
    save_as_text: bool = False,
) -> pd.DataFrame:
    """
    Create and return the averaged results DataFrame.

    Writing to text/excel will be implemented separately; for now we just
    return the pandas DataFrame with all expected columns.
    """
    return build_results_dataframe(
        time_config=time_config,
        converted_data=converted_data,
        metadata=metadata,
        airquality_data=airquality_data,
        model_parameters=model_parameters,
        model_flags=model_flags,
        model_variables=model_variables,
        input_activity=input_activity,
        av=av,
    )