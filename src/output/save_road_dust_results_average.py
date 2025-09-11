from __future__ import annotations

import os
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
from config_classes.model_file_paths import model_file_paths
import constants
from .results_dataframe import build_results_dataframe
import logging

logger = logging.getLogger(__name__)

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
    av: list[int],
    save_as_text: bool,
    paths: model_file_paths,
    parameter_sheets: dict[str, pd.DataFrame],
):
    """
    Create and return the averaged results DataFrame.

    Writing to text/excel will be implemented separately; for now we just
    return the pandas DataFrame with all expected columns.
    """
    df_results = build_results_dataframe(
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
    
    os.makedirs(paths.path_outputdata, exist_ok=True)

    av_index = av[0] if av else model_flags.plot_type_flag

    av_label = constants.av_str[max(0, av_index - 1)]
    if paths.filename_outputdata == "":
        paths.filename_outputdata = "missing_filename"
    base = os.path.join(paths.path_outputdata, paths.filename_outputdata)
    if save_as_text:
        out_file = f"{base}_{av_label}.txt"

        logger.info(f"Saving results to {out_file}...")
        df_results.to_csv(out_file, sep="\t", index=False, na_rep="-999")
    else:
        out_file = f"{base}_{av_label}.xlsx"
        logger.info(f"Saving results to {out_file}...")
        with pd.ExcelWriter(out_file, engine="openpyxl") as writer:
            df_results.to_excel(writer, index=False, sheet_name="data")
            if parameter_sheets:
                for sheet_name in ("Parameters", "Flags", "Activities"):
                    parameter_sheets[sheet_name].to_excel(
                        writer, index=False, sheet_name=sheet_name
                    )

    logger.info("Successfully saved results!")