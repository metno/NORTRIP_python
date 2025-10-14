from __future__ import annotations

import os
import pandas as pd

from initialise import time_config
from initialise.road_dust_initialise_variables import model_variables
from pathlib import Path
from input_classes import (
    converted_data,
    input_metadata,
    input_airquality,
    input_activity,
)
from config_classes import (
    model_parameters,
    model_flags,
)
from config_classes.model_file_paths import model_file_paths
import constants
from .results_dataframe import build_results_dataframe
import logging

logger = logging.getLogger(__name__)


def save_road_dust_results_average(
    *,
    time_config: time_config,
    converted_data: converted_data,
    metadata: input_metadata,
    airquality_data: input_airquality,
    model_parameters: model_parameters,
    model_flags: model_flags,
    model_variables: model_variables,
    input_activity: input_activity,
    av: list[int],
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
    base = os.path.join(paths.path_outputdata, Path(paths.filename_outputdata).stem)
    if paths.filename_outputdata.endswith(".txt"):
        out_file = f"{base}_{av_label}.txt"
        logger.info(f"Saving results to {out_file}...")
        df_results.to_csv(out_file, sep="\t", index=False, na_rep="-999")
    elif paths.filename_outputdata.endswith(".xlsx"):
        out_file = f"{base}_{av_label}.xlsx"
        logger.info(f"Saving results to {out_file}...")
        with pd.ExcelWriter(out_file, engine="openpyxl") as writer:
            df_results.to_excel(writer, index=False, sheet_name="data")
            if parameter_sheets:
                for sheet_name in ("Parameters", "Flags", "Activities"):
                    parameter_sheets[sheet_name].to_excel(
                        writer, index=False, sheet_name=sheet_name
                    )
    else:
        logger.error(
            f"Invalid output data file type: {paths.filename_outputdata}. Must be .txt or .xlsx."
        )
        exit(1)

    logger.info("Successfully saved results!")
