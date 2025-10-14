from .read_model_flags import read_model_flags
from .read_model_parameters import read_model_parameters
from .read_model_activities import read_model_activities
from config_classes import model_flags, model_parameters, model_activities
import pandas as pd
import os
import logging
from pd_util import read_txt
from pathlib import Path

logger = logging.getLogger(__name__)


def read_road_dust_parameters(
    parameter_file_path: str,
) -> tuple[model_parameters, model_flags, model_activities, dict[str, pd.DataFrame]]:
    """
    Read road dust model parameters, flags, and activities from specified file.

    Args:
        parameter_file_path (str): Path to the xlsx or txt containing model parameters.

    Returns:
        tuple: A tuple containing:
            - model_parameters: An instance of model_parameters with loaded values.
            - model_flags: An instance of model_flags with loaded values.
            - model_activities: An instance of model_activities with loaded values.
            - all_sheets: A dictionary containing all sheets from the input file or None if read_as_text is True.
    """

    if parameter_file_path.endswith(".txt"):
        # Extract directory and base filename without extension
        base_dir = os.path.dirname(parameter_file_path)
        base_name = Path(parameter_file_path).stem

        parameters_path = os.path.join(base_dir, f"{base_name}_params.txt")
        flags_path = os.path.join(base_dir, f"{base_name}_flag.txt")
        activities_path = os.path.join(base_dir, f"{base_name}_activities.txt")

        try:
            parameter_df = read_txt(parameters_path)
            flags_df = read_txt(flags_path)
            activities_df = read_txt(activities_path)

            all_sheets = {}
            all_sheets["Parameters"] = parameter_df
            all_sheets["Flags"] = flags_df
            all_sheets["Activities"] = activities_df

        except FileNotFoundError as e:
            logger.error(f"Parameter file not found: {e.filename}")
            exit(1)

    elif parameter_file_path.endswith(".xlsx"):
        try:
            all_sheets = pd.read_excel(parameter_file_path, sheet_name=None)
        except FileNotFoundError:
            logger.error(f"Parameter file not found: {parameter_file_path}")
            exit(1)
        parameter_df = all_sheets["Parameters"]
        flags_df = all_sheets["Flags"]
        activities_df = all_sheets["Activities"]

    else:
        logger.error(f"Invalid parameter file type: {parameter_file_path}")
        exit(1)

    parameters = read_model_parameters(parameter_df)  # type: ignore
    flags = read_model_flags(flags_df)  # type: ignore
    activities = read_model_activities(activities_df, parameters)  # type: ignore

    return parameters, flags, activities, all_sheets
