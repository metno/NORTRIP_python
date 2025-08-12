from .read_model_flags import read_model_flags
from .read_model_parameters import read_model_parameters
from .read_model_activities import read_model_activities
from config_classes import model_flags, model_parameters, model_activities
import pandas as pd
import os
import logging
from pd_util import read_txt

logger = logging.getLogger(__name__)


def read_road_dust_parameters(
    parameter_file_path: str,
    read_as_text=False,
) -> tuple[model_parameters, model_flags, model_activities]:
    """
    Read road dust model parameters, flags, and activities from specified file.

    Args:
        parameter_file_path (str): Path to the xlsx containing model parameters.
        read_as_text (bool, optional): If True, read the file as text. Will reformat parameter_file_path to text format.

    Returns:
        tuple: A tuple containing:
            - model_parameters: An instance of model_parameters with loaded values.
            - model_flags: An instance of model_flags with loaded values.
            - model_activities: An instance of model_activities with loaded values.
    """

    if read_as_text:
        # Extract directory and base filename without extension
        base_dir = os.path.dirname(parameter_file_path)
        base_name = os.path.splitext(os.path.basename(parameter_file_path))[0]
        text_dir = os.path.join(base_dir, "text")

        parameters_path = os.path.join(text_dir, f"{base_name}_params.txt")
        flags_path = os.path.join(text_dir, f"{base_name}_flag.txt")
        activities_path = os.path.join(text_dir, f"{base_name}_activities.txt")

        try:
            parameter_df = read_txt(parameters_path)
            flags_df = read_txt(flags_path)
            activities_df = read_txt(activities_path)
        except FileNotFoundError as e:
            logger.error(f"File not found: {e.filename}")
            exit(1)

    else:
        try:
            all_sheets = pd.read_excel(parameter_file_path, sheet_name=None)
        except FileNotFoundError:
            logger.error(f"File not found: {parameter_file_path}")
            exit(1)
        parameter_df = all_sheets["Parameters"]
        flags_df = all_sheets["Flags"]
        activities_df = all_sheets["Activities"]

    parameters = read_model_parameters(parameter_df)  # type: ignore
    flags = read_model_flags(flags_df)  # type: ignore
    activities = read_model_activities(activities_df, parameters)  # type: ignore

    return parameters, flags, activities
