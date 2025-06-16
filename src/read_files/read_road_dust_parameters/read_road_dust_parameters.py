from .read_model_flags import read_model_flags
from .read_model_parameters import read_model_parameters
from .read_model_activities import read_model_activities
from config_classes import model_flags, model_parameters, model_activities
import pandas as pd
import os


def read_road_dust_parameters(
    parameter_file_path: str,
    read_as_text=0,
) -> tuple[model_parameters, model_flags, model_activities]:
    """
    Read road dust model parameters, flags, and activities from specified file.

    Args:
        parameter_file_path (str): Path to the file containing model parameters.
        read_as_text (int, optional): If 1, read the file as text. Defaults to 0.

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

        parameters_path = os.path.join(text_dir, f"{base_name}_parameters.txt")
        flags_path = os.path.join(text_dir, f"{base_name}_flags.txt")
        activities_path = os.path.join(text_dir, f"{base_name}_params.txt")

        parameter_df = pd.read_csv(parameters_path, sep="\t", header=None)
        flags_df = pd.read_csv(flags_path, sep="\t", header=None)
        activities_df = pd.read_csv(activities_path, sep="\t", header=None)

    else:
        all_sheet = pd.read_excel(parameter_file_path, sheet_name=None)
        parameter_df = all_sheet["Parameters"]
        flags_df = all_sheet["Flags"]
        activities_df = all_sheet["Activities"]

    parameters = read_model_parameters(parameter_df)
    flags = read_model_flags(flags_df)
    activities = read_model_activities(activities_df)

    return parameters, flags, activities
