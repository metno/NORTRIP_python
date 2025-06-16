from config_classes.model_parameters import model_parameters
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def read_model_parameters(paramter_df: pd.DataFrame) -> model_parameters:
    """
    Load model parameters from xlsx file and return an instance of model_parameters.

    Args:
        file_path (str): Path to the file containing model parameters.

    Returns:
        loaded_parameters (model paramters): An instance of model_parameters with loaded values.
    """
    loaded_parameters = model_parameters()

    return loaded_parameters
