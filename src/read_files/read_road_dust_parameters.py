from config_classes.model_flags import model_flags
from config_classes.model_parameters import model_parameters
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def read_model_flags(flags_df: pd.DataFrame) -> model_flags:
    """
    Load model flags from xlsx file and return an instance of model_flags.

    Args:
        file_path (str): Path to the file containing model flags.

    Returns:
        model_flags: An instance of model_flags with loaded values.
    """
    loaded_flags = model_flags()

    try:
        for _, row in flags_df.iterrows():
            flag_name = str(row.iloc[0]).strip()
            if hasattr(loaded_flags, flag_name):
                flag_value = int(row.iloc[1])
                setattr(loaded_flags, flag_name, flag_value)
            else:
                logger.warning(
                    f"Flag '{flag_name}' not found in model_flags dataclass."
                )

                pass

    except Exception as e:
        logger.error(f"Error loading model flags: {e}")
        exit()

    return loaded_flags


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
