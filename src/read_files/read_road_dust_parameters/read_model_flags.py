from config_classes.model_flags import model_flags
import pandas as pd
import logging
from pd_util import find_float_or_default

logger = logging.getLogger(__name__)


def read_model_flags(flags_df: pd.DataFrame) -> model_flags:
    """
    Load model flags from DataFrame and return an instance of model_flags.

    This function works by using the fields in the model_flags dataclass,
    meaning if new flags are to be added they only need to be added to the dataclass.

    Args:
        flags_df (DataFrame): DataFrame containing model flags.

    Returns:
        model_flags: An instance of model_flags with loaded values.
    """
    loaded_flags = model_flags()
    header_col = flags_df.iloc[:, 0]
    data_col = flags_df.iloc[:, 1]

    for field_name in loaded_flags.__dataclass_fields__:
        default_value = getattr(loaded_flags, field_name)
        new_value = find_float_or_default(
            field_name, header_col, data_col, default_value
        )
        # Cast to int if defined default is int
        if type(default_value) is int:
            setattr(loaded_flags, field_name, int(new_value))
        else:
            setattr(loaded_flags, field_name, new_value)

    logger.info("Successfully loaded model flags")

    if loaded_flags.use_multiple_save_dates_flag != 0:
        print(
            "The 'use multiple save dates' feature is no longer supported, remove the flag or set it to 0"
        )
        exit()

    return loaded_flags
