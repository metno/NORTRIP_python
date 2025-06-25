from config_classes.model_flags import model_flags
import pandas as pd
import logging
from pd_util import find_float_or_default

logger = logging.getLogger(__name__)


def read_model_flags(flags_df: pd.DataFrame) -> model_flags:
    """
    Load model flags from DataFrame and return an instance of model_flags.

    Args:
        flags_df (DataFrame): DataFrame containing model flags.

    Returns:
        model_flags: An instance of model_flags with loaded values.
    """
    loaded_flags = model_flags()
    header_col = flags_df.iloc[:, 0]
    data_col = flags_df.iloc[:, 1]
    loaded_count = 0

    for field_name in loaded_flags.__dataclass_fields__:
        new_value = find_float_or_default(
            field_name, header_col, data_col, getattr(loaded_flags, field_name)
        )
        setattr(loaded_flags, field_name, int(new_value))
        loaded_count += 1

    logger.info(f"Successfully loaded {loaded_count} model flag parameters")
    return loaded_flags
