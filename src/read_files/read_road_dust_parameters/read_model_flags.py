from config_classes.model_flags import model_flags
import pandas as pd
import logging

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

    i = 0
    try:
        for _, row in flags_df.iterrows():
            flag_name = str(row.iloc[0]).strip()
            if hasattr(loaded_flags, flag_name):
                flag_value = int(row.iloc[1])
                setattr(loaded_flags, flag_name, flag_value)
                i += 1
            else:
                logger.warning(
                    f"Flag '{flag_name}' not found in model_flags dataclass."
                )
    except Exception as e:
        logger.error(f"Error loading model flags: {e}")
        raise

    logger.info(f"Successfully loaded {i} model flags")
    return loaded_flags
