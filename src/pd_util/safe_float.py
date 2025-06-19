import pandas as pd
import logging

logger = logging.getLogger(__name__)


def safe_float(value) -> float:
    """
    Convert value to float, handling European decimal format (comma as decimal separator).

    Args:
        value: Value to convert to float

    Returns:
        float: Converted value, returns 0.0 for invalid/missing values
    """
    if pd.isna(value):
        return 0.0

    str_val = str(value).strip()
    if str_val == "" or str_val.lower() == "nan":
        return 0.0

    # Replace comma with period for European decimal format
    str_val = str_val.replace(",", ".")

    try:
        return float(str_val)
    except ValueError:
        logger.warning(f"Could not convert '{value}' to float, returning 0.0")
        return 0.0
