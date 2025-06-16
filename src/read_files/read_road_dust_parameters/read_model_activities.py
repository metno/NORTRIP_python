import pandas as pd
from config_classes import model_activities, model_parameters
from pd_util import find_value_or_default
import logging

logger = logging.getLogger(__name__)


def read_model_activities(
    activities_df: pd.DataFrame, model_parameters: model_parameters
) -> model_activities:
    """
    Load model activities from DataFrame and return an instance of model_activities.

    Args:
        activities_df (DataFrame): DataFrame containing model activities.
        model_parameters (model_parameters): model_parameters to potentially overwrite ploughing_threshold.

    Returns:
        model_activities: An instance of model_activities with loaded values.
    """
    loaded_activities = model_activities()

    # Extract header and data columns
    header_col = activities_df.iloc[:, 0]  # First column contains headers
    data_col = (
        activities_df.iloc[:, 1]
        if activities_df.shape[1] > 1
        else pd.Series(dtype=float)
    )

    loaded_count = 0

    # Fields that need manual handling (arrays and special cases)
    manual_fields = {"salting_hour", "sanding_hour", "binding_hour"}

    # Process all fields automatically, except those needing manual handling
    for field_name in loaded_activities.__dataclass_fields__:
        if field_name not in manual_fields:
            current_value = getattr(loaded_activities, field_name)
            new_value = find_value_or_default(
                field_name, header_col, data_col, current_value
            )
            if new_value != current_value:
                setattr(loaded_activities, field_name, new_value)
                loaded_count += 1

    # Manual handling for array fields and special cases
    loaded_activities.salting_hour = [
        find_value_or_default(
            "salting_hour(1)", header_col, data_col, loaded_activities.salting_hour[0]
        ),
        find_value_or_default(
            "salting_hour(2)", header_col, data_col, loaded_activities.salting_hour[1]
        ),
    ]

    loaded_activities.sanding_hour = [
        find_value_or_default(
            "sanding_hour(1)", header_col, data_col, loaded_activities.sanding_hour[0]
        ),
        find_value_or_default(
            "sanding_hour(2)", header_col, data_col, loaded_activities.sanding_hour[1]
        ),
    ]

    loaded_activities.binding_hour = [
        find_value_or_default(
            "binding_hour(1)", header_col, data_col, loaded_activities.binding_hour[0]
        ),
        find_value_or_default(
            "binding_hour(2)", header_col, data_col, loaded_activities.binding_hour[1]
        ),
    ]

    # Can overwrite the ploughing threshold set in model parameters
    model_parameters.ploughing_thresh = find_value_or_default(
        "ploughing_thresh", header_col, data_col, model_parameters.ploughing_thresh
    )

    logger.info(f"Successfully loaded {loaded_count} model activity parameters")
    return loaded_activities
