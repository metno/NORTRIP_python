import pandas as pd
from config_classes import model_activities
import logging

logger = logging.getLogger(__name__)


def read_model_activities(activities_df: pd.DataFrame) -> model_activities:
    """
    Load model activities from DataFrame and return an instance of model_activities.

    Args:
        activities_df (DataFrame): DataFrame containing model activities.

    Returns:
        model_activities: An instance of model_activities with loaded values.
    """
    loaded_activities = model_activities()

    return loaded_activities
