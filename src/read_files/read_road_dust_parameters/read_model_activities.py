import pandas as pd
from config_classes import model_activities
from pd_util import find_value_or_default
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

    # Extract header and data columns
    header_col = activities_df.iloc[:, 0]  # First column contains headers
    data_col = (
        activities_df.iloc[:, 1]
        if activities_df.shape[1] > 1
        else pd.Series(dtype=float)
    )  # Second column contains values

    # Salting parameters
    loaded_activities.salting_hour = [
        find_value_or_default(
            "salting_hour(1)", header_col, data_col, loaded_activities.salting_hour[0]
        ),
        find_value_or_default(
            "salting_hour(2)", header_col, data_col, loaded_activities.salting_hour[1]
        ),
    ]
    loaded_activities.delay_salting_day = find_value_or_default(
        "delay_salting_day", header_col, data_col, loaded_activities.delay_salting_day
    )
    loaded_activities.check_salting_day = find_value_or_default(
        "check_salting_day", header_col, data_col, loaded_activities.check_salting_day
    )
    loaded_activities.min_temp_salt = find_value_or_default(
        "min_temp_salt", header_col, data_col, loaded_activities.min_temp_salt
    )
    loaded_activities.max_temp_salt = find_value_or_default(
        "max_temp_salt", header_col, data_col, loaded_activities.max_temp_salt
    )
    loaded_activities.precip_rule_salt = find_value_or_default(
        "precip_rule_salt", header_col, data_col, loaded_activities.precip_rule_salt
    )
    loaded_activities.RH_rule_salt = find_value_or_default(
        "RH_rule_salt", header_col, data_col, loaded_activities.RH_rule_salt
    )
    loaded_activities.g_salting_rule = find_value_or_default(
        "g_salting_rule", header_col, data_col, loaded_activities.g_salting_rule
    )
    loaded_activities.salt_mass = find_value_or_default(
        "salt_mass", header_col, data_col, loaded_activities.salt_mass
    )
    loaded_activities.salt_dilution = find_value_or_default(
        "salt_dilution", header_col, data_col, loaded_activities.salt_dilution
    )
    loaded_activities.salt_type_distribution = find_value_or_default(
        "salt_type_distribution",
        header_col,
        data_col,
        loaded_activities.salt_type_distribution,
    )

    # Sanding parameters
    loaded_activities.sanding_hour = [
        find_value_or_default(
            "sanding_hour(1)", header_col, data_col, loaded_activities.sanding_hour[0]
        ),
        find_value_or_default(
            "sanding_hour(2)", header_col, data_col, loaded_activities.sanding_hour[1]
        ),
    ]
    loaded_activities.delay_sanding_day = find_value_or_default(
        "delay_sanding_day", header_col, data_col, loaded_activities.delay_sanding_day
    )
    loaded_activities.check_sanding_day = find_value_or_default(
        "check_sanding_day", header_col, data_col, loaded_activities.check_sanding_day
    )
    loaded_activities.min_temp_sand = find_value_or_default(
        "min_temp_sand", header_col, data_col, loaded_activities.min_temp_sand
    )
    loaded_activities.max_temp_sand = find_value_or_default(
        "max_temp_sand", header_col, data_col, loaded_activities.max_temp_sand
    )
    loaded_activities.precip_rule_sand = find_value_or_default(
        "precip_rule_sand", header_col, data_col, loaded_activities.precip_rule_sand
    )
    loaded_activities.RH_rule_sand = find_value_or_default(
        "RH_rule_sand", header_col, data_col, loaded_activities.RH_rule_sand
    )
    loaded_activities.g_sanding_rule = find_value_or_default(
        "g_sanding_rule", header_col, data_col, loaded_activities.g_sanding_rule
    )
    loaded_activities.sand_mass = find_value_or_default(
        "sand_mass", header_col, data_col, loaded_activities.sand_mass
    )
    loaded_activities.sand_dilution = find_value_or_default(
        "sand_dilution", header_col, data_col, loaded_activities.sand_dilution
    )

    # Ploughing parameters
    loaded_activities.delay_ploughing_hour = find_value_or_default(
        "delay_ploughing_hour",
        header_col,
        data_col,
        loaded_activities.delay_ploughing_hour,
    )
    loaded_activities.ploughing_thresh = find_value_or_default(
        "ploughing_thresh", header_col, data_col, loaded_activities.ploughing_thresh
    )

    # Cleaning parameters
    loaded_activities.delay_cleaning_hour = find_value_or_default(
        "delay_cleaning_hour",
        header_col,
        data_col,
        loaded_activities.delay_cleaning_hour,
    )
    loaded_activities.min_temp_cleaning = find_value_or_default(
        "min_temp_cleaning", header_col, data_col, loaded_activities.min_temp_cleaning
    )
    loaded_activities.clean_with_salting = find_value_or_default(
        "clean_with_salting", header_col, data_col, loaded_activities.clean_with_salting
    )
    loaded_activities.start_month_cleaning = find_value_or_default(
        "start_month_cleaning",
        header_col,
        data_col,
        loaded_activities.start_month_cleaning,
    )
    loaded_activities.end_month_cleaning = find_value_or_default(
        "end_month_cleaning", header_col, data_col, loaded_activities.end_month_cleaning
    )
    loaded_activities.wetting_with_cleaning = find_value_or_default(
        "wetting_with_cleaning",
        header_col,
        data_col,
        loaded_activities.wetting_with_cleaning,
    )
    loaded_activities.efficiency_of_cleaning = find_value_or_default(
        "efficiency_of_cleaning",
        header_col,
        data_col,
        loaded_activities.efficiency_of_cleaning,
    )

    # Binding parameters
    loaded_activities.binding_hour = [
        find_value_or_default(
            "binding_hour(1)", header_col, data_col, loaded_activities.binding_hour[0]
        ),
        find_value_or_default(
            "binding_hour(2)", header_col, data_col, loaded_activities.binding_hour[1]
        ),
    ]
    loaded_activities.delay_binding_day = find_value_or_default(
        "delay_binding_day", header_col, data_col, loaded_activities.delay_binding_day
    )
    loaded_activities.check_binding_day = find_value_or_default(
        "check_binding_day", header_col, data_col, loaded_activities.check_binding_day
    )
    loaded_activities.min_temp_binding = find_value_or_default(
        "min_temp_binding", header_col, data_col, loaded_activities.min_temp_binding
    )
    loaded_activities.max_temp_binding = find_value_or_default(
        "max_temp_binding", header_col, data_col, loaded_activities.max_temp_binding
    )
    loaded_activities.precip_rule_binding = find_value_or_default(
        "precip_rule_binding",
        header_col,
        data_col,
        loaded_activities.precip_rule_binding,
    )
    loaded_activities.RH_rule_binding = find_value_or_default(
        "RH_rule_binding", header_col, data_col, loaded_activities.RH_rule_binding
    )
    loaded_activities.g_binding_rule = find_value_or_default(
        "g_binding_rule", header_col, data_col, loaded_activities.g_binding_rule
    )
    loaded_activities.binding_mass = find_value_or_default(
        "binding_mass", header_col, data_col, loaded_activities.binding_mass
    )
    loaded_activities.binding_dilution = find_value_or_default(
        "binding_dilution", header_col, data_col, loaded_activities.binding_dilution
    )
    loaded_activities.start_month_binding = find_value_or_default(
        "start_month_binding",
        header_col,
        data_col,
        loaded_activities.start_month_binding,
    )
    loaded_activities.end_month_binding = find_value_or_default(
        "end_month_binding", header_col, data_col, loaded_activities.end_month_binding
    )

    return loaded_activities
