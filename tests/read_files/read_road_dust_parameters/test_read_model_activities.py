from src.read_files.read_road_dust_parameters.read_model_activities import (
    read_model_activities,
)
import pandas as pd
from src.config_classes import model_parameters


def test_read_model_activities_basic_values():
    # Create test data using list of lists approach like other tests
    test_data = [
        ["delay_sanding_day (day)", "12"],
        ["check_sanding_day (day)", "13"],
        ["min_temp_sand (C)", "-3"],
    ]

    df = pd.DataFrame(test_data)

    result = read_model_activities(df, model_parameters())

    assert result.delay_sanding_day == 12
    assert result.check_sanding_day == 13
    assert result.min_temp_sand == -3


def test_read_model_activities_manual_values():
    test_data = [
        ["salting_hour(1)", "10"],
        ["salting_hour(2)", "11"],
        ["sanding_hour(1)", "12"],
        ["sanding_hour(2)", "13"],
        ["binding_hour(1)", "14"],
        ["binding_hour(2)", "15"],
    ]

    df = pd.DataFrame(test_data)

    result = read_model_activities(df, model_parameters())

    assert result.salting_hour == [10, 11]
    assert result.sanding_hour == [12, 13]
    assert result.binding_hour == [14, 15]
