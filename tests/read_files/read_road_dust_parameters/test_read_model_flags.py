from src.read_files.read_road_dust_parameters.read_model_flags import read_model_flags
import pandas as pd


def test_read_model_flags_partial_data():
    # Create test data using list of lists approach like other tests
    test_data = [
        ["road_wear_flag", "10"],
        ["tyre_wear_flag", "20"],
        ["brake_wear_flag", "30"],
    ]

    flags_df = pd.DataFrame(test_data)

    result = read_model_flags(flags_df)

    assert result.road_wear_flag == 10
    assert result.tyre_wear_flag == 20
    assert result.brake_wear_flag == 30


def test_read_model_flags_default_values():
    # Create test data using list of lists approach like other tests
    test_data = [
        ["road_wear_flag", "10"],
        ["tyre_wear_flag", "20"],
        ["brake_wear_flag", "30"],
    ]

    flags_df = pd.DataFrame(test_data)

    result = read_model_flags(flags_df)

    assert result.save_type_flag == 1
    assert result.dust_drainage_flag == 2
