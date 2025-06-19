from src.read_files.read_road_dust_parameters.read_model_flags import read_model_flags
import pandas as pd


def test_read_model_flags_partial_data():
    dummy_data = {
        "0": ["road_wear_flag", "tyre_wear_flag", "brake_wear_flag"],
        "1": [10, 20, 30],
    }

    flags_df = pd.DataFrame(dummy_data)

    result = read_model_flags(flags_df)

    assert result.road_wear_flag == 10
    assert result.tyre_wear_flag == 20
    assert result.brake_wear_flag == 30
