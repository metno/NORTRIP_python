import numpy as np
from input_classes import converted_data
import constants


def test_converted_data_initialization():
    """Test basic initialization of converted_data dataclass."""
    converted = converted_data()

    # Check initial values
    assert converted.n_date == 0
    assert converted.n_roads == constants.n_roads
    assert converted.nodata == -99.0

    # Check array shapes with n_date=0
    assert converted.date_data.shape == (constants.num_date_index, 0, constants.n_roads)
    assert converted.traffic_data.shape == (
        constants.num_traffic_index,
        0,
        constants.n_roads,
    )
    assert converted.meteo_data.shape == (
        constants.num_meteo_index,
        0,
        constants.n_roads,
    )
    assert converted.activity_data.shape == (
        constants.num_activity_index,
        0,
        constants.n_roads,
    )
    assert converted.f_conc.shape == (0, constants.n_roads)
    assert converted.f_dis.shape == (0, constants.n_roads)
