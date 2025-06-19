import pandas as pd
import numpy as np
from src.read_files.read_road_dust_input.read_input_traffic import read_input_traffic


# TODO: Fix the tests, they are missing header data
def test_read_input_traffic_basic():
    """Test basic traffic data reading functionality."""
    # fmt: off
    test_data = [
        ["Year", "Hour", "Minute", "N(total)", "N(he)", "N(li)", "N(st,he)", "N(wi,he)", "N(su,he)", "N(st,li)", "N(wi,li)", "N(su,li)", "V_veh(he)", "V_veh(li)"],
        ["2023", "0", "0", "100", "70", "30", "35", "20", "15", "15", "10", "5", "50", "60"],
        ["2023", "1", "0", "120", "80", "40", "40", "25", "15", "20", "12", "8", "55", "65"],
        ["2023", "2", "0", "90", "60", "30", "30", "18", "12", "15", "10", "5", "45", "55"],
    ]
    # fmt: on

    df = pd.DataFrame(test_data)

    result = read_input_traffic(df, nodata=-99.0, print_results=True)

    # Basic assertions
    assert result.n_traffic == 3
    assert len(result.year) == 3
    assert len(result.N_total) == 3
    assert result.N_v.shape == (2, 3)  # num_veh x n_traffic
    assert result.N.shape == (3, 2, 3)  # num_tyre x num_veh x n_traffic
    assert result.V_veh.shape == (2, 3)  # num_veh x n_traffic

    # Check specific values
    np.testing.assert_array_equal(result.year, [2023, 2023, 2023])
    np.testing.assert_array_equal(result.N_total, [100, 120, 90])
    np.testing.assert_array_equal(result.N_v[0, :], [70, 80, 60])  # he
    np.testing.assert_array_equal(result.N_v[1, :], [30, 40, 30])  # li


def test_read_input_traffic_missing_data_forward_fill():
    """Test forward fill functionality with simple missing data."""
    # fmt: off
    test_data = [
        ["Year", "Hour", "Minute", "N(total)", "N(he)", "N(li)", "N(st,he)", "N(wi,he)", "N(su,he)", "N(st,li)", "N(wi,li)", "N(su,li)", "V_veh(he)", "V_veh(li)"],
        ["2023", "0", "0", "100", "70", "30", "35", "20", "15", "15", "10", "5", "50", "60"],
        ["2023", "1", "0", "120", "80", "40", "40", "25", "15", "20", "12", "8", "55", "65"],
        ["2023", "2", "0", "90", "60", "30", "30", "18", "12", "15", "10", "5", "45", "55"],
    ]
    # fmt: on
    df = pd.DataFrame(test_data)

    result = read_input_traffic(df, nodata=-99.0, print_results=True)

    assert len(result.N_total_nodata) == 1
    assert 1 in result.N_total_nodata  # Index 1 had missing N_total

    assert result.N_total[1] != -99.0
    assert result.N_total[1] == 100 or result.N_total[1] > 0


def test_read_input_traffic_missing_data_complex():
    """Test handling of missing data with multiple missing values (original test)."""
    # fmt: off
    test_data = [
        ["Year", "Hour", "Minute", "N(total)", "N(he)", "N(li)", "N(st,he)", "N(wi,he)", "N(su,he)", "N(st,li)", "N(wi,li)", "N(su,li)", "V_veh(he)", "V_veh(li)"],
        ["2023", "0", "0", "100", "70", "30", "35", "20", "15", "15", "10", "5", "50", "60"],
        ["2023", "1", "0", "120", "80", "40", "40", "25", "15", "20", "12", "8", "55", "65"],
        ["2023", "2", "0", "90", "60", "30", "30", "18", "12", "15", "10", "5", "45", "55"],
    ]
    # fmt: on

    df = pd.DataFrame(test_data)

    result = read_input_traffic(df, nodata=-99.0, print_results=True)

    assert len(result.N_total_nodata) == 1
    assert 1 in result.N_total_nodata

    assert result.N_total[1] != -99.0

    assert 2 in result.V_veh_nodata[1]
