import pandas as pd
import numpy as np
from src.read_files.read_road_dust_input.read_input_airquality import (
    read_input_airquality,
)
import src.constants as constants


def test_read_input_airquality_basic():
    """Test basic functionality of air quality data reading."""
    # fmt: off
    test_data = [
        ["PM10_obs", "PM10_background", "PM25_obs", "PM25_background", "NOX_obs", "NOX_background", "NOX_emis", "EP_emis", "Salt_obs(na)", "Disp_fac"],
        ["25.5", "15.0", "18.2", "12.0", "45.0", "25.0", "0.8", "0.6", "2.1", "1.2"],
        ["30.1", "16.5", "22.8", "13.5", "52.3", "28.1", "0.9", "0.7", "2.5", "1.1"],
        ["28.7", "14.8", "20.5", "11.8", "48.9", "26.5", "0.85", "0.65", "2.3", "1.15"],
    ]
    # fmt: on

    df = pd.DataFrame(test_data)
    result = read_input_airquality(
        df,
        ospm_df=None,
        nodata=-99.0,
        traffic_date_num=np.array([]),
        traffic_hour=np.array([]),
        N_total_nodata=[],
        N_good_data=np.array([]),
    )

    # Check basic structure
    assert result.n_date == 3
    assert result.PM_obs.shape == (constants.num_size, 3)
    assert result.PM_background.shape == (constants.num_size, 3)
    assert len(result.NOX_obs) == 3
    assert len(result.NOX_background) == 3
    assert len(result.NOX_emis) == 3
    assert len(result.EP_emis) == 3

    # Check data values
    assert result.PM_obs[constants.pm_10, 0] == 25.5
    assert result.PM_background[constants.pm_10, 0] == 15.0
    assert result.PM_obs[constants.pm_25, 0] == 18.2
    assert result.PM_background[constants.pm_25, 0] == 12.0
    assert result.NOX_obs[0] == 45.0
    assert result.NOX_background[0] == 25.0
    assert result.NOX_emis[0] == 0.8
    assert result.EP_emis[0] == 0.6

    # Check availability flags
    assert result.NOX_emis_available == 1
    assert result.EP_emis_available == 1
    assert result.Salt_obs_available[constants.na] == 1
    assert result.f_dis_available == 1


def test_read_input_airquality_empty_data():
    """Test handling of empty air quality data."""
    df = pd.DataFrame()
    result = read_input_airquality(
        df,
        ospm_df=None,
        nodata=-99.0,
        traffic_date_num=np.array([]),
        traffic_hour=np.array([]),
        N_total_nodata=[],
        N_good_data=np.array([]),
    )

    # Should return default initialized dataclass
    assert result.n_date == 0
    assert result.NOX_emis_available == 0
    assert result.EP_emis_available == 0
    assert result.OSPM_data_exists == 0


def test_read_input_airquality_minimal_data():
    """Test with minimal required data."""
    # fmt: off
    test_data = [
        ["PM10_obs", "PM10_background", "NOX_obs", "NOX_background"],
        ["25.5", "15.0", "45.0", "25.0"],
        ["30.1", "16.5", "52.3", "28.1"],
    ]
    # fmt: on

    df = pd.DataFrame(test_data)
    result = read_input_airquality(
        df,
        ospm_df=None,
        nodata=-99.0,
        traffic_date_num=np.array([]),
        traffic_hour=np.array([]),
        N_total_nodata=[],
        N_good_data=np.array([]),
    )

    # Check that basic data is read
    assert result.n_date == 2
    assert result.PM_obs[constants.pm_10, 0] == 25.5
    assert result.NOX_obs[0] == 45.0

    # Check that optional data is properly initialized
    assert result.NOX_emis_available == 0
    assert result.EP_emis_available == 0
    assert result.Salt_obs_available[constants.na] == 0
    assert result.f_dis_available == 0

    # Check that missing data arrays are filled with nodata
    assert np.all(result.NOX_emis == -99.0)
    assert np.all(result.EP_emis == -99.0)


def test_read_input_airquality_missing_columns():
    """Test handling of missing optional columns."""
    # fmt: off
    test_data = [
        ["PM10_obs", "PM10_background", "NOX_obs", "NOX_background"],  # Missing optional columns
        ["25.5", "15.0", "45.0", "25.0"],
        ["30.1", "16.5", "52.3", "28.1"],
        ["28.7", "14.8", "48.9", "26.5"],
    ]
    # fmt: on

    df = pd.DataFrame(test_data)
    result = read_input_airquality(
        df,
        ospm_df=None,
        nodata=-99.0,
        traffic_date_num=np.array([]),
        traffic_hour=np.array([]),
        N_total_nodata=[],
        N_good_data=np.array([]),
    )

    # Check that data is read correctly
    assert result.n_date == 3
    assert result.PM_obs[constants.pm_10, 0] == 25.5
    assert result.NOX_obs[0] == 45.0

    # Check that missing columns result in unavailable flags
    assert result.NOX_emis_available == 0
    assert result.EP_emis_available == 0
    assert result.Salt_obs_available[constants.na] == 0
    assert result.f_dis_available == 0


def test_read_input_airquality_nan_handling():
    """Test handling of NaN values in data."""
    # fmt: off
    test_data = [
        ["PM10_obs", "PM10_background", "PM25_obs", "PM25_background", "NOX_obs", "NOX_background", "NOX_emis", "EP_emis"],
        ["25.5", "15.0", "18.2", "12.0", "45.0", "25.0", "0.8", "0.6"],
        ["NAN", "NaN", "22.8", "NaN", "52.3", "28.1", "NaN", "0.7"],
        ["28.7", "14.8", "20.5", "11.8", "48.9", "26.5", "0.85", "NaN"],
    ]
    # fmt: on

    df = pd.DataFrame(test_data)
    result = read_input_airquality(
        df,
        ospm_df=None,
        nodata=-99.0,
        traffic_date_num=np.array([]),
        traffic_hour=np.array([]),
        N_total_nodata=[],
        N_good_data=np.array([]),
    )

    # Check that NaN values are converted to 0.0 by safe_float, then we don't have special logic for 0.0 here
    # The "NaN" string gets converted to 0.0 by safe_float, not to nodata
    assert (
        result.PM_obs[constants.pm_10, 1] == -99.0
    )  # Was "NaN" string, converted to 0.0
    assert (
        result.PM_background[constants.pm_25, 1] == -99.0
    )  # Was "NaN" string, converted to 0.0
    assert result.NOX_emis[1] == -99.0  # Was "NaN" string, converted to 0.0
    assert result.EP_emis[2] == -99.0  # Was "NaN" string, converted to 0.0

    # Check that valid values are preserved
    assert result.PM_obs[constants.pm_10, 0] == 25.5
    assert result.NOX_obs[1] == 52.3


def test_read_input_airquality_net_concentration_calculation():
    """Test calculation of net concentrations (obs - background)."""
    # fmt: off
    test_data = [
        ["PM10_obs", "PM10_background", "PM25_obs", "PM25_background", "NOX_obs", "NOX_background"],
        ["25.5", "15.0", "18.2", "12.0", "45.0", "25.0"],  # Net: 10.5, 6.2, 20.0
        ["30.1", "16.5", "22.8", "13.5", "52.3", "28.1"],  # Net: 13.6, 9.3, 24.2
        ["12.0", "14.8", "10.5", "11.8", "20.0", "26.5"],  # Net: negative (should be nodata)
    ]
    # fmt: on

    df = pd.DataFrame(test_data)
    result = read_input_airquality(
        df,
        ospm_df=None,
        nodata=-99.0,
        traffic_date_num=np.array([]),
        traffic_hour=np.array([]),
        N_total_nodata=[],
        N_good_data=np.array([]),
    )

    # Check positive net concentrations
    np.testing.assert_almost_equal(
        result.PM_obs_net[constants.pm_10, 0], 10.5, decimal=1
    )
    np.testing.assert_almost_equal(
        result.PM_obs_net[constants.pm_25, 0], 6.2, decimal=1
    )
    np.testing.assert_almost_equal(result.NOX_obs_net[0], 20.0, decimal=1)

    np.testing.assert_almost_equal(
        result.PM_obs_net[constants.pm_10, 1], 13.6, decimal=1
    )
    np.testing.assert_almost_equal(
        result.PM_obs_net[constants.pm_25, 1], 9.3, decimal=1
    )
    np.testing.assert_almost_equal(result.NOX_obs_net[1], 24.2, decimal=1)

    # Check that negative net concentrations are set to nodata
    assert result.PM_obs_net[constants.pm_10, 2] == -99.0
    assert result.PM_obs_net[constants.pm_25, 2] == -99.0
    assert result.NOX_obs_net[2] == -99.0

    # Check that background concentrations are copied to PM_obs_bg
    assert result.PM_obs_bg[constants.pm_10, 0] == 15.0
    assert result.PM_obs_bg[constants.pm_25, 0] == 12.0


def test_read_input_airquality_daily_average_filling():
    """Test daily average filling for missing emission data."""
    # Create traffic data for pattern
    traffic_date_num = np.array(
        [738000.0, 738000.33, 738000.67, 738001.0, 738001.33, 738001.67]
    )
    traffic_hour = np.array([0, 8, 16, 0, 8, 16])
    N_total_nodata = [3, 5]  # Missing data at indices 3 and 5
    N_good_data = np.array([0, 1, 2])  # Good data at indices 0, 1, 2

    # fmt: off
    test_data = [
        ["NOX_emis", "EP_emis"],
        ["0.5", "0.4"],    # Hour 0
        ["0.8", "0.6"],    # Hour 8  
        ["0.6", "0.5"],    # Hour 16
        ["-99", "-99"],    # Hour 0 (missing - should get 0.5, 0.4)
        ["0.9", "0.65"],   # Hour 8
        ["-99", "-99"],    # Hour 16 (missing - should get 0.6, 0.5)
    ]
    # fmt: on

    df = pd.DataFrame(test_data)
    result = read_input_airquality(
        df,
        ospm_df=None,
        nodata=-99.0,
        traffic_date_num=traffic_date_num,
        traffic_hour=traffic_hour,
        N_total_nodata=N_total_nodata,
        N_good_data=N_good_data,
    )

    # Check that daily averages were applied
    # Hour 0 average: 0.5 (from index 0)
    # Hour 16 average: 0.6 (from index 2)
    assert result.NOX_emis[3] == 0.5  # Filled with hour 0 average
    assert result.NOX_emis[5] == 0.6  # Filled with hour 16 average
    assert result.EP_emis[3] == 0.4  # Filled with hour 0 average
    assert result.EP_emis[5] == 0.5  # Filled with hour 16 average


def test_read_input_airquality_ospm_data():
    """Test reading OSMP data when available."""
    # Main air quality data
    # fmt: off
    airquality_data = [
        ["PM10_obs", "PM10_background", "NOX_obs", "NOX_background"],
        ["25.5", "15.0", "45.0", "25.0"],
        ["30.1", "16.5", "52.3", "28.1"],
    ]

    # OSPM data
    osmp_data = [
        ["FFospm(m/s)", "DDospm(deg)", "TKospm(degK)", "Globalradiationospm(W/m^2)", "Cbackgroundospm(ug/m^3)", "N(li)ospm", "N(he)ospm", "V_veh(li)ospm(km/hr)", "V_veh(he)ospm(km/hr)", "Cemisospm(ug/m/s)"],
        ["3.2", "180", "285.5", "450", "15.2", "50", "30", "55", "65", "0.8"],
        ["2.8", "165", "287.1", "520", "16.1", "45", "35", "60", "70", "0.9"],
    ]
    # fmt: on

    airquality_df = pd.DataFrame(airquality_data)
    osmp_df = pd.DataFrame(osmp_data)

    result = read_input_airquality(
        airquality_df,
        ospm_df=osmp_df,
        nodata=-99.0,
        traffic_date_num=np.array([]),
        traffic_hour=np.array([]),
        N_total_nodata=[],
        N_good_data=np.array([]),
    )

    # Check that OSPM data exists
    assert result.OSPM_data_exists == 1

    # Check OSPM meteorological data
    assert len(result.U_mast_ospm_orig) == 2
    assert result.U_mast_ospm_orig[0] == 3.2
    assert result.wind_dir_ospm_orig[0] == 180
    assert result.TK_ospm_orig[0] == 285.5
    assert result.GlobalRad_ospm_orig[0] == 450

    # Check OSPM concentration data
    assert result.cNOx_b_ospm_orig[0] == 15.2
    assert result.qNOX_ospm_orig[0] == 0.8

    # Check OSPM traffic data
    assert result.NNp_ospm_orig[0] == 50  # N(li)ospm
    assert result.NNt_ospm_orig[0] == 30  # N(he)ospm
    assert result.Vp_ospm_orig[0] == 55  # V_veh(li)ospm
    assert result.Vt_ospm_orig[0] == 65  # V_veh(he)ospm


def test_read_input_airquality_osmp_missing_data():
    """Test OSMP data with missing values."""
    # Main air quality data
    # fmt: off
    airquality_data = [
        ["PM10_obs", "PM10_background"],
        ["25.5", "15.0"],
    ]

    # OSMP data with missing values
    osmp_data = [
        ["FFospm(m/s)", "DDospm(deg)", "TKospm(degK)"],
        ["3.2", "-99", "285.5"],
    ]
    # fmt: on

    airquality_df = pd.DataFrame(airquality_data)
    osmp_df = pd.DataFrame(osmp_data)

    result = read_input_airquality(
        airquality_df,
        ospm_df=osmp_df,
        nodata=-99.0,
        traffic_date_num=np.array([]),
        traffic_hour=np.array([]),
        N_total_nodata=[],
        N_good_data=np.array([]),
    )

    # Check that OSMP data exists
    assert result.OSPM_data_exists == 1

    # Check that missing values are handled
    assert result.U_mast_ospm_orig[0] == 3.2  # Valid value
    assert result.wind_dir_ospm_orig[0] == -99.0  # Missing value
    assert result.TK_ospm_orig[0] == 285.5  # Valid value


def test_read_input_airquality_no_osmp_data():
    """Test when no OSMP data is provided."""
    # fmt: off
    test_data = [
        ["PM10_obs", "PM10_background", "NOX_obs", "NOX_background"],
        ["25.5", "15.0", "45.0", "25.0"],
        ["30.1", "16.5", "52.3", "28.1"],
    ]
    # fmt: on

    df = pd.DataFrame(test_data)
    result = read_input_airquality(
        df,
        ospm_df=None,
        nodata=-99.0,
        traffic_date_num=np.array([]),
        traffic_hour=np.array([]),
        N_total_nodata=[],
        N_good_data=np.array([]),
    )

    # Check that OSMP data does not exist
    assert result.OSPM_data_exists == 0

    # Check that OSMP arrays are empty
    assert len(result.U_mast_ospm_orig) == 0
    assert len(result.wind_dir_ospm_orig) == 0
    assert len(result.TK_ospm_orig) == 0
    assert len(result.GlobalRad_ospm_orig) == 0


def test_read_input_airquality_salt_observations():
    """Test reading salt observation data."""
    # fmt: off
    test_data = [
        ["PM10_obs", "PM10_background", "Salt_obs(na)"],
        ["25.5", "15.0", "2.1"],
        ["30.1", "16.5", "2.5"],
        ["28.7", "14.8", "NaN"],  # Missing salt data - safe_float converts "NaN" to 0.0
    ]
    # fmt: on

    df = pd.DataFrame(test_data)
    result = read_input_airquality(
        df,
        ospm_df=None,
        nodata=-99.0,
        traffic_date_num=np.array([]),
        traffic_hour=np.array([]),
        N_total_nodata=[],
        N_good_data=np.array([]),
    )

    # Check salt data availability
    assert result.Salt_obs_available[constants.na] == 1

    # Check salt values
    assert result.Salt_obs[constants.na, 0] == 2.1
    assert result.Salt_obs[constants.na, 1] == 2.5
    assert (
        result.Salt_obs[constants.na, 2] == -99.0
    )  # "NaN" string converted to -99.0 by safe_float


def test_read_input_airquality_emission_availability():
    """Test emission data availability detection."""
    # Test with all emission data as nodata
    # fmt: off
    test_data_no_emis = [
        ["PM10_obs", "PM10_background", "NOX_emis", "EP_emis"],
        ["25.5", "15.0", "-99", "-99"],
        ["30.1", "16.5", "-99", "-99"],
    ]
    # fmt: on

    df = pd.DataFrame(test_data_no_emis)
    result = read_input_airquality(
        df,
        ospm_df=None,
        nodata=-99.0,
        traffic_date_num=np.array([]),
        traffic_hour=np.array([]),
        N_total_nodata=[],
        N_good_data=np.array([]),
    )

    # Check that emission data is marked as unavailable
    assert result.NOX_emis_available == 0
    assert result.EP_emis_available == 0

    # Test with some valid emission data
    # fmt: off
    test_data_some_emis = [
        ["PM10_obs", "PM10_background", "NOX_emis", "EP_emis"],
        ["25.5", "15.0", "0.8", "-99"],
        ["30.1", "16.5", "-99", "0.6"],
    ]
    # fmt: on

    df2 = pd.DataFrame(test_data_some_emis)
    result2 = read_input_airquality(
        df2,
        ospm_df=None,
        nodata=-99.0,
        traffic_date_num=np.array([]),
        traffic_hour=np.array([]),
        N_total_nodata=[],
        N_good_data=np.array([]),
    )

    # Check that emission data with some valid values is marked as available
    assert result2.NOX_emis_available == 1
    assert result2.EP_emis_available == 1


def test_read_input_airquality_array_dimensions():
    """Test that all arrays have correct dimensions according to constants."""
    # fmt: off
    test_data = [
        ["PM10_obs", "PM10_background", "PM25_obs", "PM25_background", "NOX_obs", "NOX_background", "NOX_emis", "EP_emis", "Salt_obs(na)", "Disp_fac"],
        ["25.5", "15.0", "18.2", "12.0", "45.0", "25.0", "0.8", "0.6", "2.1", "1.2"],
        ["30.1", "16.5", "22.8", "13.5", "52.3", "28.1", "0.9", "0.7", "2.5", "1.1"],
        ["28.7", "14.8", "20.5", "11.8", "48.9", "26.5", "0.85", "0.65", "2.3", "1.15"],
    ]
    # fmt: on

    df = pd.DataFrame(test_data)
    result = read_input_airquality(
        df,
        ospm_df=None,
        nodata=-99.0,
        traffic_date_num=np.array([]),
        traffic_hour=np.array([]),
        N_total_nodata=[],
        N_good_data=np.array([]),
    )

    n_date = result.n_date

    # Check array dimensions match constants
    assert result.PM_obs.shape == (constants.num_size, n_date)
    assert result.PM_background.shape == (constants.num_size, n_date)
    assert result.PM_obs_net.shape == (constants.num_size, n_date)
    assert result.PM_obs_bg.shape == (constants.num_size, n_date)
    assert result.Salt_obs.shape == (constants.num_salt, n_date)
    assert result.Salt_obs_available.shape == (constants.num_salt,)

    # Check 1D arrays
    assert len(result.NOX_obs) == n_date
    assert len(result.NOX_background) == n_date
    assert len(result.NOX_emis) == n_date
    assert len(result.EP_emis) == n_date
    assert len(result.NOX_obs_net) == n_date
    assert len(result.f_dis_input) == n_date


def test_read_input_airquality_edge_cases():
    """Test edge cases and boundary conditions."""
    # Test with zero concentrations
    # fmt: off
    test_data = [
        ["PM10_obs", "PM10_background", "NOX_obs", "NOX_background"],
        ["0.0", "0.0", "0.0", "0.0"],  # All zeros
        ["10.0", "15.0", "20.0", "25.0"],  # Background higher than obs
    ]
    # fmt: on

    df = pd.DataFrame(test_data)
    result = read_input_airquality(
        df,
        ospm_df=None,
        nodata=-99.0,
        traffic_date_num=np.array([]),
        traffic_hour=np.array([]),
        N_total_nodata=[],
        N_good_data=np.array([]),
    )

    # Check zero concentrations
    assert result.PM_obs[constants.pm_10, 0] == 0.0
    assert result.NOX_obs[0] == 0.0

    # Check that negative net concentrations are set to nodata
    assert result.PM_obs_net[constants.pm_10, 1] == -99.0  # 10 - 15 = -5, set to nodata
    assert result.NOX_obs_net[1] == -99.0  # 20 - 25 = -5, set to nodata


def test_read_input_airquality_data_consistency():
    """Test data consistency validation."""
    # fmt: off
    test_data = [
        ["PM10_obs", "PM10_background", "PM25_obs", "PM25_background", "NOX_obs", "NOX_background"],
        ["25.5", "15.0", "18.2", "12.0", "45.0", "25.0"],
        ["30.1", "16.5", "22.8", "13.5", "52.3", "28.1"],
        ["28.7", "14.8", "20.5", "11.8", "48.9", "26.5"],
    ]
    # fmt: on

    df = pd.DataFrame(test_data)
    result = read_input_airquality(
        df,
        ospm_df=None,
        nodata=-99.0,
        traffic_date_num=np.array([]),
        traffic_hour=np.array([]),
        N_total_nodata=[],
        N_good_data=np.array([]),
    )

    # Check that net concentrations are calculated consistently
    for i in range(result.n_date):
        # PM10 net calculation
        if (
            result.PM_obs[constants.pm_10, i] != -99.0
            and result.PM_background[constants.pm_10, i] != -99.0
        ):
            expected_net = (
                result.PM_obs[constants.pm_10, i]
                - result.PM_background[constants.pm_10, i]
            )
            if expected_net > 0:
                np.testing.assert_almost_equal(
                    result.PM_obs_net[constants.pm_10, i], expected_net, decimal=1
                )

        # NOX net calculation
        if result.NOX_obs[i] != -99.0 and result.NOX_background[i] != -99.0:
            expected_net = result.NOX_obs[i] - result.NOX_background[i]
            if expected_net > 0:
                np.testing.assert_almost_equal(
                    result.NOX_obs_net[i], expected_net, decimal=1
                )


def test_read_input_airquality_comprehensive_missing_data():
    """Test comprehensive missing data scenarios."""
    # fmt: off
    test_data = [
        ["PM10_obs", "PM10_background", "PM25_obs", "PM25_background", "NOX_obs", "NOX_background", "NOX_emis", "EP_emis"],
        ["25.5", "15.0", "18.2", "12.0", "45.0", "25.0", "0.8", "0.6"],
        ["-99", "-99", "-99", "-99", "-99", "-99", "-99", "-99"],  # All missing
        ["28.7", "14.8", "20.5", "11.8", "48.9", "26.5", "0.85", "0.65"],
    ]
    # fmt: on

    df = pd.DataFrame(test_data)
    result = read_input_airquality(
        df,
        ospm_df=None,
        nodata=-99.0,
        traffic_date_num=np.array([]),
        traffic_hour=np.array([]),
        N_total_nodata=[],
        N_good_data=np.array([]),
    )

    # Check that missing data is properly handled
    assert result.PM_obs[constants.pm_10, 1] == -99.0
    assert result.PM_background[constants.pm_10, 1] == -99.0
    assert result.NOX_obs[1] == -99.0
    assert result.NOX_background[1] == -99.0

    # Check that net concentrations for missing data are set to nodata (-99.0), not NaN
    assert result.PM_obs_net[constants.pm_10, 1] == -99.0
    assert result.NOX_obs_net[1] == -99.0

    # Check that valid data is preserved
    assert result.PM_obs[constants.pm_10, 0] == 25.5
    assert result.PM_obs[constants.pm_10, 2] == 28.7
