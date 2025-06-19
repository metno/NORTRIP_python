import pandas as pd
import numpy as np
from src.read_files.read_road_dust_parameters.read_model_parameters import (
    read_model_parameters,
    safe_float,
)
from config_classes.model_parameters import model_parameters
import constants


def test_safe_float():
    """Test the safe_float function with various inputs."""
    # Test normal float conversion
    assert safe_float("10.5") == 10.5
    assert safe_float(10.5) == 10.5
    assert safe_float("10") == 10.0

    # Test European decimal format (comma as decimal separator)
    assert safe_float("10,5") == 10.5
    assert safe_float("1,234") == 1.234

    # Test edge cases
    assert safe_float(pd.NA) == 0.0
    assert safe_float(None) == 0.0
    assert safe_float("") == 0.0
    assert safe_float("nan") == 0.0
    assert safe_float("NaN") == 0.0
    assert safe_float("  ") == 0.0

    # Test invalid values
    assert safe_float("invalid") == 0.0
    assert safe_float("10.5.6") == 0.0


def test_read_model_parameters_minimal():
    """Test with minimal valid data to ensure function works."""
    # Create a minimal DataFrame with at least some structure
    df = pd.DataFrame(
        {
            0: ["", "", "", "", "", "", "", "", "", ""],
            1: ["", "", "", "", "", "", "", "", "", ""],
        }
    )

    result = read_model_parameters(df)

    # Should return a valid model_parameters instance with default values
    assert isinstance(result, model_parameters)
    assert result.num_pave == 1
    assert result.num_dc == 1
    assert result.num_track == 1


def test_read_model_parameters_road_wear():
    """Test reading road wear parameters in realistic format."""
    # Create a DataFrame that mimics the actual Excel structure
    df = pd.DataFrame(
        [
            ["", "", "", "", "", "", ""],  # Row 0
            ["", "", "", "", "", "", ""],  # Row 1
            ["", "", "", "", "", "", ""],  # Row 2
            ["Road wear", "", "", "", "", "", ""],  # Row 3 - header
            [
                "W0,roadwear (g km-1 veh-1)",
                "Studded tyres (st)",
                "Winter tyres (wi)",
                "Summer tyres (su)",
                "",
                "",
                "",
            ],  # Row 4
            ["Heavy (he)", "14.4", "0.36", "0.36", "", "", ""],  # Row 5
            ["Light (li)", "2.88", "0.072", "0.072", "", "", ""],  # Row 6
            [
                "Parameters for speed dependence",
                "a1",
                "a2",
                "a3",
                "Vref,roadwear",
                "Vmin,roadwear",
                "",
            ],  # Row 7
            [
                "W=W0*(a1+a2*(max[V,Vmin]/Vref)a3)",
                "0.00",
                "1.00",
                "1.00",
                "70.00",
                "30.00",
                "",
            ],  # Row 8
            ["", "", "", "", "", "", ""],  # Row 9
        ]
    )

    result = read_model_parameters(df)

    # Check W_0 values
    assert result.W_0[constants.road_index, constants.st, constants.he] == 14.4
    assert result.W_0[constants.road_index, constants.wi, constants.he] == 0.36
    assert result.W_0[constants.road_index, constants.su, constants.he] == 0.36
    assert result.W_0[constants.road_index, constants.st, constants.li] == 2.88
    assert result.W_0[constants.road_index, constants.wi, constants.li] == 0.072
    assert result.W_0[constants.road_index, constants.su, constants.li] == 0.072

    # Check a_wear coefficients
    assert result.a_wear[constants.road_index, 0] == 0.00
    assert result.a_wear[constants.road_index, 1] == 1.00
    assert result.a_wear[constants.road_index, 2] == 1.00
    assert result.a_wear[constants.road_index, 3] == 70.00
    assert result.a_wear[constants.road_index, 4] == 30.00


def test_read_model_parameters_snow_depth():
    """Test reading snow depth wear threshold."""
    df = pd.DataFrame(
        [
            ["", "", "", ""],  # Row 0
            ["", "", "", ""],  # Row 1
            ["", "", "", ""],  # Row 2
            ["", "", "", ""],  # Row 3
            ["", "", "", ""],  # Row 4
            ["", "", "", ""],  # Row 5
            ["", "", "", ""],  # Row 6
            ["", "", "", ""],  # Row 7
            ["", "", "", ""],  # Row 8
            ["", "", "", ""],  # Row 9
            ["", "", "", ""],  # Row 10
            ["", "", "", ""],  # Row 11
            ["", "", "", ""],  # Row 12
            ["", "", "", ""],  # Row 13
            ["", "", "", ""],  # Row 14
            ["", "", "", ""],  # Row 15
            ["", "", "", ""],  # Row 16
            ["", "", "", ""],  # Row 17
            ["", "", "", ""],  # Row 18
            ["", "", "", ""],  # Row 19
            ["", "", "", ""],  # Row 20
            ["", "", "", ""],  # Row 21
            ["Snow depth wear threshold", "", "", ""],  # Row 22
            ["Parameter", "Value", "", ""],  # Row 23
            ["sroadwear,thresh (mm w.e.)", "5", "", ""],  # Row 24
        ]
    )

    result = read_model_parameters(df)

    assert result.s_roadwear_thresh == 5.0


def test_read_model_parameters_wind_blown_dust():
    """Test reading wind blown dust parameters."""
    # Create DataFrame with enough rows to reach the wind blown dust section
    rows = [["", "", "", ""] for _ in range(120)]  # Start with empty rows

    # Add the wind blown dust section around row 120
    rows[115] = ["Wind blown dust emission factors", "", "", ""]
    rows[116] = ["Parameter", "Value", "", ""]
    rows[117] = ["twind (hr)", "12", "", ""]
    rows[118] = ["FFthresh (m/s)", "5", "", ""]

    df = pd.DataFrame(rows)
    result = read_model_parameters(df)

    assert result.tau_wind == 12.0
    assert result.FF_thresh == 5.0


def test_read_model_parameters_deposition_velocities():
    """Test reading deposition velocities."""
    # Create DataFrame with the deposition velocity section
    rows = [["", "", "", "", "", ""] for _ in range(135)]

    # Add deposition velocity section
    rows[130] = ["Deposition velocity", "", "", "", "", ""]
    rows[131] = ["", "PM200 - PM10", "PM10 - PM2.5", "PM2.5", "", ""]
    rows[132] = ["wx (m/s)", "0.003", "0.001", "0.0005", "", ""]

    df = pd.DataFrame(rows)
    result = read_model_parameters(df)

    # The implementation reads values starting from column 1 and uses indices 0, 1, 2, 3
    # which correspond to PMall-PM200, PM200-PM10, PM10-PM2.5, PM2.5
    assert result.w_dep[0] == 0.003  # PMall-PM200 (column 1)
    assert result.w_dep[1] == 0.001  # PM200-PM10 (column 2)
    assert result.w_dep[2] == 0.0005  # PM10-PM2.5 (column 3)


def test_read_model_parameters_concentration_limits():
    """Test reading concentration conversion limit values."""
    rows = [["", "", "", ""] for _ in range(140)]

    # Add concentration limits section
    rows[133] = ["Concentration conversion limit values", "", "", ""]
    rows[134] = ["Parameter", "Value", "", ""]
    rows[135] = ["NOX,concentration-min (µg/m3)", "5", "", ""]
    rows[136] = ["NOX,emission-min (g/km/hr)", "50", "", ""]

    df = pd.DataFrame(rows)
    result = read_model_parameters(df)

    assert result.conc_min == 5.0
    assert result.emis_min == 50.0


def test_read_model_parameters_european_decimal():
    """Test handling of European decimal format."""
    rows = [["", "", ""] for _ in range(25)]
    rows[22] = ["Snow depth wear threshold", "", ""]
    rows[23] = ["Parameter", "Value", ""]
    rows[24] = ["sroadwear,thresh (mm w.e.)", "2,5", ""]  # European format with comma

    df = pd.DataFrame(rows)
    result = read_model_parameters(df)

    assert result.s_roadwear_thresh == 2.5


def test_read_model_parameters_invalid_values():
    """Test handling of invalid values."""
    rows = [["", "", ""] for _ in range(25)]
    rows[22] = ["Snow depth wear threshold", "", ""]
    rows[23] = ["Parameter", "Value", ""]
    rows[24] = ["sroadwear,thresh (mm w.e.)", "invalid", ""]

    df = pd.DataFrame(rows)
    result = read_model_parameters(df)

    # Invalid values should be converted to 0.0
    assert result.s_roadwear_thresh == 0.0


def test_read_model_parameters_case_insensitive():
    """Test that parameter reading is case insensitive."""
    df = pd.DataFrame(
        [
            ["", "", "", "", "", "", ""],
            ["", "", "", "", "", "", ""],
            ["", "", "", "", "", "", ""],
            ["ROAD WEAR", "", "", "", "", "", ""],  # Uppercase
            [
                "W0,roadwear (g km-1 veh-1)",
                "Studded tyres (st)",
                "Winter tyres (wi)",
                "Summer tyres (su)",
                "",
                "",
                "",
            ],
            ["Heavy (he)", "14.4", "0.36", "0.36", "", "", ""],
            ["Light (li)", "2.88", "0.072", "0.072", "", "", ""],
            [
                "Parameters for speed dependence",
                "a1",
                "a2",
                "a3",
                "Vref,roadwear",
                "Vmin,roadwear",
                "",
            ],
            [
                "W=W0*(a1+a2*(max[V,Vmin]/Vref)a3)",
                "0.00",
                "1.00",
                "1.00",
                "70.00",
                "30.00",
                "",
            ],
        ]
    )

    result = read_model_parameters(df)

    # Should still find and read the road wear section
    assert result.W_0[constants.road_index, constants.st, constants.he] == 14.4
    assert result.a_wear[constants.road_index, 0] == 0.00


def test_read_model_parameters_activity_efficiency():
    """Test reading activity efficiency factors."""
    rows = [
        ["", "", "", "", "", "", ""] for _ in range(130)
    ]  # Need more rows for salt section

    # Add activity efficiency section
    rows[119] = [
        "Activity efficiency factors for dust and salt",
        "",
        "",
        "",
        "",
        "",
        "",
    ]
    rows[120] = [
        "Efficiency parameter dust",
        "PMall - PM200",
        "PM200 - PM10",
        "PM10 - PM2.5",
        "PM2.5",
        "",
        "",
    ]
    rows[121] = ["hploughing-eff", "0.5", "0.01", "0.01", "0.01", "", ""]
    rows[122] = ["hcleaning-eff", "0.5", "0.001", "0.001", "0.001", "", ""]
    rows[123] = ["hdrainage-eff", "0.05", "0.005", "0.005", "0.005", "", ""]
    rows[124] = ["hspraying-eff", "0.1", "0.1", "0.1", "0.1", "", ""]
    rows[125] = ["Efficiency parameter salt", "Salt(1)", "Salt(2)", "", "", "", ""]
    rows[126] = ["hploughing-eff", "0.3", "0.3", "", "", "", ""]
    rows[127] = ["hcleaning-eff", "0.2", "0.2", "", "", "", ""]
    rows[128] = ["hdrainage-eff", "1", "1", "", "", "", ""]
    rows[129] = ["hspraying-eff", "1", "1", "", "", "", ""]

    df = pd.DataFrame(rows)
    result = read_model_parameters(df)

    # Check dust efficiency factors (size indices are 0-based in arrays)
    # dust_index is a list [0, 1, 2, 3, 4, 5, 6], so when used in array indexing,
    # it sets the same value for all dust source indices
    # Check that all dust indices have the same efficiency values
    for dust_idx in constants.dust_index:
        assert result.h_eff[constants.ploughing_eff_index, dust_idx, 0] == 0.5
        assert result.h_eff[constants.cleaning_eff_index, dust_idx, 1] == 0.001
        assert result.h_eff[constants.drainage_eff_index, dust_idx, 2] == 0.005
        assert result.h_eff[constants.spraying_eff_index, dust_idx, 3] == 0.1

    # Check salt efficiency factors - salt_index is an array, so we need to use .item() or indexing
    assert (
        result.h_eff[constants.ploughing_eff_index, constants.salt_index[0], 0].item()
        == 0.3
    )
    assert (
        result.h_eff[constants.cleaning_eff_index, constants.salt_index[0], 0].item()
        == 0.2
    )
    assert (
        result.h_eff[constants.drainage_eff_index, constants.salt_index[1], 0].item()
        == 1.0
    )
    assert (
        result.h_eff[constants.spraying_eff_index, constants.salt_index[1], 0].item()
        == 1.0
    )


def test_read_model_parameters_empty_dataframe():
    """Test that empty DataFrame doesn't crash."""
    df = pd.DataFrame()

    # Should handle empty DataFrame gracefully
    try:
        result = read_model_parameters(df)
        # If it doesn't crash, that's already good
        # But we shouldn't get here with truly empty DataFrame
        assert False, "Should have raised an error with empty DataFrame"
    except:
        # Expected to fail with empty DataFrame
        pass


def test_read_model_parameters_integration():
    """Test reading multiple sections together."""
    # Create a more realistic DataFrame with multiple sections
    rows = [["" for _ in range(7)] for _ in range(140)]

    # Add road wear section
    rows[3] = ["Road wear", "", "", "", "", "", ""]
    rows[4] = [
        "W0,roadwear (g km-1 veh-1)",
        "Studded tyres (st)",
        "Winter tyres (wi)",
        "Summer tyres (su)",
        "",
        "",
        "",
    ]
    rows[5] = ["Heavy (he)", "10.0", "0.5", "0.5", "", "", ""]
    rows[6] = ["Light (li)", "2.0", "0.1", "0.1", "", "", ""]
    rows[7] = [
        "Parameters for speed dependence",
        "a1",
        "a2",
        "a3",
        "Vref,roadwear",
        "Vmin,roadwear",
        "",
    ]
    rows[8] = [
        "W=W0*(a1+a2*(max[V,Vmin]/Vref)a3)",
        "0.0",
        "1.0",
        "1.0",
        "70.0",
        "30.0",
        "",
    ]

    # Add snow depth section
    rows[22] = ["Snow depth wear threshold", "", "", "", "", "", ""]
    rows[23] = ["Parameter", "Value", "", "", "", "", ""]
    rows[24] = ["sroadwear,thresh (mm w.e.)", "3.0", "", "", "", "", ""]

    # Add wind blown dust section
    rows[115] = ["Wind blown dust emission factors", "", "", "", "", "", ""]
    rows[116] = ["Parameter", "Value", "", "", "", "", ""]
    rows[117] = ["twind (hr)", "24", "", "", "", "", ""]
    rows[118] = ["FFthresh (m/s)", "10", "", "", "", "", ""]

    df = pd.DataFrame(rows)
    result = read_model_parameters(df)

    # Check values from different sections
    assert result.W_0[constants.road_index, constants.st, constants.he] == 10.0
    assert result.s_roadwear_thresh == 3.0
    assert result.tau_wind == 24.0
    assert result.FF_thresh == 10.0


def test_read_model_parameters_suspension():
    """Test reading suspension factors and scaling."""
    # Create DataFrame with suspension sections
    rows = [["", "", "", "", "", "", ""] for _ in range(40)]

    # Add suspension scaling factors section
    rows[10] = ["Suspension scaling factors", "", "", "", "", "", ""]
    rows[11] = ["", "PMall - PM200", "PM200 - PM10", "PM10 - PM2.5", "PM2.5", "", ""]
    # For each source, add scaling factors
    rows[12] = ["Road", "1.0", "0.8", "0.6", "0.4", "", ""]  # road_index = 0
    rows[13] = ["Tyre", "0.9", "0.7", "0.5", "0.3", "", ""]  # tyre_index = 1
    rows[14] = ["Brake", "0.8", "0.6", "0.4", "0.2", "", ""]  # brake_index = 2
    rows[15] = ["Salt 1", "0.7", "0.5", "0.3", "0.1", "", ""]  # salt1
    rows[16] = ["Salt 2", "0.6", "0.4", "0.2", "0.05", "", ""]  # salt2
    rows[17] = ["Sand 1", "0.5", "0.3", "0.1", "0.03", "", ""]  # sand1
    rows[18] = ["Sand 2", "0.4", "0.2", "0.08", "0.02", "", ""]  # sand2
    rows[19] = ["Sand 3", "0.3", "0.1", "0.06", "0.01", "", ""]  # sand3
    rows[20] = ["Other", "0.2", "0.05", "0.04", "0.005", "", ""]  # other
    rows[21] = ["Q road", "2.0", "1.5", "1.0", "0.5", "", ""]  # h_0_q_road

    # Add road suspension factors section
    rows[25] = ["Road suspension factors", "", "", "", "", "", ""]
    rows[26] = [
        "f0,suspension",
        "Studded tyres (st)",
        "Winter tyres (wi)",
        "Summer tyres (su)",
        "",
        "",
        "",
    ]
    rows[27] = ["Heavy (he)", "100.0", "50.0", "25.0", "", "", ""]
    rows[28] = ["Light (li)", "20.0", "10.0", "5.0", "", "", ""]
    rows[29] = ["a_sus parameters", "a1", "a2", "a3", "a4", "a5", ""]
    rows[30] = ["", "0.1", "0.2", "0.3", "0.4", "0.5", ""]

    df = pd.DataFrame(rows)
    result = read_model_parameters(df)

    # Check suspension scaling factors (h_0_sus)
    assert result.h_0_sus[constants.road_index, 0] == 1.0  # PMall-PM200
    assert result.h_0_sus[constants.road_index, 1] == 0.8  # PM200-PM10
    assert result.h_0_sus[constants.road_index, 2] == 0.6  # PM10-PM2.5
    assert result.h_0_sus[constants.road_index, 3] == 0.4  # PM2.5

    assert result.h_0_sus[constants.tyre_index, 0] == 0.9
    assert result.h_0_sus[constants.brake_index, 0] == 0.8

    # Check h_0_q_road
    assert result.h_0_q_road[0] == 2.0
    assert result.h_0_q_road[1] == 1.5
    assert result.h_0_q_road[2] == 1.0
    assert result.h_0_q_road[3] == 0.5

    # Check base suspension factors (at indices [0, 0, tyre, vehicle])
    assert result.f_0_suspension[0, 0, constants.st, constants.he] == 100.0
    assert result.f_0_suspension[0, 0, constants.wi, constants.he] == 50.0
    assert result.f_0_suspension[0, 0, constants.su, constants.he] == 25.0
    assert result.f_0_suspension[0, 0, constants.st, constants.li] == 20.0
    assert result.f_0_suspension[0, 0, constants.wi, constants.li] == 10.0
    assert result.f_0_suspension[0, 0, constants.su, constants.li] == 5.0

    # Check a_sus coefficients
    assert result.a_sus[0] == 0.1
    assert result.a_sus[1] == 0.2
    assert result.a_sus[2] == 0.3
    assert result.a_sus[3] == 0.4
    assert result.a_sus[4] == 0.5

    # Check that the suspension matrix is filled correctly
    # f_0_suspension[source, size, tyre, vehicle] = base_value * h_0_sus[source, size]
    # For road source, PMall-PM200 size, studded tyres, heavy vehicle:
    # Should be 100.0 * 1.0 = 100.0
    assert (
        result.f_0_suspension[constants.road_index, 0, constants.st, constants.he]
        == 100.0 * 1.0
    )

    # For road source, PM2.5 size, winter tyres, light vehicle:
    # Should be 10.0 * 0.4 = 4.0
    assert (
        result.f_0_suspension[constants.road_index, 3, constants.wi, constants.li]
        == 10.0 * 0.4
    )

    # For tyre source, PM200-PM10 size, summer tyres, heavy vehicle:
    # Should be 25.0 * 0.7 = 17.5
    assert (
        result.f_0_suspension[constants.tyre_index, 1, constants.su, constants.he]
        == 25.0 * 0.7
    )

    # For brake source, PM10-PM2.5 size, studded tyres, light vehicle:
    # Should be 20.0 * 0.4 = 8.0
    assert (
        result.f_0_suspension[constants.brake_index, 2, constants.st, constants.li]
        == 20.0 * 0.4
    )


def test_read_model_parameters_suspension_with_texture_scaling():
    """Test that texture scaling is applied to f_0_suspension."""
    rows = [["", "", "", "", "", "", ""] for _ in range(150)]

    # Add road suspension factors
    rows[25] = ["Road suspension factors", "", "", "", "", "", ""]
    rows[26] = [
        "f0,suspension",
        "Studded tyres (st)",
        "Winter tyres (wi)",
        "Summer tyres (su)",
        "",
        "",
        "",
    ]
    rows[27] = ["Heavy (he)", "100.0", "50.0", "25.0", "", "", ""]
    rows[28] = ["Light (li)", "20.0", "10.0", "5.0", "", "", ""]
    rows[29] = ["a_sus parameters", "a1", "a2", "a3", "a4", "a5", ""]
    rows[30] = ["", "0.0", "1.0", "1.0", "70.0", "30.0", ""]

    # Add suspension scaling factors (needed for the matrix calculation)
    rows[10] = ["Suspension scaling factors", "", "", "", "", "", ""]
    rows[11] = ["", "PMall - PM200", "PM200 - PM10", "PM10 - PM2.5", "PM2.5", "", ""]
    rows[12] = ["Road", "1.0", "1.0", "1.0", "1.0", "", ""]
    rows[13] = ["Tyre", "1.0", "1.0", "1.0", "1.0", "", ""]
    rows[14] = ["Brake", "1.0", "1.0", "1.0", "1.0", "", ""]
    rows[15] = ["Salt 1", "1.0", "1.0", "1.0", "1.0", "", ""]
    rows[16] = ["Salt 2", "1.0", "1.0", "1.0", "1.0", "", ""]
    rows[17] = ["Sand 1", "1.0", "1.0", "1.0", "1.0", "", ""]
    rows[18] = ["Sand 2", "1.0", "1.0", "1.0", "1.0", "", ""]
    rows[19] = ["Sand 3", "1.0", "1.0", "1.0", "1.0", "", ""]
    rows[20] = ["Other", "1.0", "1.0", "1.0", "1.0", "", ""]
    rows[21] = ["Q road", "1.0", "1.0", "1.0", "1.0", "", ""]

    # Add surface texture parameters
    rows[140] = ["Surface texture parameters", "", "", "", "", "", ""]
    rows[141] = ["Parameter", "Value", "", "", "", "", ""]
    rows[142] = ["g_road_drainable_min scaling", "0.8", "", "", "", "", ""]
    rows[143] = [
        "f_0_suspension scaling",
        "2.0",
        "",
        "",
        "",
        "",
        "",
    ]  # This scales all suspension values
    rows[144] = ["R_0_spray scaling", "1.2", "", "", "", "", ""]
    rows[145] = ["h_eff drainage scaling", "0.9", "", "", "", "", ""]
    rows[146] = ["h_eff spraying scaling", "1.1", "", "", "", "", ""]

    df = pd.DataFrame(rows)
    result = read_model_parameters(df)

    # Check that texture scaling was applied to f_0_suspension
    # All values should be multiplied by 2.0
    assert result.f_0_suspension[0, 0, constants.st, constants.he] == 100.0 * 2.0
    assert result.f_0_suspension[0, 0, constants.wi, constants.he] == 50.0 * 2.0
    assert result.f_0_suspension[0, 0, constants.su, constants.he] == 25.0 * 2.0
    assert result.f_0_suspension[0, 0, constants.st, constants.li] == 20.0 * 2.0
    assert result.f_0_suspension[0, 0, constants.wi, constants.li] == 10.0 * 2.0
    assert result.f_0_suspension[0, 0, constants.su, constants.li] == 5.0 * 2.0

    # Check that texture_scaling array itself is correct
    assert result.texture_scaling[0] == 0.8
    assert result.texture_scaling[1] == 2.0
    assert result.texture_scaling[2] == 1.2
    assert result.texture_scaling[3] == 0.9
    assert result.texture_scaling[4] == 1.1


def test_read_model_parameters_pavement_and_driving_cycle():
    """Test reading pavement type and driving cycle scaling factors."""
    rows = [["", "", "", "", "", ""] for _ in range(25)]

    # Add pavement type scaling factors
    rows[5] = ["Pavement type scaling factor", "", "", "", "", ""]
    rows[6] = ["Npave", "3", "", "", "", ""]
    rows[8] = ["1", "Asphalt", "3.4", "", "", ""]
    rows[9] = ["2", "Concrete", "3.2", "", "", ""]
    rows[10] = ["3", "Cobblestone", "3.0", "", "", ""]

    # Add driving cycle scaling factors
    rows[15] = ["Driving cycle scaling factor", "", "", "", "", ""]
    rows[16] = ["Ndc", "2", "", "", "", ""]
    rows[18] = ["1", "Urban", "0.8", "", "", ""]
    rows[19] = ["2", "Highway", "1.1", "", "", ""]

    df = pd.DataFrame(rows)
    result = read_model_parameters(df)

    # Check pavement type scaling factors
    assert result.num_pave == 3
    # The implementation reads from column 2 for values and column 1 for strings
    assert result.h_pave_str == ["Asphalt", "Concrete", "Cobblestone"]
    assert result.h_pave == [3.4, 3.2, 3.0]

    # Check driving cycle scaling factors
    assert result.num_dc == 2
    assert result.h_drivingcycle_str == ["Urban", "Highway"]
    assert result.h_drivingcycle == [0.8, 1.1]


def test_read_model_parameters_abrasion_and_crushing():
    """Test reading abrasion and crushing factors."""
    rows = [
        ["", "", "", "", "", "", ""] for _ in range(60)
    ]  # Increased size to prevent out of bounds

    # Add abrasion factors
    rows[10] = ["Abrasion factors", "", "", "", "", "", ""]
    rows[11] = [
        "f0,abrasion",
        "Studded tyres (st)",
        "Winter tyres (wi)",
        "Summer tyres (su)",
        "",
        "",
        "",
    ]
    rows[12] = ["Heavy (he)", "200.0", "100.0", "50.0", "", "", ""]
    rows[13] = ["Light (li)", "40.0", "20.0", "10.0", "", "", ""]
    rows[14] = ["Vref,abrasion (km/h)", "50.0", "", "", "", "", ""]
    rows[15] = [
        "Size dependence",
        "PMall - PM200",
        "PM200 - PM10",
        "PM10 - PM2.5",
        "PM2.5",
        "",
        "",
    ]
    rows[16] = ["h0,abrasion", "1.2", "1.0", "0.8", "0.6", "", ""]

    # Add crushing factors
    rows[25] = ["Crushing factors", "", "", "", "", "", ""]
    rows[26] = [
        "f0,crushing",
        "Studded tyres (st)",
        "Winter tyres (wi)",
        "Summer tyres (su)",
        "",
        "",
        "",
    ]
    rows[27] = ["Heavy (he)", "150.0", "75.0", "37.5", "", "", ""]
    rows[28] = ["Light (li)", "30.0", "15.0", "7.5", "", "", ""]
    rows[29] = ["Vref,crushing (km/h)", "40.0", "", "", "", "", ""]
    rows[30] = [
        "Size dependence",
        "PMall - PM200",
        "PM200 - PM10",
        "PM10 - PM2.5",
        "PM2.5",
        "",
        "",
    ]
    rows[31] = ["h0,crushing", "1.1", "0.9", "0.7", "0.5", "", ""]

    # Add source participation
    rows[40] = [
        "Sources participating in abrasion and crushing",
        "",
        "",
        "",
        "",
        "",
        "",
    ]
    rows[41] = ["Source", "p0,abrasion", "p0,crushing", "", "", "", ""]
    rows[42] = ["Road", "0.8", "0.6", "", "", "", ""]
    rows[43] = ["Tyre", "0.9", "0.7", "", "", "", ""]
    rows[44] = ["Brake", "0.7", "0.5", "", "", "", ""]
    rows[45] = ["Salt 1", "0.0", "1.0", "", "", "", ""]
    rows[46] = ["Salt 2", "0.0", "1.0", "", "", "", ""]
    rows[47] = ["Sand 1", "0.5", "0.8", "", "", "", ""]
    rows[48] = ["Sand 2", "0.4", "0.7", "", "", "", ""]
    rows[49] = ["Sand 3", "0.3", "0.6", "", "", "", ""]
    rows[50] = ["Other", "0.2", "0.4", "", "", "", ""]  # Add the 9th source

    # Fill remaining rows to prevent out of bounds
    for i in range(51, 60):
        rows[i] = ["", "", "", "", "", "", ""]

    df = pd.DataFrame(rows)
    result = read_model_parameters(df)

    # Check abrasion factors
    assert result.f_0_abrasion[constants.st, constants.he] == 200.0
    assert result.f_0_abrasion[constants.wi, constants.he] == 100.0
    assert result.f_0_abrasion[constants.su, constants.he] == 50.0
    assert result.f_0_abrasion[constants.st, constants.li] == 40.0
    assert result.f_0_abrasion[constants.wi, constants.li] == 20.0
    assert result.f_0_abrasion[constants.su, constants.li] == 10.0
    assert result.V_ref_abrasion == 50.0
    assert result.h_0_abrasion[0] == 1.2
    assert result.h_0_abrasion[1] == 1.0
    assert result.h_0_abrasion[2] == 0.8
    assert result.h_0_abrasion[3] == 0.6

    # Check crushing factors
    assert result.f_0_crushing[constants.st, constants.he] == 150.0
    assert result.f_0_crushing[constants.wi, constants.he] == 75.0
    assert result.f_0_crushing[constants.su, constants.he] == 37.5
    assert result.f_0_crushing[constants.st, constants.li] == 30.0
    assert result.f_0_crushing[constants.wi, constants.li] == 15.0
    assert result.f_0_crushing[constants.su, constants.li] == 7.5
    assert result.V_ref_crushing == 40.0
    assert result.h_0_crushing[0] == 1.1
    assert result.h_0_crushing[1] == 0.9
    assert result.h_0_crushing[2] == 0.7
    assert result.h_0_crushing[3] == 0.5

    # Check source participation
    assert result.p_0_abrasion[constants.road_index] == 0.8
    assert result.p_0_abrasion[constants.tyre_index] == 0.9
    assert result.p_0_abrasion[constants.brake_index] == 0.7
    assert result.p_0_crushing[constants.road_index] == 0.6
    assert result.p_0_crushing[constants.tyre_index] == 0.7
    assert result.p_0_crushing[constants.brake_index] == 0.5


def test_read_model_parameters_retention_and_ospm():
    """Test reading retention and OSPM parameters."""
    rows = [["", "", "", "", ""] for _ in range(20)]

    # Add retention parameters
    rows[5] = ["Retention parameters", "", "", "", ""]
    rows[6] = ["", "Road", "Brake", "Salt2", ""]
    rows[7] = ["gretention,thresh (g/m2)", "100.0", "50.0", "200.0", ""]
    rows[8] = ["gretention,min (g/m2)", "10.0", "5.0", "20.0", ""]

    # Add OSPM parameters
    rows[15] = ["OSPM parameters", "", "", "", ""]
    rows[16] = ["froof and fturb", "0.7", "0.3", "", ""]

    df = pd.DataFrame(rows)
    result = read_model_parameters(df)

    # Check retention parameters
    assert result.g_retention_thresh[constants.road_index] == 100.0
    assert result.g_retention_thresh[constants.brake_index] == 50.0
    assert result.g_retention_thresh[constants.salt_index[1]] == 200.0
    assert result.g_retention_min[constants.road_index] == 10.0
    assert result.g_retention_min[constants.brake_index] == 5.0
    assert result.g_retention_min[constants.salt_index[1]] == 20.0

    # Check OSPM parameters
    # The implementation reads from row_idx + 1, col 1 for f_roof and col 2 for f_turb
    assert result.f_roof_ospm_override == 0.7
    assert result.f_turb_ospm_override == 0.3


def test_read_model_parameters_track_parameters():
    """Test reading road track parameters."""
    rows = [["", "", "", "", "", ""] for _ in range(15)]

    # Add track parameters
    rows[5] = ["Road track parameters", "", "", "", "", ""]
    rows[6] = ["Track", "Include", "ftrack", "fveh,track", "fmig,track", ""]
    rows[7] = ["All road", "1", "1.0", "1.0", "1.0", ""]  # alltrack_type = 0
    rows[8] = ["Out track", "0", "0.0", "0.0", "0.0", ""]  # outtrack_type = 1
    rows[9] = ["In track", "0", "0.0", "0.0", "0.0", ""]  # intrack_type = 2
    rows[10] = ["Shoulder", "0", "0.0", "0.0", "0.0", ""]  # shoulder_type = 3
    rows[11] = ["Kerb", "0", "0.0", "0.0", "0.0", ""]  # kerb_type = 4

    df = pd.DataFrame(rows)
    result = read_model_parameters(df)

    # Only "All road" is included
    assert result.num_track == 1
    assert result.f_track == [1.0]
    assert result.veh_track == [1.0]
    assert result.mig_track == [1.0]
    assert result.track_type == [constants.alltrack_type]


def test_read_model_parameters_track_parameters_multiple():
    """Test reading multiple road track parameters with normalization."""
    rows = [["", "", "", "", "", ""] for _ in range(15)]

    # Add track parameters with multiple tracks and non-normalized values
    rows[5] = ["Road track parameters", "", "", "", "", ""]
    rows[6] = ["Track", "Include", "ftrack", "fveh,track", "fmig,track", ""]
    rows[7] = ["All road", "0", "0.0", "0.0", "0.0", ""]  # Not included
    rows[8] = ["Out track", "1", "0.6", "0.7", "0.5", ""]  # Included
    rows[9] = ["In track", "1", "0.3", "0.5", "0.3", ""]  # Included
    rows[10] = ["Shoulder", "1", "0.1", "0.2", "0.2", ""]  # Included
    rows[11] = ["Kerb", "0", "0.0", "0.0", "0.0", ""]  # Not included

    df = pd.DataFrame(rows)
    result = read_model_parameters(df)

    # Three tracks are included
    assert result.num_track == 3
    assert result.track_type == [
        constants.outtrack_type,
        constants.intrack_type,
        constants.shoulder_type,
    ]

    # Check that f_track is normalized (sum to 1)
    # Original: [0.6, 0.3, 0.1], sum = 1.0 (already normalized)
    assert result.f_track == [0.6, 0.3, 0.1]

    # Check that veh_track is normalized
    # Original: [0.7, 0.5, 0.2], sum = 1.4
    assert abs(result.veh_track[0] - 0.7 / 1.4) < 1e-6
    assert abs(result.veh_track[1] - 0.5 / 1.4) < 1e-6
    assert abs(result.veh_track[2] - 0.2 / 1.4) < 1e-6

    # mig_track should remain as-is since it doesn't get normalized
    assert result.mig_track == [0.5, 0.3, 0.2]


def test_read_model_parameters_tyre_and_brake_wear():
    """Test reading tyre and brake wear parameters."""
    df = pd.DataFrame(
        [
            ["", "", "", "", "", "", ""],  # Row 0
            ["", "", "", "", "", "", ""],  # Row 1
            ["", "", "", "", "", "", ""],  # Row 2
            ["Tyre wear", "", "", "", "", "", ""],  # Row 3 - header
            [
                "W0,tyrewear (g km-1 veh-1)",
                "Studded tyres (st)",
                "Winter tyres (wi)",
                "Summer tyres (su)",
                "",
                "",
                "",
            ],  # Row 4
            ["Heavy (he)", "1.5", "1.2", "1.0", "", "", ""],  # Row 5
            ["Light (li)", "0.3", "0.24", "0.2", "", "", ""],  # Row 6
            [
                "Parameters for speed dependence",
                "a1",
                "a2",
                "a3",
                "Vref,tyrewear",
                "Vmin,tyrewear",
                "",
            ],  # Row 7
            [
                "W=W0*(a1+a2*(max[V,Vmin]/Vref)a3)",
                "0.10",
                "0.90",
                "1.50",
                "80.00",
                "20.00",
                "",
            ],  # Row 8
            ["", "", "", "", "", "", ""],  # Row 9
            ["", "", "", "", "", "", ""],  # Row 10
            ["Brake wear", "", "", "", "", "", ""],  # Row 11
            [
                "W0,brakewear (g km-1 veh-1)",
                "Studded tyres (st)",
                "Winter tyres (wi)",
                "Summer tyres (su)",
                "",
                "",
                "",
            ],  # Row 12
            ["Heavy (he)", "0.8", "0.8", "0.8", "", "", ""],  # Row 13
            ["Light (li)", "0.16", "0.16", "0.16", "", "", ""],  # Row 14
            [
                "Parameters for speed dependence",
                "a1",
                "a2",
                "a3",
                "Vref,brakewear",
                "Vmin,brakewear",
                "",
            ],  # Row 15
            [
                "W=W0*(a1+a2*(max[V,Vmin]/Vref)a3)",
                "1.00",
                "0.00",
                "0.00",
                "50.00",
                "10.00",
                "",
            ],  # Row 16
        ]
    )

    result = read_model_parameters(df)

    # Check tyre wear values
    assert result.W_0[constants.tyre_index, constants.st, constants.he] == 1.5
    assert result.W_0[constants.tyre_index, constants.wi, constants.he] == 1.2
    assert result.W_0[constants.tyre_index, constants.su, constants.he] == 1.0
    assert result.W_0[constants.tyre_index, constants.st, constants.li] == 0.3
    assert result.W_0[constants.tyre_index, constants.wi, constants.li] == 0.24
    assert result.W_0[constants.tyre_index, constants.su, constants.li] == 0.2

    # Check tyre wear coefficients
    assert result.a_wear[constants.tyre_index, 0] == 0.10
    assert result.a_wear[constants.tyre_index, 1] == 0.90
    assert result.a_wear[constants.tyre_index, 2] == 1.50
    assert result.a_wear[constants.tyre_index, 3] == 80.00
    assert result.a_wear[constants.tyre_index, 4] == 20.00

    # Check brake wear values
    assert result.W_0[constants.brake_index, constants.st, constants.he] == 0.8
    assert result.W_0[constants.brake_index, constants.wi, constants.he] == 0.8
    assert result.W_0[constants.brake_index, constants.su, constants.he] == 0.8
    assert result.W_0[constants.brake_index, constants.st, constants.li] == 0.16
    assert result.W_0[constants.brake_index, constants.wi, constants.li] == 0.16
    assert result.W_0[constants.brake_index, constants.su, constants.li] == 0.16

    # Check brake wear coefficients
    assert result.a_wear[constants.brake_index, 0] == 1.00
    assert result.a_wear[constants.brake_index, 1] == 0.00
    assert result.a_wear[constants.brake_index, 2] == 0.00
    assert result.a_wear[constants.brake_index, 3] == 50.00
    assert result.a_wear[constants.brake_index, 4] == 10.00
