import pandas as pd
import numpy as np
from src.pd_util import read_txt


def test_read_txt_basic_functionality(tmp_path):
    # Create a temporary text file with precise content
    file_content = "-noe\ncol1\tcol2\nvalue1\tvalue2"
    file_path = tmp_path / "test.txt"
    file_path.write_text(file_content)

    df = read_txt(str(file_path))

    expected_data = {
        "0": ["-noe", "col1", "value1"],
        "1": [np.nan, "col2", "value2"],
    }
    expected_df = pd.DataFrame(expected_data)

    # Assert that the DataFrames are equal
    pd.testing.assert_frame_equal(df, expected_df)


def test_read_txt_single_column_line_with_spaces(tmp_path):
    file_content = "first item\nsecond\titem\n"
    file_path = tmp_path / "test.txt"
    file_path.write_text(file_content)

    df = read_txt(str(file_path))

    expected_data = {
        "0": ["first item", "second"],
        "1": [np.nan, "item"],
    }
    expected_df = pd.DataFrame(expected_data)

    pd.testing.assert_frame_equal(df, expected_df)


def test_read_txt_mixed_delimiters_and_padding(tmp_path):
    file_content = "A\tB\tC\nD E\nF\tG\n"
    file_path = tmp_path / "test.txt"
    file_path.write_text(file_content)

    df = read_txt(str(file_path))

    expected_data = {
        "0": ["A", "D E", "F"],
        "1": ["B", np.nan, "G"],
        "2": ["C", np.nan, np.nan],
    }
    expected_df = pd.DataFrame(expected_data)

    pd.testing.assert_frame_equal(df, expected_df)


def test_read_txt_empty_file(tmp_path):
    file_content = ""
    file_path = tmp_path / "empty.txt"
    file_path.write_text(file_content)

    df = read_txt(str(file_path))
    expected_df = pd.DataFrame(columns=pd.Index([]))

    # pd.DataFrame created with no columns when empty, compare to that
    assert (
        df.empty and expected_df.empty and len(df.columns) == len(expected_df.columns)
    )


def test_read_txt_only_newlines(tmp_path):
    file_content = "\n\n\n"
    file_path = tmp_path / "newlines.txt"
    file_path.write_text(file_content)

    df = read_txt(str(file_path))
    # Expected output for only newlines would be a DataFrame of empty strings with max_cols = 1
    expected_data = {"0": ["", "", ""]}
    expected_df = pd.DataFrame(expected_data)

    pd.testing.assert_frame_equal(df, expected_df)
