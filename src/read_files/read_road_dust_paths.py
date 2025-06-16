from config_classes.model_file_paths import model_file_paths
import pandas as pd
from pd_util.find_value import find_value


def read_road_dust_paths(read_as_text=0) -> model_file_paths:
    paths = model_file_paths()

    try:
        if read_as_text == 0:
            print("Reading info file and setting paths from excel")
            paths_df = pd.read_excel(
                "model_paths/model_paths_and_files.xlsx",
                sheet_name="Filenames",
                header=None,
            )
            header_text = paths_df.iloc[:, 0].astype(str)
            file_text = paths_df.iloc[:, 1].astype(str)
        else:
            print("Reading info file and setting paths from text")
            # Convert Excel file to text equivalent
            txt_filename = "model_paths/text/model_paths_and_files.txt"
            paths_df = pd.read_csv(txt_filename, sep="\t", header=None)
            header_text = paths_df.iloc[:, 0].astype(str)
            file_text = paths_df.iloc[:, 1].fillna("").astype(str)

    except FileNotFoundError as e:
        print(f"Error: {e}. The file was not found.")
        exit()

    # Extract paths and filenames
    paths.path_inputparam = find_value(
        "Model input parameter path", header_text, file_text
    )
    paths.path_inputdata = find_value("Model input data path", header_text, file_text)
    paths.path_outputdata = find_value("Model output data path", header_text, file_text)
    paths.path_outputfig = find_value(
        "Model output figures path", header_text, file_text
    )

    paths.filename_inputparam = find_value(
        "Model parameter filename", header_text, file_text
    )
    paths.filename_inputdata = find_value(
        "Model input data filename", header_text, file_text
    )
    paths.filename_outputdata = find_value(
        "Model output data filename", header_text, file_text
    )

    paths.path_ospm = find_value("Model ospm path", header_text, file_text)
    paths.path_fortran = find_value("Model fortran path", header_text, file_text)
    paths.path_fortran_output = find_value(
        "Model fortran output path", header_text, file_text
    )
    paths.filename_log = find_value("Log file name", header_text, file_text)
    paths.file_fortran_exe = find_value(
        "Model fortran executable filename", header_text, file_text
    )

    # Set automatic defaults if not found
    if not paths.path_fortran and paths.path_outputdata:
        output_idx = paths.path_outputdata.find("output")
        if output_idx > 0:
            paths.path_fortran = paths.path_outputdata[:output_idx] + "fortran/"

    if not paths.path_fortran_output and paths.path_outputdata:
        output_idx = paths.path_outputdata.find("output")
        if output_idx > 0:
            paths.path_fortran_output = (
                paths.path_outputdata[:output_idx] + "fortran/output/"
            )

    if not paths.filename_log and paths.path_fortran:
        paths.filename_log = paths.path_fortran + "NORTRIP_log.txt"

    if not paths.file_fortran_exe:
        paths.file_fortran_exe = "nortrip"

    # Set title string by removing extension from input data filename
    if paths.filename_inputdata:
        dot_idx = paths.filename_inputdata.find(".")
        if dot_idx > 0:
            paths.title_str = paths.filename_inputdata[:dot_idx]
            # Remove "input data" if present
            input_data_idx = paths.title_str.find("input data")
            if input_data_idx > 0:
                paths.title_str = paths.title_str[: input_data_idx - 1]
        else:
            paths.title_str = paths.filename_inputdata

    # Set combined path+filename properties
    paths.path_filename_inputparam = paths.path_inputparam + paths.filename_inputparam
    paths.path_filename_inputdata = paths.path_inputdata + paths.filename_inputdata

    return paths
