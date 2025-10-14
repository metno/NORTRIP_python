from config_classes.model_file_paths import model_file_paths
import pandas as pd
from pd_util import read_txt, find_str_or_default
import logging

logger = logging.getLogger(__name__)


def read_road_dust_paths(paths_path: str) -> model_file_paths:
    """
    Load model paths and filenames from the configured paths file.

    Args:
        paths_path (str): Path to the Excel file that defines model paths.

    Returns:
        model_file_paths: Dataclass with populated path settings.
    """
    paths = model_file_paths()

    try:
        if paths_path.endswith(".xlsx"):
            logger.info(f"Setting paths from Excel file: {paths_path}")
            paths_df = pd.read_excel(
                paths_path,
                sheet_name="Filenames",
                header=None,
            )
        elif paths_path.endswith(".txt"):
            logger.info(f"Setting paths from text file: {paths_path}")
            paths_df = read_txt(paths_path)
        else:
            logger.error(f"Invalid file type: {paths_path}")
            exit(1)

    except FileNotFoundError as e:
        logger.error(f"Path file not found: {e.filename}")
        exit(1)

    header_text = paths_df.iloc[:, 0]
    file_text = paths_df.iloc[:, 1]
    # Extract paths and filenames
    paths.path_inputparam = find_str_or_default(
        "Model input parameter path", header_text, file_text, ""
    )
    paths.path_inputdata = find_str_or_default(
        "Model input data path", header_text, file_text, ""
    )
    paths.path_outputdata = find_str_or_default(
        "Model output data path", header_text, file_text, ""
    )
    paths.path_outputfig = find_str_or_default(
        "Model output figures path", header_text, file_text, ""
    )

    paths.filename_inputparam = find_str_or_default(
        "Model parameter filename", header_text, file_text, ""
    )
    paths.filename_inputdata = find_str_or_default(
        "Model input data filename", header_text, file_text, ""
    )
    paths.filename_outputdata = find_str_or_default(
        "Model output data filename", header_text, file_text, ""
    )

    paths.path_ospm = find_str_or_default("Model ospm path", header_text, file_text, "")
    paths.path_fortran = find_str_or_default(
        "Model fortran path", header_text, file_text, ""
    )
    paths.path_fortran_output = find_str_or_default(
        "Model fortran output path", header_text, file_text, ""
    )
    paths.filename_log = find_str_or_default(
        "Log file name", header_text, file_text, ""
    )
    paths.file_fortran_exe = find_str_or_default(
        "Model fortran executable filename", header_text, file_text, ""
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

    logger.info("Successfully loaded model file paths")
    return paths
