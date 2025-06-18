from .read_input_activity import read_input_activity
from .read_input_airquality import read_input_airquality
from .read_input_meteorology import read_input_meteorology
from .read_input_traffic import read_input_traffic
from .read_input_initial import read_input_initial
from .read_input_metadata import read_input_metadata
from config_classes import model_parameters
import pandas as pd
import os
import logging

logger = logging.getLogger(__name__)


def read_road_dust_input(
    input_file_path: str,
    model_parameters: model_parameters,
    read_as_text=False,
    print_results=False,
):
    """
    Read road dust input data from specified file.

    Args:
        input_file_path (str): Path to the input file.
        read_as_text (bool, optional): If True, read the file as text. Will reformat input_file_path to text format.
        print_results (bool, optional): If True, print the results to the console
    Returns:

    """

    activity_df = airquality_df = meteorology_df = traffic_df = initial_df = (
        metadata_df
    ) = None

    if read_as_text:
        # Extract directory and base filename without extension
        base_dir = os.path.dirname(input_file_path)
        base_name = os.path.splitext(os.path.basename(input_file_path))[0]
        text_dir = os.path.join(base_dir, "text")

        activity_path = os.path.join(text_dir, f"{base_name}_activity.txt")
        airquality_path = os.path.join(text_dir, f"{base_name}_airquality.txt")
        meteorology_path = os.path.join(text_dir, f"{base_name}_meteorology.txt")
        traffic_path = os.path.join(text_dir, f"{base_name}_traffic.txt")
        initial_path = os.path.join(text_dir, f"{base_name}_initial.txt")
        metadata_path = os.path.join(text_dir, f"{base_name}_metadata.txt")

        encodings = ["utf-8", "latin-1", "cp1252", "iso-8859-1"]

        for encoding in encodings:
            try:
                activity_df = pd.read_csv(
                    activity_path, sep="\t", header=None, encoding=encoding
                )
                airquality_df = pd.read_csv(
                    airquality_path, sep="\t", header=None, encoding=encoding
                )
                meteorology_df = pd.read_csv(
                    meteorology_path, sep="\t", header=None, encoding=encoding
                )
                traffic_df = pd.read_csv(
                    traffic_path, sep="\t", header=None, encoding=encoding
                )
                initial_df = pd.read_csv(
                    initial_path, sep="\t", header=None, encoding=encoding
                )
                metadata_df = pd.read_csv(
                    metadata_path, sep="\t", header=None, encoding=encoding
                )
                logger.info(f"Read input data with encoding: {encoding}")
                break
            except FileNotFoundError as e:
                logger.error(f"File not found: {e.filename}")
                exit(1)
            except UnicodeDecodeError:
                if encoding == encodings[-1]:
                    logger.error(
                        f"Failed to read input data with all encodings: {encodings}"
                    )
                    exit()
                continue

    else:
        try:
            all_sheets = pd.read_excel(input_file_path, sheet_name=None)
        except FileNotFoundError:
            logger.error(f"File not found: {input_file_path}")
            exit(1)

        try:
            activity_df = all_sheets["Activity"]
            airquality_df = all_sheets["Airquality"]
            meteorology_df = all_sheets["Meteorology"]
            traffic_df = all_sheets["Traffic"]
            initial_df = all_sheets["Initialconditions"]
            metadata_df = all_sheets["Metadata"]
        except KeyError:
            logger.error(f"Sheet not found in file: {input_file_path}")
            exit(1)

    # Check if all required sheets are present (for the linter)
    if (
        metadata_df is None
        or initial_df is None
        or activity_df is None
        or airquality_df is None
        or meteorology_df is None
        or traffic_df is None
    ):
        logger.error("One or more required sheets are missing from the input file.")
        exit(1)

    metadata_data = read_input_metadata(metadata_df)
    initial_data = read_input_initial(
        initial_df, model_parameters, metadata_data, print_results
    )
    activity_data = read_input_activity(activity_df)
    airquality_data = read_input_airquality(airquality_df)
    meteorology_data = read_input_meteorology(meteorology_df)
    traffic_data = read_input_traffic(traffic_df)

    return (
        activity_data,
        airquality_data,
        meteorology_data,
        traffic_data,
        initial_data,
        metadata_data,
    )
