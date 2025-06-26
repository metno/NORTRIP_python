from .read_input_activity import read_input_activity
from .read_input_airquality import read_input_airquality
from .read_input_meteorology import read_input_meteorology
from .read_input_traffic import read_input_traffic
from .read_input_initial import read_input_initial
from .read_input_metadata import read_input_metadata
from .traffic_utils import calculate_good_traffic_data_indices
from config_classes import model_parameters
import pandas as pd
import os
import logging
from pd_util import read_txt

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
        tuple: A tuple containing the following dataclasses:
            - activity_data: input_activity dataclass
            - airquality_data: input_airquality dataclass
            - meteorology_data: input_meteorology dataclass
            - traffic_data: input_traffic dataclass
            - initial_data: input_initial dataclass
            - metadata_data: input_metadata dataclass

    """

    activity_df = airquality_df = meteorology_df = traffic_df = initial_df = (
        metadata_df
    ) = ospm_df = None

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
        ospm_path = os.path.join(text_dir, f"{base_name}_ospm.txt")

        try:
            activity_df = read_txt(activity_path)
            airquality_df = read_txt(airquality_path)
            meteorology_df = read_txt(meteorology_path)
            traffic_df = read_txt(traffic_path)
            initial_df = read_txt(initial_path)
            metadata_df = read_txt(metadata_path)

            # OSPM is optional
            try:
                ospm_df = read_txt(ospm_path)
            except FileNotFoundError:
                ospm_df = None

        except FileNotFoundError as e:
            logger.error(f"File not found: {e.filename}")
            exit(1)

    else:
        try:
            all_sheets = pd.read_excel(input_file_path, sheet_name=None, header=None)
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

            # OSPM is optional
            ospm_df = all_sheets.get("OSPM", None)

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

    metadata_data = read_input_metadata(metadata_df, print_results)
    initial_data = read_input_initial(
        initial_df, model_parameters, metadata_data, print_results
    )
    traffic_data = read_input_traffic(traffic_df, metadata_data.nodata, print_results)
    meteorology_data = read_input_meteorology(
        meteorology_df,
        nodata=metadata_data.nodata,
        wind_speed_correction=metadata_data.wind_speed_correction,
        pressure_default=metadata_data.Pressure,
        print_results=print_results,
    )
    activity_data = read_input_activity(
        activity_df,
        traffic_data.year,
        traffic_data.month,
        traffic_data.day,
        traffic_data.hour,
        traffic_data.minute,
        print_results,
    )

    # Calculate N_good_data from traffic data for emission data filling
    N_good_data = None
    if len(traffic_data.N_total_nodata) > 0:
        N_good_data = calculate_good_traffic_data_indices(
            traffic_data, metadata_data.nodata
        )

    airquality_data = read_input_airquality(
        airquality_df,
        ospm_df=ospm_df,
        nodata=metadata_data.nodata,
        traffic_date_num=traffic_data.date_num,
        traffic_hour=traffic_data.hour,
        N_total_nodata=traffic_data.N_total_nodata,
        N_good_data=N_good_data,
        print_results=print_results,
    )

    return (
        activity_data,
        airquality_data,
        meteorology_data,
        traffic_data,
        initial_data,
        metadata_data,
    )
