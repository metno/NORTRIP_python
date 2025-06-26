import pandas as pd
import logging
from input_classes import input_metadata
from pd_util import find_float_or_default, find_str_or_default

logger = logging.getLogger(__name__)


def read_input_metadata(
    metadata_df: pd.DataFrame, print_results: bool = False
) -> input_metadata:
    """
    Reads the metadata from the input DataFrame and returns an instance of input_metadata.

    Args:
        metadata_df (pd.DataFrame): DataFrame containing metadata information.
        print_results (bool): Whether to print the results to the console

    Returns:
        input_metadata: An instance of input_metadata containing the metadata.
    """
    loaded_metadata = input_metadata()

    # Extract header and value columns
    header_col = metadata_df.iloc[:, 0]
    data_col = metadata_df.iloc[:, 1]

    # Mapping: (header, field)
    mapping = [
        ("Driving cycle", "d_index"),
        ("Pavement type", "p_index"),
        ("Road width", "b_road"),
        ("Latitude", "LAT"),
        ("Longitude", "LON"),
        ("Height obs wind", "z_FF"),
        ("Height obs temperature", "z_T"),
        ("Time difference", "DIFUTC_H"),
        ("Elevation", "Z_SURF"),
        ("Height obs other temperature", "z2_T"),
        ("Surface albedo", "albedo_road"),
        ("Surface pressure", "Pressure"),
        ("Missing data", "nodata"),
        ("Number of lanes", "n_lanes"),
        ("Width of lane", "b_lane"),
        ("Street orientation", "ang_road"),
        ("Street slope", "slope_road"),
        ("Wind speed correction", "wind_speed_correction"),
        ("Observed moisture cut off", "observed_moisture_cutoff_value"),
        ("Suspension rate scaling factor", "h_sus"),
        ("Surface texture scaling", "h_texture"),
        ("Choose receptor position for ospm", "choose_receptor_ospm"),
        ("Street canyon length north for ospm", "SL1_ospm"),
        ("Street canyon length south for ospm", "SL2_ospm"),
        ("f_roof factor for ospm", "f_roof_ospm"),
        ("Receptor height for ospm", "RecHeight_ospm"),
        ("f_turb factor for ospm", "f_turb_ospm"),
    ]

    loaded_count = 0

    # Set all simple fields using dataclass default as fallback
    for header, field in mapping:
        val = find_float_or_default(header, header_col, data_col, float("nan"))
        if pd.isna(val):
            continue
        loaded_count += 1
        setattr(loaded_metadata, field, val)

    # b_canyon: default to b_road if missing or less than b_road
    b_canyon = find_float_or_default(
        "Street canyon width", header_col, data_col, loaded_metadata.b_road
    )
    if b_canyon < loaded_metadata.b_road:
        b_canyon = loaded_metadata.b_road
    if b_canyon == loaded_metadata.b_road:
        if print_results:
            logger.warning(
                f"Parameter 'Street canyon width' not found or less than 'Road width'. Using 'Road width' value: {loaded_metadata.b_road}"
            )
    if b_canyon != loaded_metadata.b_canyon:
        loaded_count += 1
    loaded_metadata.b_canyon = b_canyon

    # h_canyon: handle north/south logic
    h_canyon_north = find_float_or_default(
        "Street canyon height north", header_col, data_col, 0.0
    )
    h_canyon_south = find_float_or_default(
        "Street canyon height south", header_col, data_col, 0.0
    )
    # If both are zero, try 'Street canyon height' (single value)
    if h_canyon_north == 0.0 and h_canyon_south == 0.0:
        h_canyon_single = find_float_or_default(
            "Street canyon height", header_col, data_col, 0.0
        )
        loaded_metadata.h_canyon = [h_canyon_single, h_canyon_single]
        if h_canyon_single == 0.0:
            if print_results:
                logger.warning(
                    "Parameter 'Street canyon height' not found in metadata. Using default value: [0.0, 0.0]"
                )
        else:
            loaded_count += 1
    else:
        loaded_metadata.h_canyon = [h_canyon_north, h_canyon_south]
        if h_canyon_north == 0.0 and h_canyon_south == 0.0:
            if print_results:
                logger.warning(
                    "Parameter 'Street canyon height north' and 'Street canyon height south' not found in metadata. Using default value: [0.0, 0.0]"
                )
        else:
            loaded_count += 1

    # exhaust_EF and NOX_EF arrays
    exhaust_EF_0 = find_float_or_default(
        "Exhaust EF (he)", header_col, data_col, loaded_metadata.exhaust_EF[0]
    )
    exhaust_EF_1 = find_float_or_default(
        "Exhaust EF (li)", header_col, data_col, loaded_metadata.exhaust_EF[1]
    )
    if exhaust_EF_0 == loaded_metadata.exhaust_EF[0]:
        if print_results:
            logger.warning(
                "Parameter 'Exhaust EF (he)' not found in metadata. Using default value: 0.0"
            )
    if exhaust_EF_1 == loaded_metadata.exhaust_EF[1]:
        if print_results:
            logger.warning(
                "Parameter 'Exhaust EF (li)' not found in metadata. Using default value: 0.0"
            )
    if (
        exhaust_EF_0 != loaded_metadata.exhaust_EF[0]
        or exhaust_EF_1 != loaded_metadata.exhaust_EF[1]
    ):
        loaded_count += 1
    loaded_metadata.exhaust_EF = [exhaust_EF_0, exhaust_EF_1]
    loaded_metadata.exhaust_EF_available = int(sum(loaded_metadata.exhaust_EF) != 0)

    NOX_EF_0 = find_float_or_default(
        "NOX EF (he)", header_col, data_col, loaded_metadata.NOX_EF[0]
    )
    NOX_EF_1 = find_float_or_default(
        "NOX EF (li)", header_col, data_col, loaded_metadata.NOX_EF[1]
    )
    if NOX_EF_0 == loaded_metadata.NOX_EF[0]:
        if print_results:
            logger.warning(
                "Parameter 'NOX EF (he)' not found in metadata. Using default value: 0.0"
            )
    if NOX_EF_1 == loaded_metadata.NOX_EF[1]:
        if print_results:
            logger.warning(
                "Parameter 'NOX EF (li)' not found in metadata. Using default value: 0.0"
            )
    if NOX_EF_0 != loaded_metadata.NOX_EF[0] or NOX_EF_1 != loaded_metadata.NOX_EF[1]:
        loaded_count += 1
    loaded_metadata.NOX_EF = [NOX_EF_0, NOX_EF_1]
    loaded_metadata.NOX_EF_available = int(sum(loaded_metadata.NOX_EF) != 0)

    # Dates (strings)
    start_date_str = find_str_or_default(
        "Start date", header_col, data_col, loaded_metadata.start_date_str
    )
    end_date_str = find_str_or_default(
        "End date", header_col, data_col, loaded_metadata.end_date_str
    )
    if start_date_str and len(start_date_str) < 11:
        start_date_str += " 00:00:00"
    if end_date_str and len(end_date_str) < 11:
        end_date_str += " 00:00:00"
    if start_date_str == loaded_metadata.start_date_str:
        if print_results:
            logger.warning(
                f"Parameter 'Start date' not found in metadata. Using default value: {loaded_metadata.start_date_str}"
            )
    if end_date_str == loaded_metadata.end_date_str:
        if print_results:
            logger.warning(
                f"Parameter 'End date' not found in metadata. Using default value: {loaded_metadata.end_date_str}"
            )
    if start_date_str != loaded_metadata.start_date_str:
        loaded_count += 1
    if end_date_str != loaded_metadata.end_date_str:
        loaded_count += 1
    loaded_metadata.start_date_str = start_date_str
    loaded_metadata.end_date_str = end_date_str

    # Save dates (multiple possible)
    start_date_save_str = find_str_or_default(
        "Start save date", header_col, data_col, loaded_metadata.start_date_save_str
    )
    end_date_save_str = find_str_or_default(
        "End save date", header_col, data_col, loaded_metadata.end_date_save_str
    )
    if start_date_save_str and len(start_date_save_str) < 11:
        start_date_save_str += " 00:00:00"
    if end_date_save_str and len(end_date_save_str) < 11:
        end_date_save_str += " 00:00:00"
    if start_date_save_str == loaded_metadata.start_date_save_str:
        if print_results:
            logger.warning(
                f"Parameter 'Start save date' not found in metadata. Using default value: {loaded_metadata.start_date_save_str}"
            )
    if end_date_save_str == loaded_metadata.end_date_save_str:
        if print_results:
            logger.warning(
                f"Parameter 'End save date' not found in metadata. Using default value: {loaded_metadata.end_date_save_str}"
            )
    if start_date_save_str != loaded_metadata.start_date_save_str:
        loaded_count += 1
    if end_date_save_str != loaded_metadata.end_date_save_str:
        loaded_count += 1
    loaded_metadata.start_date_save_str = start_date_save_str
    loaded_metadata.end_date_save_str = end_date_save_str

    # For now, only support one subdate (can be extended for multiple)
    loaded_metadata.start_subdate_save_str = (
        [loaded_metadata.start_date_save_str]
        if loaded_metadata.start_date_save_str
        else []
    )
    loaded_metadata.end_subdate_save_str = (
        [loaded_metadata.end_date_save_str] if loaded_metadata.end_date_save_str else []
    )
    loaded_metadata.n_save_subdate = max(1, len(loaded_metadata.start_subdate_save_str))

    # Calculated field
    loaded_metadata.b_road_lanes = loaded_metadata.n_lanes * loaded_metadata.b_lane

    logger.info(f"Successfully loaded {loaded_count} metadata values")
    return loaded_metadata
