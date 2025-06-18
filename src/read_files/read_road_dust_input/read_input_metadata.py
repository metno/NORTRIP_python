import pandas as pd
import numpy as np
from input_classes import input_metadata
from pd_util.find_value_or_default import find_value_or_default
from pd_util.find_value import find_value


def read_input_metadata(metadata_df: pd.DataFrame) -> input_metadata:
    """
    Reads the metadata from the input DataFrame and returns an instance of input_metadata.

    Args:
        metadata_df (pd.DataFrame): DataFrame containing metadata information.

    Returns:
        input_metadata: An instance of input_metadata containing the metadata.
    """
    loaded_metadata = input_metadata()

    # Create mapping of header text to field names and their required status
    field_mapping = {
        "Driving cycle": ("d_index", True, 1),
        "Pavement type": ("p_index", True, 1),
        "Road width": ("b_road", False, None),  # Required field
        "Latitude": ("LAT", False, None),  # Required field
        "Longitude": ("LON", False, None),  # Required field
        "Elevation": ("Z_SURF", True, 0.0),
        "Height obs wind": ("z_FF", False, None),  # Required field
        "Height obs temperature": ("z_T", False, None),  # Required field
        "Height obs other temperature": ("z2_T", True, 25.0),
        "Surface albedo": ("albedo_road", True, 0.3),
        "Time difference": ("DIFUTC_H", False, None),  # Required field
        "Surface pressure": ("Pressure", True, 1000.0),
        "Missing data": ("nodata", True, -99.0),
        "Number of lanes": ("n_lanes", True, 2),
        "Width of lane": ("b_lane", True, 3.5),
        "Street canyon width": ("b_canyon", True, None),  # Special handling
        "Street orientation": ("ang_road", True, 0.0),
        "Street slope": ("slope_road", True, 0.0),
        "Wind speed correction": ("wind_speed_correction", True, 1.0),
        "Observed moisture cut off": ("observed_moisture_cutoff_value", True, 1.5),
        "Suspension rate scaling factor": ("h_sus", True, 1.0),
        "Surface texture scaling": ("h_texture", True, 1.0),
        "Choose receptor position for ospm": ("choose_receptor_ospm", True, 3),
        "Street canyon length north for ospm": ("SL1_ospm", True, 100.0),
        "Street canyon length south for ospm": ("SL2_ospm", True, 100.0),
        "f_roof factor for ospm": ("f_roof_ospm", True, 0.82),
        "Receptor height for ospm": ("RecHeight_ospm", True, 2.5),
        "f_turb factor for ospm": ("f_turb_ospm", True, 1.0),
    }

    header_series = pd.Series(metadata_df.index)
    data_series = metadata_df.iloc[:, 0]

    # Process standard fields using mapping
    for header_text, (field_name, has_default, default_value) in field_mapping.items():
        if has_default:
            value = find_value_or_default(header_text, header_series, data_series, float(default_value) if default_value is not None else 0.0)
        else:
            # Required field: use find_value, raise error if missing
            value = find_value(header_text, header_series, data_series)
            if value == "" or pd.isna(value):
                raise ValueError(f"Required field '{header_text}' not found or is NaN")
            # Try to convert to float if possible
            try:
                value = float(value)
            except Exception:
                pass
        setattr(loaded_metadata, field_name, value)

    # Special handling for b_canyon
    if loaded_metadata.b_canyon is None:
        loaded_metadata.b_canyon = loaded_metadata.b_road
    if loaded_metadata.b_canyon < loaded_metadata.b_road:
        loaded_metadata.b_canyon = loaded_metadata.b_road

    # Handle street canyon height (can be single value or north/south)
    h_canyon = [0.0, 0.0]
    canyon_height = find_value_or_default("Street canyon height", header_series, data_series, 0.0)
    if isinstance(canyon_height, (list, np.ndarray)) and len(canyon_height) > 1:
        h_canyon[0] = float(canyon_height[0])
        h_canyon[1] = float(canyon_height[1]) if len(canyon_height) > 1 else float(canyon_height[0])
    else:
        h_canyon[0] = float(canyon_height)
        h_canyon[1] = float(canyon_height)

    # Override with specific north/south values if available
    h_canyon[0] = find_value_or_default("Street canyon height north", header_series, data_series, h_canyon[0])
    h_canyon[1] = find_value_or_default("Street canyon height south", header_series, data_series, h_canyon[1])
    loaded_metadata.h_canyon = h_canyon

    # Handle emission factors
    exhaust_ef = [0.0, 0.0]
    exhaust_ef[0] = find_value_or_default("Exhaust EF (he)", header_series, data_series, 0.0)  # Heavy vehicles
    exhaust_ef[1] = find_value_or_default("Exhaust EF (li)", header_series, data_series, 0.0)  # Light vehicles
    loaded_metadata.exhaust_EF = exhaust_ef
    loaded_metadata.exhaust_EF_available = 1 if sum(exhaust_ef) > 0 else 0

    nox_ef = [0.0, 0.0]
    nox_ef[0] = find_value_or_default("NOX EF (he)", header_series, data_series, 0.0)  # Heavy vehicles
    nox_ef[1] = find_value_or_default("NOX EF (li)", header_series, data_series, 0.0)  # Light vehicles
    loaded_metadata.NOX_EF = nox_ef
    loaded_metadata.NOX_EF_available = 1 if sum(nox_ef) > 0 else 0

    # Handle date strings (assuming they're in text columns)
    def format_date_string(date_str):
        """Add time if missing from date string."""
        if date_str and len(str(date_str)) < 11:
            return f"{date_str} 00:00:00"
        return str(date_str) if date_str else ""

    # Simple date handling (not implementing multiple save dates for now)
    start_date = find_value("Start date", header_series, data_series)
    end_date = find_value("End date", header_series, data_series)
    start_save_date = find_value("Start save date", header_series, data_series)
    end_save_date = find_value("End save date", header_series, data_series)

    loaded_metadata.start_date_str = format_date_string(start_date)
    loaded_metadata.end_date_str = format_date_string(end_date)
    loaded_metadata.start_date_save_str = format_date_string(start_save_date)
    loaded_metadata.end_date_save_str = format_date_string(end_save_date)

    # Calculate derived field
    loaded_metadata.b_road_lanes = loaded_metadata.n_lanes * loaded_metadata.b_lane

    return loaded_metadata
