from dataclasses import dataclass, field
from typing import List


@dataclass
class input_metadata:
    """
    Class to hold metadata for the input data.
    """

    # Driving and pavement indices
    d_index: int = 1
    p_index: int = 1

    # Required fields (no defaults in MATLAB)
    b_road: float = 0.0
    LAT: float = 0.0
    LON: float = 0.0
    z_FF: float = 0.0
    z_T: float = 0.0
    DIFUTC_H: float = 0.0

    # Optional fields with defaults
    Z_SURF: float = 0.0
    z2_T: float = 25.0
    albedo_road: float = 0.3
    Pressure: float = 1000.0
    nodata: float = -99.0
    n_lanes: int = 2
    b_lane: float = 3.5
    b_canyon: float = 0.0  # Set to b_road if 0 or less than b_road
    h_canyon: List[float] = field(default_factory=lambda: [0.0, 0.0])
    ang_road: float = 0.0
    slope_road: float = 0.0
    wind_speed_correction: float = 1.0
    observed_moisture_cutoff_value: float = 1.5
    h_sus: float = 1.0
    h_texture: float = 1.0

    # OSPM parameters
    choose_receptor_ospm: int = 3
    SL1_ospm: float = 100.0
    SL2_ospm: float = 100.0
    f_roof_ospm: float = 0.82
    RecHeight_ospm: float = 2.5
    f_turb_ospm: float = 1.0

    # Emission factors (arrays for heavy and light vehicles)
    exhaust_EF: List[float] = field(default_factory=lambda: [0.0, 0.0])
    exhaust_EF_available: int = 0
    NOX_EF: List[float] = field(default_factory=lambda: [0.0, 0.0])
    NOX_EF_available: int = 0

    # Date strings
    start_date_str: str = ""
    end_date_str: str = ""
    start_date_save_str: str = ""
    end_date_save_str: str = ""
    start_subdate_save_str: List[str] = field(default_factory=list)
    end_subdate_save_str: List[str] = field(default_factory=list)
    n_save_subdate: int = 1

    # Calculated field
    b_road_lanes: float = 0.0  # n_lanes * b_lane
