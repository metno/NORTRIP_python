from dataclasses import dataclass, field
import numpy as np
import constants


@dataclass
class input_activity:
    """
    Dataclass containing parsed activity input data.
    """

    # Date/time fields
    year: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    month: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    day: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    hour: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    minute: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))

    # Activity data arrays
    M_sanding: np.ndarray = field(default_factory=lambda: np.array([]))
    M_salting: np.ndarray = field(
        default_factory=lambda: np.zeros((constants.num_salt, 0))
    )  # [2, n_act] for na and secondary salt
    t_ploughing: np.ndarray = field(default_factory=lambda: np.array([]))
    t_cleaning: np.ndarray = field(default_factory=lambda: np.array([]))
    g_road_wetting: np.ndarray = field(default_factory=lambda: np.array([]))
    M_fugitive: np.ndarray = field(default_factory=lambda: np.array([]))

    # Input data arrays (original input before redistribution)
    M_sanding_input: np.ndarray = field(default_factory=lambda: np.array([]))
    M_salting_input: np.ndarray = field(
        default_factory=lambda: np.zeros((constants.num_salt, 0))
    )
    t_ploughing_input: np.ndarray = field(default_factory=lambda: np.array([]))
    t_cleaning_input: np.ndarray = field(default_factory=lambda: np.array([]))
    g_road_wetting_input: np.ndarray = field(default_factory=lambda: np.array([]))
    M_fugitive_input: np.ndarray = field(default_factory=lambda: np.array([]))

    # Salt type information
    salt_type: np.ndarray = field(
        default_factory=lambda: np.array([constants.na, constants.mg], dtype=np.int32)
    )  # Default: [na, mg]
    second_salt_type: int = constants.mg  # Default secondary salt type
    second_salt_available: int = (
        0  # Flag indicating if secondary salt data is available
    )
    salt2_str: str = "mg"  # String representation of secondary salt type

    # Availability flags
    g_road_wetting_available: int = 0

    # Number of activity records
    n_act: int = 0
