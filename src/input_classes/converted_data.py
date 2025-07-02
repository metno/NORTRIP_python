from dataclasses import dataclass, field
import numpy as np
import constants


@dataclass
class converted_data:
    """
    Dataclass containing converted data in the new variable structure.

    This dataclass holds the consolidated data arrays that are created by converting
    the individual input data arrays into a unified structure for the NORTRIP model.
    All arrays are organized with consistent indexing and dimensions.
    """

    # Main data arrays with dimensions [index, n_date, n_roads]
    date_data: np.ndarray = field(
        default_factory=lambda: np.full(
            (constants.num_date_index, 0, constants.n_roads), -99.0
        )
    )
    traffic_data: np.ndarray = field(
        default_factory=lambda: np.full(
            (constants.num_traffic_index, 0, constants.n_roads), -99.0
        )
    )
    meteo_data: np.ndarray = field(
        default_factory=lambda: np.full(
            (constants.num_meteo_index, 0, constants.n_roads), -99.0
        )
    )
    activity_data: np.ndarray = field(
        default_factory=lambda: np.full(
            (constants.num_activity_index, 0, constants.n_roads), -99.0
        )
    )
    activity_data_input: np.ndarray = field(
        default_factory=lambda: np.full(
            (constants.num_activity_index, 0, constants.n_roads), -99.0
        )
    )

    # Additional data arrays with dimensions [n_date, n_roads]
    f_conc: np.ndarray = field(
        default_factory=lambda: np.full((0, constants.n_roads), -99.0)
    )
    f_dis: np.ndarray = field(
        default_factory=lambda: np.full((0, constants.n_roads), -99.0)
    )

    # Data availability and metadata
    n_date: int = 0
    n_roads: int = constants.n_roads
    nodata: float = -99.0
