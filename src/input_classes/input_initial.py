from dataclasses import dataclass, field
import numpy as np
import constants


@dataclass
class input_initial:
    """
    Dataclass containing initial input data after parsing.
    m_road_init: np.ndarray, shape (num_source, num_track)
    g_road_init: np.ndarray, shape (num_moisture, num_track)
    """

    m_road_init: np.ndarray = field(
        default_factory=lambda: np.zeros(
            (constants.num_source, constants.num_track_max)
        )
    )
    g_road_init: np.ndarray = field(
        default_factory=lambda: np.zeros(
            (constants.num_moisture, constants.num_track_max)
        )
    )
