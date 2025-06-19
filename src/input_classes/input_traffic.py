from dataclasses import dataclass, field
import numpy as np
import constants


@dataclass
class input_traffic:
    """
    Dataclass containing parsed traffic input data.
    """

    # Date/time fields
    year: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    month: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    day: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    hour: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    minute: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    date_num: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    date_str: np.ndarray = field(
        default_factory=lambda: np.array([[], []], dtype=object)
    )

    # Traffic volumes
    N_total: np.ndarray = field(default_factory=lambda: np.array([]))
    N_v: np.ndarray = field(default_factory=lambda: np.zeros((constants.num_veh, 0)))
    N: np.ndarray = field(
        default_factory=lambda: np.zeros((constants.num_tyre, constants.num_veh, 0))
    )

    # Traffic speeds
    V_veh: np.ndarray = field(default_factory=lambda: np.zeros((constants.num_veh, 0)))

    # Indices of missing data
    N_total_nodata: list = field(default_factory=list)
    N_v_nodata: list = field(default_factory=list)
    N_nodata: list = field(default_factory=list)
    V_veh_nodata: list = field(default_factory=list)

    # Ratios for imputation
    N_ratio: np.ndarray = field(
        default_factory=lambda: np.zeros((constants.num_tyre, constants.num_veh, 0))
    )

    # Number of records
    n_traffic: int = 0
