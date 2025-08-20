from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class shared_plot_data:
    # Time and configuration
    av: Tuple[int, ...]
    i_min: int
    i_max: int
    xlabel_text: str
    date_num: np.ndarray
    b_factor: float
    plot_size_fraction: int
    nodata: float

    # Precomputed arrays
    C_data_sum_tracks: np.ndarray
    E_road_data_sum_tracks: np.ndarray
    M_road_data_sum_tracks: np.ndarray
    M_road_balance_data_sum_tracks: np.ndarray
    WR_time_sum_tracks: np.ndarray

    road_meteo_weighted: np.ndarray
    g_road_weighted: np.ndarray
    g_road_balance_weighted: np.ndarray
    road_salt_weighted: np.ndarray
    f_q_weighted: np.ndarray
    f_q_obs_weighted: np.ndarray

    meteo_data_ro: np.ndarray
    traffic_data_ro: np.ndarray
    activity_data_ro: np.ndarray
    f_conc: np.ndarray

    # Observations
    PM_obs_net: np.ndarray
    PM_obs_bg: np.ndarray
    Salt_obs: np.ndarray
    Salt_obs_available: np.ndarray

    # Derived course-fraction (present only if requested)
    pm_course_derived: bool
    C_data_course: Optional[np.ndarray]
    E_road_data_course: Optional[np.ndarray]
    M_road_data_course: Optional[np.ndarray]
    M_road_balance_data_course: Optional[np.ndarray]
    PM_obs_net_course: Optional[np.ndarray]
    PM_obs_bg_course: Optional[np.ndarray]
