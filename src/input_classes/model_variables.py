from dataclasses import dataclass, field
import numpy as np


@dataclass
class model_variables:
    """
    Dataclass containing all model variables and arrays for NORTRIP execution.
    """

    # Forecast missing data array
    road_temperature_forecast_missing: np.ndarray = field(
        default_factory=lambda: np.array([])
    )

    # Main model data arrays
    M_road_data: np.ndarray = field(default_factory=lambda: np.array([]))
    M_road_bin_data: np.ndarray = field(default_factory=lambda: np.array([]))
    M_road_bin_balance_data: np.ndarray = field(default_factory=lambda: np.array([]))
    M_road_balance_data: np.ndarray = field(default_factory=lambda: np.array([]))
    WR_time_data: np.ndarray = field(default_factory=lambda: np.array([]))
    road_salt_data: np.ndarray = field(default_factory=lambda: np.array([]))
    C_bin_data: np.ndarray = field(default_factory=lambda: np.array([]))
    C_data: np.ndarray = field(default_factory=lambda: np.array([]))
    E_road_data: np.ndarray = field(default_factory=lambda: np.array([]))
    E_road_bin_data: np.ndarray = field(default_factory=lambda: np.array([]))
    road_meteo_data: np.ndarray = field(default_factory=lambda: np.array([]))
    g_road_balance_data: np.ndarray = field(default_factory=lambda: np.array([]))
    g_road_data: np.ndarray = field(default_factory=lambda: np.array([]))

    # Forecast and correction arrays
    forecast_hours: np.ndarray = field(default_factory=lambda: np.array([]))
    E_corr_array: np.ndarray = field(default_factory=lambda: np.array([]))
    forecast_T_s: np.ndarray = field(default_factory=lambda: np.array([]))
    original_bias_T_s: float = 0.0

    # Quality factor arrays
    f_q: np.ndarray = field(default_factory=lambda: np.array([]))
    f_q_obs: np.ndarray = field(default_factory=lambda: np.array([]))

    # Dispersion factor array
    f_conc: np.ndarray = field(default_factory=lambda: np.array([]))

    # Initial mass data
    M_road_0_data: np.ndarray = field(default_factory=lambda: np.array([]))
