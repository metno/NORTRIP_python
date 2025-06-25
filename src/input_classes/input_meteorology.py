from dataclasses import dataclass, field
import numpy as np


@dataclass
class input_meteorology:
    """
    Dataclass containing parsed meteorological input data, matching the MATLAB structure and defaults.

    This class contains all meteorological variables that are read from the input file,
    along with their availability flags and missing data indices.
    """

    # Main meteorological data arrays (1D time series)
    T_a: np.ndarray = field(default_factory=lambda: np.array([]))
    T2_a: np.ndarray = field(default_factory=lambda: np.array([]))
    FF: np.ndarray = field(default_factory=lambda: np.array([]))
    DD: np.ndarray = field(default_factory=lambda: np.array([]))
    RH: np.ndarray = field(default_factory=lambda: np.array([]))
    T_dewpoint: np.ndarray = field(default_factory=lambda: np.array([]))
    Rain: np.ndarray = field(default_factory=lambda: np.array([]))
    Snow: np.ndarray = field(default_factory=lambda: np.array([]))
    short_rad_in: np.ndarray = field(default_factory=lambda: np.array([]))
    long_rad_in: np.ndarray = field(default_factory=lambda: np.array([]))
    cloud_cover: np.ndarray = field(default_factory=lambda: np.array([]))
    road_wetness_obs: np.ndarray = field(default_factory=lambda: np.array([]))
    road_temperature_obs: np.ndarray = field(default_factory=lambda: np.array([]))
    Pressure_a: np.ndarray = field(default_factory=lambda: np.array([]))
    T_sub: np.ndarray = field(default_factory=lambda: np.array([]))

    # Availability flags
    T2_a_available: int = 0
    DD_available: int = 0
    RH_available: int = 0
    T_dewpoint_available: int = 0
    short_rad_in_available: int = 0
    long_rad_in_available: int = 0
    cloud_cover_available: int = 0
    road_wetness_obs_available: int = 0
    road_temperature_obs_available: int = 0
    pressure_obs_available: int = 0
    T_sub_available: int = 0

    # Missing data indices
    T_a_nodata: list = field(default_factory=list)
    FF_nodata: list = field(default_factory=list)
    RH_nodata: list = field(default_factory=list)
    Rain_nodata: list = field(default_factory=list)
    Snow_nodata: list = field(default_factory=list)
    DD_nodata: list = field(default_factory=list)
    T2_a_nodata: list = field(default_factory=list)
    T_sub_nodata: list = field(default_factory=list)
    short_rad_in_missing: list = field(default_factory=list)
    long_rad_in_missing: list = field(default_factory=list)
    cloud_cover_missing: list = field(default_factory=list)
    road_wetness_obs_missing: list = field(default_factory=list)
    road_temperature_obs_missing: list = field(default_factory=list)
    pressure_obs_missing: list = field(default_factory=list)

    # Road wetness specific fields
    road_wetness_obs_in_mm: int = 0
    max_road_wetness_obs: float = np.nan
    min_road_wetness_obs: float = np.nan
    mean_road_wetness_obs: float = np.nan

    # Number of meteorological data points
    n_meteo: int = 0
