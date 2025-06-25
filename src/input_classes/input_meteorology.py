from dataclasses import dataclass, field
import numpy as np


@dataclass
class input_meteorology:
    """
    Dataclass containing parsed meteorological input data, matching the MATLAB structure and defaults.

    This class contains all meteorological variables that are read from the input file,
    along with their availability flags and missing data indices.
    """

    # Core meteorological data (required)
    T_a: np.ndarray = field(
        default_factory=lambda: np.array([])
    )  # Air temperature (T2m)
    FF: np.ndarray = field(default_factory=lambda: np.array([]))  # Wind speed
    Rain: np.ndarray = field(default_factory=lambda: np.array([]))  # Rain precipitation
    Snow: np.ndarray = field(default_factory=lambda: np.array([]))  # Snow precipitation

    # Optional meteorological data
    T2_a: np.ndarray = field(
        default_factory=lambda: np.array([])
    )  # Alternative air temperature (T_a_alt)
    DD: np.ndarray = field(default_factory=lambda: np.array([]))  # Wind direction
    RH: np.ndarray = field(default_factory=lambda: np.array([]))  # Relative humidity
    T_dewpoint: np.ndarray = field(
        default_factory=lambda: np.array([])
    )  # Dewpoint temperature (T2m dewpoint)
    short_rad_in: np.ndarray = field(
        default_factory=lambda: np.array([])
    )  # Global radiation
    long_rad_in: np.ndarray = field(
        default_factory=lambda: np.array([])
    )  # Longwave radiation
    cloud_cover: np.ndarray = field(default_factory=lambda: np.array([]))  # Cloud cover
    road_wetness_obs: np.ndarray = field(
        default_factory=lambda: np.array([])
    )  # Road wetness observations
    road_temperature_obs: np.ndarray = field(
        default_factory=lambda: np.array([])
    )  # Road surface temperature
    Pressure_a: np.ndarray = field(
        default_factory=lambda: np.array([])
    )  # Atmospheric pressure
    T_sub: np.ndarray = field(
        default_factory=lambda: np.array([])
    )  # Subsurface temperature

    # Availability flags for optional data
    T2_a_available: int = 0
    DD_available: int = 0
    RH_available: int = 0
    T_dewpoint_available: int = 0
    short_rad_in_available: int = 0
    long_rad_in_available: int = 0
    cloud_cover_available: int = 0
    road_wetness_obs_available: int = 0
    road_temperature_obs_available: int = 0
    pressure_obs_available: int = 1  # Default to available, set to 0 if all nodata
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

    # Road wetness specific properties
    road_wetness_obs_in_mm: int = 0  # Flag if road wetness is in mm units
    max_road_wetness_obs: float = float("nan")  # Maximum observed road wetness
    min_road_wetness_obs: float = float("nan")  # Minimum observed road wetness
    mean_road_wetness_obs: float = float("nan")  # Mean observed road wetness

    # Number of meteorological records
    n_meteo: int = 0
