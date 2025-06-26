from dataclasses import dataclass, field
import numpy as np
import constants


@dataclass
class input_airquality:
    """
    Dataclass containing parsed air quality input data, matching the MATLAB structure and defaults.

    This class contains all air quality variables that are read from the input file,
    along with their availability flags and calculated net concentrations.
    """

    # Main air quality observation data arrays [size_fraction, n_date] for PM data
    PM_obs: np.ndarray = field(
        default_factory=lambda: np.full((constants.num_size, 0), -99.0)
    )
    PM_background: np.ndarray = field(
        default_factory=lambda: np.full((constants.num_size, 0), -99.0)
    )

    # NOX data arrays [n_date] - 1D arrays
    NOX_obs: np.ndarray = field(default_factory=lambda: np.full(0, -99.0))
    NOX_background: np.ndarray = field(default_factory=lambda: np.full(0, -99.0))
    NOX_emis: np.ndarray = field(default_factory=lambda: np.full(0, -99.0))
    EP_emis: np.ndarray = field(default_factory=lambda: np.full(0, -99.0))

    # Salt observations [salt_type, n_date] - currently only na (sodium) is read
    Salt_obs: np.ndarray = field(
        default_factory=lambda: np.full((constants.num_salt, 0), -99.0)
    )

    # Dispersion factor
    f_dis_input: np.ndarray = field(default_factory=lambda: np.full(0, -99.0))

    # Availability flags
    NOX_emis_available: int = 0
    EP_emis_available: int = 0
    Salt_obs_available: np.ndarray = field(
        default_factory=lambda: np.zeros(constants.num_salt, dtype=int)
    )
    f_dis_available: int = 0

    # Calculated net concentrations (PM_obs - PM_background)
    PM_obs_net: np.ndarray = field(
        default_factory=lambda: np.full((constants.num_size, 0), np.nan)
    )
    PM_obs_bg: np.ndarray = field(
        default_factory=lambda: np.full((constants.num_size, 0), np.nan)
    )
    NOX_obs_net: np.ndarray = field(default_factory=lambda: np.full(0, np.nan))

    # OSPM (Operational Street Pollution Model) data - exists only if OSMP sheet is present
    OSPM_data_exists: int = 0

    # OSPM meteorological data
    U_mast_ospm_orig: np.ndarray = field(default_factory=lambda: np.array([]))
    wind_dir_ospm_orig: np.ndarray = field(default_factory=lambda: np.array([]))
    TK_ospm_orig: np.ndarray = field(default_factory=lambda: np.array([]))
    GlobalRad_ospm_orig: np.ndarray = field(default_factory=lambda: np.array([]))

    # OSPM concentration and emission data
    cNOx_b_ospm_orig: np.ndarray = field(default_factory=lambda: np.array([]))
    qNOX_ospm_orig: np.ndarray = field(default_factory=lambda: np.array([]))

    # OSPM traffic data
    NNp_ospm_orig: np.ndarray = field(
        default_factory=lambda: np.array([])
    )  # N(li) ospm
    NNt_ospm_orig: np.ndarray = field(
        default_factory=lambda: np.array([])
    )  # N(he) ospm
    Vp_ospm_orig: np.ndarray = field(
        default_factory=lambda: np.array([])
    )  # V_veh(li) ospm
    Vt_ospm_orig: np.ndarray = field(
        default_factory=lambda: np.array([])
    )  # V_veh(he) ospm

    # Number of air quality data points
    n_date: int = 0
