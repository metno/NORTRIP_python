from dataclasses import dataclass, field
from typing import List
import numpy as np
import constants


@dataclass
class model_parameters:
    """
    Dataclass containing model parameters
    """

    ploughing_thresh: float = 0.1

    # Road wear parameters - W_0[source, tyre_type, vehicle_type]
    W_0: np.ndarray = field(
        default_factory=lambda: np.zeros(
            (constants.num_wear, constants.num_tyre, constants.num_veh)
        )
    )

    # Wear coefficients - a_wear[source, coefficient_index] (a1,a2,a3,V_ref,V_min)
    a_wear: np.ndarray = field(
        default_factory=lambda: np.zeros((constants.num_wear, 5))
    )

    # Snow depth wear threshold
    s_roadwear_thresh: float = 0.0

    # Pavement type scaling factors
    num_pave: int = 1  # reallocated
    h_pave_str: List[str] = field(default_factory=lambda: [""])
    h_pave: List[float] = field(default_factory=lambda: [0.0])

    # Driving cycle scaling factors
    num_dc: int = 1  # reallocated
    h_drivingcycle_str: List[str] = field(default_factory=lambda: [""])
    h_drivingcycle: List[float] = field(default_factory=lambda: [0.0])

    # Suspension scaling factors - h_0_sus[source, size]
    h_0_sus: np.ndarray = field(
        default_factory=lambda: np.zeros((constants.num_source, constants.num_size))
    )
    h_0_q_road: np.ndarray = field(default_factory=lambda: np.zeros(constants.num_size))

    # Road suspension factors - f_0_suspension[source, size, tyre_type, vehicle_type]
    f_0_suspension: np.ndarray = field(
        default_factory=lambda: np.zeros(
            (
                constants.num_source,
                constants.num_size,
                constants.num_tyre,
                constants.num_veh,
            )
        )
    )
    a_sus: np.ndarray = field(default_factory=lambda: np.zeros(5))

    # Abrasion factors - f_0_abrasion[tyre_type, vehicle_type]
    f_0_abrasion: np.ndarray = field(
        default_factory=lambda: np.zeros((constants.num_tyre, constants.num_veh))
    )
    V_ref_abrasion: float = 0.0
    h_0_abrasion: np.ndarray = field(
        default_factory=lambda: np.ones(constants.num_size)
    )  # Default to 1

    # Crushing factors - f_0_crushing[tyre_type, vehicle_type]
    f_0_crushing: np.ndarray = field(
        default_factory=lambda: np.zeros((constants.num_tyre, constants.num_veh))
    )
    V_ref_crushing: float = 0.0
    h_0_crushing: np.ndarray = field(
        default_factory=lambda: np.ones(constants.num_size)
    )  # Default to 1

    # Source participation in abrasion and crushing
    p_0_abrasion: np.ndarray = field(
        default_factory=lambda: np.zeros(constants.num_source)
    )
    p_0_crushing: np.ndarray = field(
        default_factory=lambda: np.zeros(constants.num_source)
    )

    # Direct emission factors
    f_0_dir: np.ndarray = field(
        default_factory=lambda: np.ones(constants.num_source_all_extra)
    )  # Default to 1

    # Fractional size distribution - f_PM[source, size, tyre_type]
    f_PM: np.ndarray = field(
        default_factory=lambda: np.zeros(
            (constants.num_source_all_extra, constants.num_size, constants.num_tyre)
        )
    )
    f_PM_bin: np.ndarray = field(
        default_factory=lambda: np.zeros(
            (constants.num_source_all_extra, constants.num_size, constants.num_tyre)
        )
    )
    V_ref_pm_fraction: float = 0.0
    c_pm_fraction: float = 0.0

    # Wind blown dust parameters
    tau_wind: float = 0.0
    FF_thresh: float = 0.0

    # Activity efficiency factors - h_eff[efficiency_type, source_index, size]
    h_eff: np.ndarray = field(
        default_factory=lambda: np.zeros(
            (4, constants.num_source_all - 1, constants.num_size)
        )
    )

    # Deposition velocities
    w_dep: np.ndarray = field(default_factory=lambda: np.zeros(constants.num_size))

    # Concentration conversion limit values
    conc_min: float = 0.0
    emis_min: float = 0.0

    # Spray and splash factors
    R_0_spray: np.ndarray = field(
        default_factory=lambda: np.zeros((constants.num_veh, constants.num_moisture))
    )
    V_ref_spray: np.ndarray = field(
        default_factory=lambda: np.zeros(constants.num_moisture)
    )
    g_road_sprayable_min: np.ndarray = field(
        default_factory=lambda: np.zeros(constants.num_moisture)
    )
    a_spray: np.ndarray = field(
        default_factory=lambda: np.zeros(constants.num_moisture)
    )
    V_thresh_spray: np.ndarray = field(
        default_factory=lambda: np.zeros(constants.num_moisture)
    )

    # Drainage parameters
    g_road_drainable_min: float = 0.0
    g_road_drainable_thresh: float = 0.0
    snow_dust_drainage_retainment_limit: float = 0.0
    tau_road_drainage: float = 0.0

    # Ploughing parameters
    h_ploughing_moisture: np.ndarray = field(
        default_factory=lambda: np.zeros(constants.num_moisture)
    )
    ploughing_thresh_moisture: np.ndarray = field(
        default_factory=lambda: np.zeros(constants.num_moisture)
    )

    # Energy balance parameters
    g_road_evaporation_thresh: float = 0.0
    z0: float = 0.0  # Roughness length in meters
    albedo_snow: float = 0.0
    dzs: float = 0.0
    sub_surf_average_time: float = 0.0
    sub_surf_param: np.ndarray = field(default_factory=lambda: np.zeros(3))
    a_traffic: np.ndarray = field(default_factory=lambda: np.zeros(constants.num_veh))
    H_veh: np.ndarray = field(default_factory=lambda: np.zeros(constants.num_veh))

    # Retention parameters - using arrays instead of dict
    g_retention_thresh: np.ndarray = field(
        default_factory=lambda: np.zeros(constants.num_source)
    )
    g_retention_min: np.ndarray = field(
        default_factory=lambda: np.zeros(constants.num_source)
    )

    # OSPM parameters
    f_roof_ospm_override: float = 0.0
    f_turb_ospm_override: float = 0.0

    # Surface texture parameters
    texture_scaling: np.ndarray = field(
        default_factory=lambda: np.ones(5)
    )  # Default to 1

    # Track parameters
    num_track: int = 1  # reallocated on the fly
    f_track: List[float] = field(default_factory=lambda: [1.0])
    veh_track: List[float] = field(default_factory=lambda: [1.0])
    mig_track: List[float] = field(default_factory=lambda: [1.0])
    track_type: List[int] = field(default_factory=lambda: [1])
