import numpy as np
import constants
from functions import (
    r_aero_func_with_stability,
    r_aero_func,
    dewpoint_from_rh_func,
    f_spray_func,
    mass_balance_func,
    surface_energy_model_4_func,
    e_diff_func,
    relax_func,
)
from initialise.road_dust_initialise_time import time_config
from initialise.road_dust_initialise_variables import model_variables
from input_classes import (
    converted_data,
    input_activity,
    input_metadata,
    input_meteorology,
)
from config_classes import (
    model_parameters,
    model_flags,
)


def road_dust_surface_wetness(
    ti: int,
    tr: int,
    ro: int,
    time_config: time_config,
    converted_data: converted_data,
    model_variables: model_variables,
    model_parameters: model_parameters,
    model_flags: model_flags,
    metadata: input_metadata,
    input_activity: input_activity,
    tf: int = None,
    meteorology_input: "input_meteorology" = None,
):
    """
    Calculate road surface moisture and retention in the NORTRIP model.

    This function calculates surface moisture, evaporation, drainage, spray,
    and retention factors for road dust modeling.

    Args:
        ti: Current time index
        tr: Track index
        ro: Road index
        time_config: Time configuration object
        converted_data: Converted input data
        model_variables: Model variables object (modified in place)
        model_parameters: Model parameters
        model_flags: Model flags
        metadata: Input metadata
        input_activity: Activity input data
        tf: Forecast time index (optional)
        meteorology_input: Meteorology input data (optional)
    """

    surface_moisture_min = 0.000001

    dz_snow_albedo = 3  # Depth of snow required before implementing snow albedo mm.w.e.
    Z_CLOUD = 100  # Only used when no global radiation is available
    z0t = model_parameters.z0 / 10

    length_veh = np.zeros(constants.num_veh)
    length_veh[constants.li] = 5.0
    length_veh[constants.he] = 15.0

    retain_water_by_snow = 1
    b_factor = 1 / (1000 * metadata.b_road_lanes * model_parameters.f_track[tr])

    g_road_0_data = np.full(constants.num_moisture, metadata.nodata)
    S_melt_temp = metadata.nodata
    g_road_fraction = np.full(constants.num_moisture, metadata.nodata)
    M2_road_salt_0 = np.full(constants.num_salt, metadata.nodata)
    g_road_temp = metadata.nodata
    R_evaporation = np.zeros(constants.num_moisture)
    R_ploughing = np.zeros(constants.num_moisture)
    R_road_drainage = np.zeros(constants.num_moisture)
    R_spray = np.zeros(constants.num_moisture)
    R_drainage = np.zeros(constants.num_moisture)
    melt_temperature_salt_temp = metadata.nodata
    RH_salt_temp = metadata.nodata
    M_road_dissolved_ratio_temp = metadata.nodata
    T_s_0 = metadata.nodata
    T_a_0 = metadata.nodata
    FF_0 = metadata.nodata
    RH_s_0 = metadata.nodata
    short_rad_net_temp = metadata.nodata
    g_road_drainable_withrain = metadata.nodata
    g_road_drainable_min_temp = metadata.nodata
    g_road_total = metadata.nodata
    g_ratio_road = metadata.nodata
    g_ratio_brake = metadata.nodata
    g_ratio_binder = 0
    g_ratio_obs = 0
    g_road_sprayable = 0

    g_road_0_data[: constants.num_moisture, 0] = model_variables.g_road_data[
        constants.num_moisture, np.max(time_config.min_time, ti - 1), tr, ro
    ]
