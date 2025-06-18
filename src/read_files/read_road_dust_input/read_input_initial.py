import pandas as pd
import constants
from config_classes import model_parameters
from input_classes import input_metadata
from input_classes import input_initial, input_values_initial
from pd_util import find_float_or_default
import logging

logger = logging.getLogger(__name__)


def read_input_initial(
    initial_df: pd.DataFrame,
    model_parameters: model_parameters,
    input_metadata: input_metadata,
    print_results: bool = False,
):
    """
    Read and process initial input data for NORTRIP.

    """
    b_road_lanes = input_metadata.b_road_lanes
    f_track = model_parameters.f_track
    num_track = model_parameters.num_track

    # Parse all possible values into loaded_values_initial
    loaded_values_initial = input_values_initial()
    header_col = initial_df.iloc[:, 0]
    data_col = initial_df.iloc[:, 1]

    for field_name in loaded_values_initial.__dataclass_fields__:
        default_val = getattr(loaded_values_initial, field_name)
        new_value = find_float_or_default(field_name, header_col, data_col, default_val)
        setattr(loaded_values_initial, field_name, new_value)

    # Prepare output
    loaded_initial = input_initial()

    # Assign to m_road_init (track 0)
    loaded_initial.m_road_init[constants.road_index, 0] = (
        loaded_values_initial.M_dust_road
        if loaded_values_initial.M2_dust_road == 0
        else loaded_values_initial.M2_dust_road * b_road_lanes * 1000
    )
    loaded_initial.m_road_init[constants.sand_index, 0] = (
        loaded_values_initial.M_sand_road
        if loaded_values_initial.M2_sand_road == 0
        else loaded_values_initial.M2_sand_road * b_road_lanes * 1000
    )
    loaded_initial.m_road_init[constants.salt_index[0], 0] = (
        loaded_values_initial.M_salt_road_na
        if loaded_values_initial.M2_salt_road_na == 0
        else loaded_values_initial.M2_salt_road_na * b_road_lanes * 1000
    )
    # Salt index 1 (mg, cma, ca)
    salt2 = 0

    # Select first non-zero value from M2_salt_road_*
    for i in [
        loaded_values_initial.M2_salt_road_mg,
        loaded_values_initial.M2_salt_road_cma,
        loaded_values_initial.M2_salt_road_ca,
    ]:
        if i != 0:
            salt2 = i * b_road_lanes * 1000
            break

    # If no non-zero value found, select first non-zero value from M_salt_road_*
    if salt2 == 0:
        for i in [
            loaded_values_initial.M_salt_road_mg,
            loaded_values_initial.M_salt_road_cma,
            loaded_values_initial.M_salt_road_ca,
        ]:
            if i != 0:
                salt2 = i
                break

    loaded_initial.m_road_init[constants.salt_index[1], 0] = salt2

    # Assign to g_road_init (track 0)
    loaded_initial.g_road_init[constants.water_index, 0] = (
        loaded_values_initial.g_road
        if loaded_values_initial.water_road == 0
        else loaded_values_initial.water_road
    )
    loaded_initial.g_road_init[constants.snow_index, 0] = (
        loaded_values_initial.s_road
        if loaded_values_initial.snow_road == 0
        else loaded_values_initial.snow_road
    )
    loaded_initial.g_road_init[constants.ice_index, 0] = (
        loaded_values_initial.i_road
        if loaded_values_initial.ice_road == 0
        else loaded_values_initial.ice_road
    )

    # Distribute over tracks
    for tr in range(num_track):
        loaded_initial.m_road_init[:, tr] = (
            loaded_initial.m_road_init[:, 0] * f_track[tr]
        )
        loaded_initial.g_road_init[:, tr] = loaded_initial.g_road_init[:, 0]

    loaded_initial.long_rad_in_offset = loaded_values_initial.long_rad_in_offset
    loaded_initial.RH_offset = loaded_values_initial.RH_offset
    loaded_initial.T_2m_offset = loaded_values_initial.T_2m_offset
    loaded_initial.P_fugitive = loaded_values_initial.P_fugitive
    loaded_initial.P2_fugitive = loaded_values_initial.P2_fugitive

    return loaded_initial
