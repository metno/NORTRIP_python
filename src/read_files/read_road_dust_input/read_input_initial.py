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
) -> input_initial:
    """
    Read and process initial input data for NORTRIP.

    Args:
        initial_df (pd.DataFrame): DataFrame containing the initial data
        model_parameters (model_parameters): Model parameters
        input_metadata (input_metadata): Input metadata
        print_results (bool): Whether to print the results to the console

    Returns:
        input_initial: Dataclass containing the initial data
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

    # Calculate initial masses for track 0 (before distribution)
    dust_mass = _get_mass_value(
        loaded_values_initial.M_dust_road,
        loaded_values_initial.M2_dust_road,
        b_road_lanes,
    )
    sand_mass = _get_mass_value(
        loaded_values_initial.M_sand_road,
        loaded_values_initial.M2_sand_road,
        b_road_lanes,
    )
    salt_na_mass = _get_mass_value(
        loaded_values_initial.M_salt_road_na,
        loaded_values_initial.M2_salt_road_na,
        b_road_lanes,
    )

    # Handle secondary salt (mg, cma, ca) with priority logic
    salt_other_mass = _get_secondary_salt_mass(loaded_values_initial, b_road_lanes)

    # Calculate initial moisture values for track 0
    water_value = _get_moisture_value(
        loaded_values_initial.g_road, loaded_values_initial.water_road
    )
    snow_value = _get_moisture_value(
        loaded_values_initial.s_road, loaded_values_initial.snow_road
    )
    ice_value = _get_moisture_value(
        loaded_values_initial.i_road, loaded_values_initial.ice_road
    )

    # Distribute mass and moisture values across all tracks
    for tr in range(num_track):
        track_multiplier = f_track[tr]

        # Mass values (distributed according to f_track)
        loaded_initial.m_road_init[constants.road_index, tr] = (
            dust_mass * track_multiplier
        )
        loaded_initial.m_road_init[constants.sand_index, tr] = (
            sand_mass * track_multiplier
        )
        loaded_initial.m_road_init[constants.salt_index[0], tr] = (
            salt_na_mass * track_multiplier
        )
        loaded_initial.m_road_init[constants.salt_index[1], tr] = (
            salt_other_mass * track_multiplier
        )

        # Moisture values (same for all tracks)
        loaded_initial.g_road_init[constants.water_index, tr] = water_value
        loaded_initial.g_road_init[constants.snow_index, tr] = snow_value
        loaded_initial.g_road_init[constants.ice_index, tr] = ice_value

    # Set offset and fugitive parameters
    loaded_initial.long_rad_in_offset = loaded_values_initial.long_rad_in_offset
    loaded_initial.RH_offset = loaded_values_initial.RH_offset
    loaded_initial.T_2m_offset = loaded_values_initial.T_2m_offset
    loaded_initial.P_fugitive = loaded_values_initial.P_fugitive

    return loaded_initial


def _get_mass_value(m_value: float, m2_value: float, b_road_lanes: float) -> float:
    """
    Get mass value, prioritizing M2_ (mass per area) over M_ (total mass).

    Args:
        m_value: Total mass value (M_*)
        m2_value: Mass per area value (M2_*)
        b_road_lanes: Road lane width

    Returns:
        Final mass value
    """
    if m2_value != 0:
        return m2_value * b_road_lanes * 1000
    return m_value


def _get_moisture_value(primary_value: float, alternative_value: float) -> float:
    """
    Get moisture value, using alternative if primary is zero.

    Args:
        primary_value: Primary moisture value (e.g., g_road)
        alternative_value: Alternative moisture value (e.g., water_road)

    Returns:
        Final moisture value
    """
    if alternative_value != 0:
        return alternative_value
    return primary_value


def _get_secondary_salt_mass(
    loaded_values_initial: input_values_initial, b_road_lanes: float
) -> float:
    """
    Get secondary salt mass (mg, cma, ca) with priority logic.
    Priority: M2_ values first, then M_ values.

    Args:
        loaded_values_initial: Loaded initial values
        b_road_lanes: Road lane width

    Returns:
        Secondary salt mass value
    """
    # Check M2_ values first (highest priority)
    m2_values = [
        loaded_values_initial.M2_salt_road_mg,
        loaded_values_initial.M2_salt_road_cma,
        loaded_values_initial.M2_salt_road_ca,
    ]

    for m2_value in m2_values:
        if m2_value != 0:
            return m2_value * b_road_lanes * 1000

    # If no M2_ values, check M_ values
    m_values = [
        loaded_values_initial.M_salt_road_mg,
        loaded_values_initial.M_salt_road_cma,
        loaded_values_initial.M_salt_road_ca,
    ]

    for m_value in m_values:
        if m_value != 0:
            return m_value

    return 0.0
