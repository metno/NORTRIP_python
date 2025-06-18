from dataclasses import dataclass


@dataclass
class input_values_initial:
    """
    Dataclass containing values for initial input data.
    """

    M_dust_road: float = 0.0
    M2_dust_road: float = 0.0
    M_sand_road: float = 0.0
    M2_sand_road: float = 0.0
    M_salt_road_na: float = 0.0
    M2_salt_road_na: float = 0.0
    M_salt_road_mg: float = 0.0
    M2_salt_road_mg: float = 0.0
    M_salt_road_cma: float = 0.0
    M2_salt_road_cma: float = 0.0
    M_salt_road_ca: float = 0.0
    M2_salt_road_ca: float = 0.0
    g_road: float = 0.0
    water_road: float = 0.0
    s_road: float = 0.0
    snow_road: float = 0.0
    i_road: float = 0.0
    ice_road: float = 0.0
    long_rad_in_offset: float = 0.0
    RH_offset: float = 0.0
    T_2m_offset: float = 0.0
    P_fugitive: float = 0.0
    P2_fugitive: float = 0.0
