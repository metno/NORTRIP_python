import numpy as np
from typing import Tuple
import constants as c
from .antoine_func import antoine_func


def salt_solution_func(
    M2_road_salt: np.ndarray,
    g_road: float,
    s_road: float,
    T_s: float,
    salt_type: np.ndarray,
    dt_h: float,
    dissolution_flag: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float, float]:
    """
    Calculate salt solution, melt temperature and RH salt.

    This function calculates salt solution properties, melt temperature,
    and relative humidity for salt solutions on road surfaces.

    Args:
        M2_road_salt: Salt mass array for each salt type (g/m²)
        g_road: Surface moisture (water) (mm)
        s_road: Surface moisture (ice/snow) (mm)
        T_s: Surface temperature (°C)
        salt_type: List of salt type indices
        dt_h: Time step (hours)
        dissolution_flag: Whether to apply dissolution calculations

    Returns:
        tuple: (melt_temperature_salt, RH_salt, M_road_dissolved_ratio,
                g_road_out, s_road_out, g_road_at_T_s_out, s_road_at_T_s_out)
            melt_temperature_salt: Melt temperature for each salt type (°C)
            RH_salt: Relative humidity for each salt type (%)
            M_road_dissolved_ratio: Dissolved salt ratio for each salt type
            g_road_out: Output surface water (mm)
            s_road_out: Output surface ice/snow (mm)
            g_road_at_T_s_out: Equilibrium water at surface temperature (mm)
            s_road_at_T_s_out: Equilibrium ice/snow at surface temperature (mm)
    """
    # Time scale for melting by salt. Greater than 0 delays the dissolvement
    tau = 1.0

    # Use the oversaturated parameterisation for melt temperature and water vapour
    use_oversaturated = True

    # Declarations and initializations
    num_salt = len(salt_type)
    N_moles_salt = np.zeros(num_salt)
    afactor = np.zeros(num_salt)
    dissolved_salt = np.zeros(num_salt)
    solution_salt_at_T_s = np.zeros(num_salt)
    N_moles_water_at_T_s = np.zeros(num_salt)
    g_road_at_T_s = np.zeros(num_salt)
    s_road_at_T_s = np.zeros(num_salt)
    solution_salt = np.zeros(num_salt)
    melt_temperature_salt = np.zeros(num_salt)
    RH_salt = np.zeros(num_salt)
    M_road_dissolved_ratio = np.zeros(num_salt)

    N_moles_water = 0.0
    T_0 = 273.13
    surface_moisture_min = 1e-6

    # Convert surface moisture to moles per m²
    N_moles_water = 1000 * g_road / c.M_ATOMIC_WATER

    if dissolution_flag:
        # Calculate the salt equilibrium water/ice dependent on temperature
        for i in range(num_salt):
            salt_idx = salt_type[i]

            # Determine moles of salt /m². M2 means salt is in g/m²
            N_moles_salt[i] = max(0, M2_road_salt[i] / c.M_ATOMIC[salt_idx])

            # Determine the melt based on instantaneous dissolving of the salt
            # in the ice and snow surface to achieve a melt temperature the same as
            # the road surface temperature
            salt_power = c.SALT_POWER_VAL[salt_idx]

            if T_s < 0 and T_s >= c.MELT_TEMPERATURE_SATURATED[salt_idx]:
                solution_salt_at_T_s[i] = c.SATURATED[salt_idx] * (
                    T_s / c.MELT_TEMPERATURE_SATURATED[salt_idx]
                ) ** (1 / salt_power)
                N_moles_water_at_T_s[i] = (
                    N_moles_salt[i] / solution_salt_at_T_s[i] - N_moles_salt[i]
                )
                g_road_at_T_s[i] = min(
                    g_road + s_road, N_moles_water_at_T_s[i] * c.M_ATOMIC_WATER / 1000
                )
                s_road_at_T_s[i] = max(0, (g_road + s_road) - g_road_at_T_s[i])

            elif T_s >= 0:
                g_road_at_T_s[i] = g_road + s_road
                s_road_at_T_s[i] = 0

            elif T_s < c.MELT_TEMPERATURE_SATURATED[salt_idx]:
                g_road_at_T_s[i] = 0
                s_road_at_T_s[i] = s_road + g_road

            else:
                g_road_at_T_s[i] = s_road + g_road
                s_road_at_T_s[i] = s_road + g_road

        g_road_at_T_s_out = np.max(g_road_at_T_s) if len(g_road_at_T_s) > 0 else g_road
        s_road_at_T_s_out = g_road + s_road - g_road_at_T_s_out

        # Only apply dissolution time scale in the melt direction
        if g_road_at_T_s_out > g_road:
            g_road_out = g_road * np.exp(-dt_h / tau) + g_road_at_T_s_out * (
                1 - np.exp(-dt_h / tau)
            )
        else:
            g_road_out = g_road_at_T_s_out

        s_road_out = max(0, (g_road + s_road) - g_road_out)

        N_moles_water = 1000 * (g_road_out + surface_moisture_min) / c.M_ATOMIC_WATER

    else:
        g_road_out = g_road
        s_road_out = s_road
        # Set equilibrium limit to total moisture
        g_road_at_T_s_out = g_road + s_road
        s_road_at_T_s_out = g_road + s_road
        N_moles_water = 1000 * (g_road_out + surface_moisture_min) / c.M_ATOMIC_WATER

        for i in range(num_salt):
            salt_idx = salt_type[i]
            N_moles_salt[i] = max(0, M2_road_salt[i] / c.M_ATOMIC[salt_idx])

    # Calculate vapour pressure and melt temperature of the solution (vp)
    for i in range(num_salt):
        salt_idx = salt_type[i]
        salt_power = c.SALT_POWER_VAL[salt_idx]

        solution_salt[i] = max(0, N_moles_salt[i] / (N_moles_water + N_moles_salt[i]))

        vp_ice = antoine_func(c.A_ANTOINE_ICE, c.B_ANTOINE_ICE, c.C_ANTOINE_ICE, T_s)
        vp_s = antoine_func(
            float(c.A_ANTOINE[salt_idx]),
            float(c.B_ANTOINE[salt_idx]),
            float(c.C_ANTOINE[salt_idx]),
            T_s,
        ) + float(c.VP_CORRECTION[salt_idx])

        antoine_scaling = vp_ice / vp_s if vp_s > 0 else 1.0

        if solution_salt[i] > c.SATURATED[salt_idx]:
            afactor[i] = 1.0
        else:
            afactor[i] = (1 - antoine_scaling) * (
                solution_salt[i] / c.SATURATED[salt_idx]
            ) ** salt_power + antoine_scaling

        RH_salt[i] = min(100, 100 * afactor[i] * vp_s / vp_ice)
        RH_salt_saturated = min(100.0, 100 * vp_s / vp_ice)
        RH_salt[i] = max(RH_salt_saturated, RH_salt[i])

        melt_temperature_salt[i] = max(
            c.MELT_TEMPERATURE_SATURATED[salt_idx],
            ((solution_salt[i] / c.SATURATED[salt_idx]) ** salt_power)
            * c.MELT_TEMPERATURE_SATURATED[salt_idx],
        )

        # Adjust for the oversaturated case
        if use_oversaturated:
            # Oversaturated
            if solution_salt[i] > c.SATURATED[salt_idx]:
                solution_salt[i] = min(c.OVER_SATURATED[salt_idx], solution_salt[i])

                RH_over_saturated = (
                    100 * (1 - c.RH_OVER_SATURATED_FRACTION[salt_idx])
                    + RH_salt_saturated * c.RH_OVER_SATURATED_FRACTION[salt_idx]
                )

                RH_salt[i] = min(
                    100,
                    RH_salt_saturated
                    + (RH_over_saturated - RH_salt_saturated)
                    / (
                        c.F_SALT_SAT[salt_idx] * c.SATURATED[salt_idx]
                        - c.SATURATED[salt_idx]
                    )
                    * (solution_salt[i] - c.SATURATED[salt_idx]),
                )

                melt_temperature_salt[i] = min(
                    0,
                    c.MELT_TEMPERATURE_SATURATED[salt_idx]
                    + (
                        c.MELT_TEMPERATURE_OVERSATURATED[salt_idx]
                        - c.MELT_TEMPERATURE_SATURATED[salt_idx]
                    )
                    / (
                        c.F_SALT_SAT[salt_idx] * c.SATURATED[salt_idx]
                        - c.SATURATED[salt_idx]
                    )
                    * (solution_salt[i] - c.SATURATED[salt_idx]),
                )

        # Calculate dissolved mass of salt
        if solution_salt[i] < c.SATURATED[salt_idx]:
            dissolved_salt[i] = N_moles_salt[i] * c.M_ATOMIC[salt_idx]
        else:
            dissolved_salt[i] = (
                c.SATURATED[salt_idx]
                * N_moles_water
                / (1 - c.SATURATED[salt_idx])
                * c.M_ATOMIC[salt_idx]
            )

        if M2_road_salt[i] > 0:
            M_road_dissolved_ratio[i] = dissolved_salt[i] / M2_road_salt[i]
        else:
            M_road_dissolved_ratio[i] = 1.0

        # Set exact limits
        M_road_dissolved_ratio[i] = max(0, min(1, M_road_dissolved_ratio[i]))

    return (
        melt_temperature_salt,
        RH_salt,
        M_road_dissolved_ratio,
        g_road_out,
        s_road_out,
        g_road_at_T_s_out,
        s_road_at_T_s_out,
    )
