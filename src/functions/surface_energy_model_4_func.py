import numpy as np
from typing import Tuple
from .q_sat_func import q_sat_func
from .q_sat_ice_func import q_sat_ice_func
from .salt_solution_func import salt_solution_func


def surface_energy_model_4_func(
    short_net: float,
    long_in: float,
    H_traffic: float,
    r_aero: float,
    TC: float,
    TCs_in: float,
    TCsub: float,
    RH: float,
    RHs_nosalt: float,
    RHs_0: float,
    P: float,
    dzs_in: float,
    dt_h_in: float,
    g_surf_in: float,
    s_surf_in: float,
    g_min: float,
    M2_road_salt_0: np.ndarray,
    salt_type: np.ndarray,
    sub_surf_param: np.ndarray,
    surface_humidity_flag: int,
    use_subsurface_flag: int,
    use_salt_humidity_flag: int,
    E_correction: float,
) -> Tuple[
    float,
    float,
    float,
    float,
    np.ndarray,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
]:
    """
    Surface energy model with salt solution effects.

    Includes melt of snow as output and includes snow surface and melt temperature as
    additional inputs. Also includes a relationship for vapour pressure of ice.

    Args:
        short_net: Net shortwave radiation (W/m²)
        long_in: Incoming longwave radiation (W/m²)
        H_traffic: Traffic heat flux (W/m²)
        r_aero: Aerodynamic resistance (s/m)
        TC: Air temperature (°C)
        TCs_in: Initial surface temperature (°C)
        TCsub: Subsurface temperature (°C)
        RH: Relative humidity (%)
        RHs_nosalt: Surface relative humidity without salt (%)
        RHs_0: Previous surface relative humidity (%)
        P: Pressure (Pa)
        dzs_in: Subsurface depth (m)
        dt_h_in: Time step (hours)
        g_surf_in: Surface water (mm)
        s_surf_in: Surface ice/snow (mm)
        g_min: Minimum surface moisture threshold (mm)
        M2_road_salt_0: Salt mass for each salt type (g/m²)
        salt_type: List of salt type indices
        sub_surf_param: Subsurface parameters [rho_s, c_s, k_s_road]
        surface_humidity_flag: Surface humidity calculation flag
        use_subsurface_flag: Subsurface calculation flag
        use_salt_humidity_flag: Salt humidity calculation flag
        E_correction: Energy correction (W/m²)

    Returns:
        tuple: (TCs_out, melt_temperature, RH_salt_final, RHs, M_road_dissolved_ratio_temp,
                evap, evap_pot, melt, freeze, H, L, G, long_out, long_net,
                rad_net, G_sub, G_freeze, G_melt)
    """

    # Set limit flags to reduce oscillations
    set_limit_noevap = 1
    limit_evap = 1
    limit_condens = 1
    dissolution_flag = 1

    # Set time step in seconds
    dt_sec = dt_h_in * 3600.0

    # Initialize variables
    G_melt = 0.0
    G_freeze = 0.0
    melt = 0.0
    freeze = 0.0
    E_diff = 0.0
    g_surf = g_surf_in
    s_surf = s_surf_in

    # Set constants
    Cp = 1006.0
    lambda_val = 2.50e6
    lambda_ice = 2.83e6
    lambda_melt = 3.33e5  # J/kg
    RD = 287.0
    T0C = 273.15
    sigma = 5.67e-8
    eps_s = 0.95

    # Set subsurface parameters
    rho_s = sub_surf_param[0]
    c_s = sub_surf_param[1]
    k_s_road = sub_surf_param[2]
    C_s = rho_s * c_s
    omega = 7.3e-5

    # Automatically set dzs if it is 0
    if dzs_in == 0:
        dzs = (k_s_road / C_s / 2 / omega) ** 0.5
    else:
        dzs = dzs_in

    mu = omega * C_s * dzs

    # If subsurface flux is turned off
    if use_subsurface_flag == 0:
        mu = 0.0

    # Set atmospheric temperature in Kelvin
    TK_a = T0C + TC

    # Set air density (P is in hPa, convert to Pa by multiplying by 100)
    rho = P * 100.0 / (RD * TK_a)

    # Initialize surface temperature
    TCs = TCs_in
    TCs_0 = TCs_in
    TCs_out = TCs_in

    # Sub time settings (not used in this implementation)
    nsub = 1
    dt_sec = dt_sec / nsub
    dt_h = dt_h_in / nsub

    # Set internal limits of latent heat flux
    L_max = 500.0
    L_min = -200.0

    # Present values of constants for implicit solution
    a_G = 1.0 / (C_s * dzs)
    a_rad = short_net + long_in + H_traffic
    a_RL = (1 - 4 * TC / TK_a) * eps_s * sigma * TK_a**4
    b_RL = 4 * eps_s * sigma * TK_a**3
    a_H = rho * Cp / r_aero

    # Specific humidity of the air
    esat, qsat, s = q_sat_func(TC, P)
    q = qsat * RH / 100.0

    # Initialize the evaporation
    evap = 0.0

    # Start the sub time routine
    for ti_sub in range(nsub):
        evap_0 = evap

        # Loop twice to update the latent heat flux and melt with the new surface temperature
        for i in range(1):  # Only one iteration in this implementation
            # Calculate the salt solution, melt temperature and freezing and melting
            if use_salt_humidity_flag:
                no_salt_factor = 1.0
            else:
                no_salt_factor = 0.0

            # Calculate the salt solution and change in water and ice/snow
            (
                melt_temperature_salt_temp,
                RH_salt_temp,
                M_road_dissolved_ratio_temp,
                g_road_temp,
                s_road_temp,
                g_road_equil_at_T_s,
                s_road_equil_at_T_s,
            ) = salt_solution_func(
                M2_road_salt_0 * no_salt_factor,
                g_surf_in,
                s_surf_in,
                TCs,
                salt_type,
                dt_h,
                dissolution_flag,
            )

            # Determine the melt or freezing due to dissolution of salt
            if i == 0:
                freeze = max(0.0, s_road_temp - s_surf_in)
                melt = max(0.0, g_road_temp - g_surf_in)

            # Use the salt with the lowest melt temperature
            melt_temperature = np.min(melt_temperature_salt_temp)

            # Set the energy used for freezing or melting
            if i == 1:  # This won't execute with current loop range
                G_freeze = freeze * lambda_melt / dt_sec
                G_melt = melt * lambda_melt / dt_sec

            # Set the surface salt humidity to be the lowest for the two salts
            RH_salt_final = np.min(RH_salt_temp)
            RH_salt_final = max(RH_salt_final, 1.0)  # Cannot be 0

            # Set the final surface humidity based on surface and salt humidity
            RHs = RHs_nosalt * RH_salt_final / 100.0

            # Smooth the RH in time to avoid oscillations
            fac = 0.333
            RHs = RHs * (1 - fac) + fac * RHs_0

            # Update g_surf to new values
            g_surf = g_road_temp
            s_surf = s_road_temp

            # Do not allow the salt equilibrium to freeze water
            if freeze > 0 and melt == 0:
                G_melt = 0.0
                G_freeze = 0.0
                melt = 0.0
                freeze = 0.0
                g_surf = g_surf_in
                s_surf = s_surf_in

            # Set sum of the water and ice and fraction
            g_s_surf = g_surf + s_surf
            if g_s_surf > 0:
                g_surf_fraction = g_surf / g_s_surf
                s_surf_fraction = s_surf / g_s_surf
            else:
                g_surf_fraction = 0.5
                s_surf_fraction = 0.5

            # Weight lambda coefficient according to water and ice distribution
            lambda_mixed = g_surf_fraction * lambda_val + s_surf_fraction * lambda_ice

            # Specific surface humidity based on current surface temperature (TCs)
            esat, qsats_water, s = q_sat_func(TCs, P)
            esat_ice, qsats_ice, s_ice = q_sat_ice_func(TCs, P)
            qsats = g_surf_fraction * qsats_water + s_surf_fraction * qsats_ice
            qs_water = qsats_water * RHs / 100.0
            qs_ice = qsats_ice * RHs / 100.0

            # Latent heat flux
            L_water = -rho * lambda_val * (q - qs_water) / r_aero
            L_ice = -rho * lambda_ice * (q - qs_ice) / r_aero

            # Limit latent heat flux to reasonable values
            L_water = max(L_min, min(L_max, L_water))
            L_ice = max(L_min, min(L_max, L_ice))
            L = g_surf_fraction * L_water + s_surf_fraction * L_ice

            # Set evaporation (convert from W/m² to mm over time step)
            # L_water is in W/m² = kg/s³, lambda_val is in J/kg = m²/s²
            # L_water/lambda_val gives kg/m²/s
            # Multiply by dt_sec to get kg/m² over time step
            # Since water density is 1000 kg/m³, kg/m² = mm
            evap_water = L_water / lambda_val * dt_sec  # mm
            evap_ice = L_ice / lambda_ice * dt_sec  # mm
            evap = g_surf_fraction * evap_water + s_surf_fraction * evap_ice

            # Limit evaporation to the amount of water available
            ratio_equil = RH * qsat / RH_salt_final / qsats
            if set_limit_noevap == 1:
                if surface_humidity_flag == 1:
                    g_equil = ratio_equil * g_min
                elif surface_humidity_flag == 2:
                    if ratio_equil < 1:
                        g_equil = -np.log(1 - ratio_equil) * g_min / 4.0
                    else:
                        g_equil = g_min * 1000.0  # Large number as limit
                else:
                    g_equil = 0.0
            else:
                g_equil = 0.0

            if limit_evap:
                if evap >= (g_s_surf - g_equil) / dt_h and evap >= 0:
                    evap = max(0.0, (g_s_surf - g_equil) / dt_h)
                    L = evap * lambda_mixed * dt_h / dt_sec  # Convert mm to W/m²
                    L = max(L_min, min(L_max, L))

            if limit_condens:
                if g_equil <= g_min and evap < 0:
                    if evap <= (g_s_surf - g_equil) / dt_h and evap < 0:
                        evap = min(0.0, (g_s_surf - g_equil) / dt_h)
                        L = evap * lambda_mixed * dt_h / dt_sec  # Convert mm to W/m²
                        L = max(L_min, min(L_max, L))

            # Calculate surface temperature implicitly to avoid instabilities
            TCs_out = (
                TCs_0
                + dt_sec
                * a_G
                * (
                    a_rad
                    - a_RL
                    - L
                    + a_H * TC
                    + mu * TCsub
                    - G_melt
                    + G_freeze
                    + E_correction
                )
            ) / (1 + dt_sec * a_G * (a_H + b_RL + mu))

            # Reset the current temperature for the iteration
            TCs = (TCs_0 + TCs_out) / 2.0

            # Diagnose sensible heat flux based on average surface temperature
            H = -rho * Cp * (TC - TCs) / r_aero

            # Diagnose potential evaporation
            L_pot_water = -rho * lambda_val * (q - qsats_water) / r_aero
            L_pot_ice = -rho * lambda_ice * (q - qsats_ice) / r_aero
            evap_pot_water = L_pot_water / lambda_val * dt_sec  # mm
            evap_pot_ice = L_pot_ice / lambda_ice * dt_sec  # mm
            evap_pot = g_surf_fraction * evap_pot_water + s_surf_fraction * evap_pot_ice

            # Note: evap_pot calculation matches MATLAB - no scaling needed

            # Diagnose radiation based on current average temperature
            long_out = eps_s * sigma * (T0C + TCs) ** 4
            long_net = long_in - long_out
            rad_net = short_net + long_net
            G_sub = -mu * (TCs - TCsub)

            # Diagnose surface flux for additional melting or freezing
            G = rad_net - H - L + H_traffic - G_melt + G_freeze

            if i == 0:
                # Calculate additional melt in first loop only
                if s_surf > 0 and G >= 0 and (TCs >= melt_temperature):
                    melt_energy = (max(0.0, G) / lambda_melt) * dt_sec  # mm
                    melt = melt + melt_energy
                    melt = min(melt, s_surf)  # Can't melt more than is ice and snow
                    G_melt = max(0.0, melt) * lambda_melt / dt_sec
                else:
                    melt = melt + 0.0
                    G_melt = melt * lambda_melt / dt_sec

                # Calculate additional freezing in first loop only
                if g_surf > 0 and (TCs < melt_temperature):
                    freeze_energy = min(0.0, G) / lambda_melt * dt_sec
                    freeze = freeze - freeze_energy
                    freeze = min(freeze, g_surf)  # Can't freeze more than is water
                    G_freeze = freeze * lambda_melt / dt_sec
                else:
                    freeze = freeze + 0.0
                    G_freeze = freeze * lambda_melt / dt_sec

            # Diagnose surface flux with melt and freeze fluxes
            G = rad_net - H - L + H_traffic - G_melt + G_freeze

        # Update the starting temperature for sub time steps
        TCs_0 = TCs_out

        # Add the evaporations when sub time steps are used
        evap = evap_0 + evap

    # Since freezing and melting can occur due to salt or due to energy,
    # make sure only one of these occurs
    if freeze > melt:
        freeze = freeze - melt
        melt = 0.0
    else:
        melt = melt - freeze
        freeze = 0.0

    return (
        TCs_out,
        melt_temperature,
        RH_salt_final,
        RHs,
        M_road_dissolved_ratio_temp,
        evap / dt_h,  # Convert to mm/hr
        evap_pot / dt_h,  # Convert to mm/hr
        melt,
        freeze,
        H,
        L,
        G,
        long_out,
        long_net,
        rad_net,
        G_sub,
        G_freeze,
        G_melt,
    )
