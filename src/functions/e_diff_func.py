from .energy_correction_func import energy_correction_func
import numpy as np

def e_diff_func(
    T_obs: float,
    TCs_0: float,
    TC: float,
    TCsub: float,
    E_correction_old: float,
    P: float,
    dzs_in: float,
    dt_h_in: float,
    r_aero: float,
    short_net: float,
    long_in: float,
    H_traffic: float,
    L: float,
    G_freeze: float,
    G_melt: float,
    sub_surf_param: np.ndarray,
    use_subsurface_flag: int,
) -> tuple[float, float, float]:
    """
    Calculate energy difference for surface energy balance.

    Args:
        T_obs: Observed temperature
        TCs_0: Initial surface temperature
        TC: Air temperature
        TCsub: Subsurface temperature
        E_correction_old: Previous energy correction
        P: Pressure
        dzs_in: Subsurface depth
        dt_h_in: Time step in hours
        r_aero: Aerodynamic resistance
        short_net: Net shortwave radiation
        long_in: Incoming longwave radiation
        H_traffic: Traffic heat flux
        L: Latent heat flux
        G_freeze: Freezing heat flux
        G_melt: Melting heat flux
        sub_surf_param: Subsurface parameters [rho_s, c_s, k_s_road]
        use_subsurface_flag: Flag for using subsurface calculations

    Returns:
        tuple: (E_diff, E_correction_new, T_new) where:
            E_diff: Energy difference
            E_correction_new: New energy correction
            T_new: New surface temperature
    """
    # Set constants
    Cp = 1006.0
    lambda_val = 2.50e6
    lambda_ice = 2.83e6
    lambda_melt = 3.33e5  # (J/kg)
    RD = 287.0
    T0C = 273.15
    sigma = 5.67e-8
    eps_s = 0.95

    # Set time step in seconds
    dt_sec = dt_h_in * 3600

    # Set subsurface parameters
    rho_s = sub_surf_param[0]
    c_s = sub_surf_param[1]
    k_s_road = sub_surf_param[2]
    C_s = rho_s * c_s
    omega = 7.3e-5

    # Automatically set dzs if it is 0.
    # This calculated value of dzs is optimal for a sinusoidal varying flux
    if dzs_in == 0:
        dzs = (k_s_road / C_s / 2 / omega) ** 0.5
    else:
        dzs = dzs_in

    mu = omega * C_s * dzs

    # If subsurface flux is turned off
    if not use_subsurface_flag:
        mu = 0

    # Set atmospheric temperature in Kelvin
    TK_a = T0C + TC

    # Set air density
    rho = P * 100.0 / (RD * TK_a)

    # Present values of constants for implicit solution
    a_G = 1 / (C_s * dzs)
    a_H = rho * Cp / r_aero
    a_RL = (1 - 4 * TC / TK_a) * eps_s * sigma * TK_a**4
    b_RL = 4 * eps_s * sigma * TK_a**3
    a_rad = short_net + long_in + H_traffic

    # Note: these might need to change if we make changes in the double loop in
    # surface_energy_model_4_func.m
    G_freeze = 0
    G_melt = 0

    # Calculate the energy difference needed to make T_mod = T_obs
    E_diff = (
        (T_obs * (1 + dt_sec * a_G * (a_H + b_RL + mu)) - TCs_0) / (dt_sec * a_G)
        - a_rad
        + a_RL
        + L
        - a_H * TC
        - mu * TCsub
        + G_melt
        - G_freeze
    )  # =E_diff + E_correction_old

    E_correction_new = energy_correction_func(E_diff, E_correction_old)

    # Use the new E_correction to determine surface temperature. If f = 1 in
    # Energy_correction_func, T_new = T_obs
    T_new = (
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
            + E_correction_new
        )
    ) / (1 + dt_sec * a_G * (a_H + b_RL + mu))

    return E_diff, E_correction_new, T_new
