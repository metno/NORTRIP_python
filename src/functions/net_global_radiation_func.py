import numpy as np
from datetime import datetime
from .datenum_to_datetime import datenum_to_datetime


def net_global_radiation_func(
    LAT: float,
    LON: float,
    date_num: np.float64,
    DIFUTC_H: float,
    Z_SURF: float,
    N_CLOUD: float,
    Z_CLOUD: float,
    ALBEDO: float,
    TC: float,
) -> tuple[float, float]:
    """
    Calculate net global radiation and longwave radiation.

    Args:
        LAT: Latitude in degrees
        LON: Longitude in degrees
        date_num: Unix timestamp (seconds since epoch)
        DIFUTC_H: Time difference from UTC in hours
        Z_SURF: Surface elevation in meters
        N_CLOUD: Cloud cover fraction (0-1)
        Z_CLOUD: Cloud height in meters
        ALBEDO: Surface albedo (0-1)
        TC: Air temperature in Celsius

    Returns:
        tuple: (SOLAR_NET, R_NET) where:
            SOLAR_NET: Net solar radiation
            R_NET: Net longwave radiation
    """
    # Constants
    SECPHOUR = 3600.0
    SECPDAY = 86400.0
    PI = np.pi / 180.0
    S0 = 1367.0  # Solar constant
    T0C = 273.15
    SIGMA = 5.67e-8

    # Convert MATLAB datenum to Python datetime
    current_datetime = datenum_to_datetime(date_num)

    Y = current_datetime.year
    M = current_datetime.month
    D = current_datetime.day
    _ = current_datetime.hour
    _ = current_datetime.minute
    _ = current_datetime.second + current_datetime.microsecond / 1e6

    # Calculate Julian day
    year_start = datetime(Y, 1, 1)
    JULIAN_DAY = (current_datetime - year_start).days + 1

    # Calculate time in seconds from start of day
    day_start = datetime(Y, M, D, 0, 0, 0)
    TIME_S = (current_datetime - day_start).total_seconds()

    # Solar calculations
    DAYANG = 360.0 / 365.0 * (JULIAN_DAY - 1.0)
    DEC = 0.396 - 22.91 * np.cos(PI * DAYANG) + 4.025 * np.sin(PI * DAYANG)
    EQTIME = (
        1.03
        + 25.7 * np.cos(PI * DAYANG)
        - 440.0 * np.sin(PI * DAYANG)
        - 201.0 * np.cos(2.0 * PI * DAYANG)
        - 562.0 * np.sin(2.0 * PI * DAYANG)
    ) / SECPHOUR

    SOLARTIME = (
        TIME_S + SECPDAY + SECPHOUR * (LON / 15.0 + DIFUTC_H + EQTIME)
    ) % SECPDAY
    HOURANG = 15.0 * (12.0 - SOLARTIME / SECPHOUR)

    # Set azimuth angle for atmospheric corrections
    AZT = np.sin(PI * DEC) * np.sin(PI * LAT) + np.cos(PI * DEC) * np.cos(
        PI * LAT
    ) * np.cos(PI * HOURANG)

    if abs(AZT) < 1:
        AZ = np.arccos(AZT) / PI
    else:
        AZ = 0.0

    # Corrections for atmosphere and cloud from Oerlemans (Greenland)
    # These need to be updated
    TAU_A = (0.75 + 6.8e-5 * Z_SURF - 7.1e-9 * Z_SURF**2) * (1 - 0.001 * AZ)
    TAU_C = 1 - 0.78 * N_CLOUD**2 * np.exp(-8.5e-4 * Z_SURF)

    # Set day beginning and end
    if abs(np.tan(PI * DEC) * np.tan(PI * LAT)) < 1:
        DAY_BIG = (
            12.0 - np.arccos(-np.tan(PI * DEC) * np.tan(PI * LAT)) / PI / 15.0
        ) * SECPHOUR
        DAY_END = (
            12.0 + np.arccos(-np.tan(PI * DEC) * np.tan(PI * LAT)) / PI / 15.0
        ) * SECPHOUR
    else:
        DAY_BIG = 0.0
        DAY_END = 24.0 * SECPHOUR

    # Determine solar radiation at surface during day
    if (SOLARTIME > DAY_BIG) and (SOLARTIME < DAY_END):
        SOLAR_IN = S0 * TAU_A * TAU_C * np.cos(AZ * PI)
    else:
        SOLAR_IN = 0.0

    SOLAR_NET = SOLAR_IN * (1 - ALBEDO)

    # Net long wave radiation
    R_NET = -SIGMA * (T0C + TC) ** 4 * (1 - 0.94e-5 * (T0C + TC) ** 2)  # GARRETT
    R_NET = (
        R_NET + 0.3 * 1.0 * SIGMA * N_CLOUD * (T0C + TC - Z_CLOUD * 0.006) ** 4
    )  # GARRETT

    return SOLAR_NET, R_NET
