import numpy as np


def road_shading_func(
    azimuth: float,
    zenith: float,
    ang_road: float,
    b_road: float,
    b_canyon: float,
    h_canyon: list,
) -> float:
    """
    Calculate shadow fraction on road surface due to street canyon.

    This function calculates the shadow fraction based on the geometric
    relationship between solar position, street canyon geometry, and road orientation.

    Args:
        azimuth: Solar azimuth angle in degrees
        zenith: Solar zenith angle in degrees
        ang_road: Road orientation angle in degrees
        b_road: Road width in meters
        b_canyon: Canyon width in meters
        h_canyon: Canyon heights [north, south] in meters

    Returns:
        float: Shadow fraction (0-1)
    """
    # Normalize road angle to 0-180 range
    if ang_road > 180:
        ang_road = ang_road - 180

    # Calculate angle difference between sun azimuth and road orientation
    ang_dif = azimuth - ang_road

    if ang_dif == 360:
        ang_dif = 0

    # Determine which canyon wall height to use based on angle difference
    if ang_dif <= -180:
        h_canyon_temp = h_canyon[1]  # south wall
        ang_dif = ang_dif + 360
    elif ang_dif < 0:
        h_canyon_temp = h_canyon[0]  # north wall
        ang_dif = ang_dif + 180
    elif ang_dif >= 180:
        h_canyon_temp = h_canyon[0]  # north wall
        ang_dif = ang_dif - 180
    else:
        h_canyon_temp = h_canyon[1]  # south wall

    # Calculate shadow fraction
    if zenith >= 90:
        shadow_fraction = 1.0
    elif ang_dif == 0:
        shadow_fraction = 0.0
    else:
        # Calculate shadow distance projected on the ground
        d_shadow = h_canyon_temp * np.tan(np.radians(zenith))

        # Calculate kerb width (sidewalk/shoulder area)
        b_kerb = max(0, (b_canyon - b_road) / 2)

        # Project kerb and road widths considering angle
        b1_kerb = b_kerb / np.sin(np.radians(ang_dif))
        b1_road = b_road / np.sin(np.radians(ang_dif))

        # Calculate shadow fraction on road surface
        shadow_fraction = max(0, (d_shadow - b1_kerb) / b1_road)
        shadow_fraction = min(1, shadow_fraction)

    return shadow_fraction
