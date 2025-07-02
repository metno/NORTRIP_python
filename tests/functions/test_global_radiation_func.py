from functions import global_radiation_func


def test_global_radiation_func():
    """Test the global radiation function."""
    LAT = 60.0
    LON = 10.0
    datenum = 738611.5  # Example datenum
    DIFUTC_H = 1.0
    Z_SURF = 100.0
    N_CLOUD = 0.5
    ALBEDO = 0.1

    short_rad_net, azimuth, zenith = global_radiation_func(
        LAT, LON, datenum, DIFUTC_H, Z_SURF, N_CLOUD, ALBEDO
    )

    assert isinstance(short_rad_net, float)
    assert isinstance(azimuth, float)
    assert isinstance(zenith, float)
    assert short_rad_net >= 0
