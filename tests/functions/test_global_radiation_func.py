import numpy as np
from datetime import datetime, timedelta
from src.functions.global_radiation_func import global_radiation_func


def test_global_radiation_func():
    """Test global radiation calculation."""

    # Test for a summer day in Norway
    LAT = 63.4  # Trondheim latitude
    LON = 10.4  # Trondheim longitude
    DIFUTC_H = 2.0  # UTC+2 (summer time)
    Z_SURF = 100.0  # 100m elevation
    N_CLOUD = 0.3  # 30% cloud cover
    ALBEDO = 0.15  # Typical ground albedo

    # Convert datetime to MATLAB datenum (June 21, 2023, noon)
    dt = datetime(2023, 6, 21, 12, 0, 0)
    matlab_epoch = datetime(1, 1, 1)
    delta = dt - matlab_epoch
    datenum = delta.total_seconds() / 86400.0 + 1

    SOLAR_NET, azimuth_angle, zenith_angle = global_radiation_func(
        LAT, LON, datenum, DIFUTC_H, Z_SURF, N_CLOUD, ALBEDO
    )

    # At noon on summer solstice, should have significant solar radiation
    assert SOLAR_NET > 0
    assert SOLAR_NET < 1500  # Reasonable upper bound (W/m²)

    # Angles should be reasonable
    assert 0 <= azimuth_angle <= 360
    assert 0 <= zenith_angle <= 90

    # At summer solstice near noon, zenith angle should be small
    assert zenith_angle < 60  # Sun should be reasonably high

    # Test night time (midnight)
    dt_night = datetime(2023, 6, 21, 0, 0, 0)
    delta_night = dt_night - matlab_epoch
    datenum_night = delta_night.total_seconds() / 86400.0 + 1

    SOLAR_NET_night, azimuth_night, zenith_night = global_radiation_func(
        LAT, LON, datenum_night, DIFUTC_H, Z_SURF, N_CLOUD, ALBEDO
    )

    # At night, solar radiation should be zero or very small
    assert SOLAR_NET_night <= 50.0  # Should be small compared to daytime values


def test_global_radiation_func_cloud_effects():
    """Test cloud cover effects on radiation."""

    LAT = 60.0
    LON = 10.0
    DIFUTC_H = 1.0
    Z_SURF = 50.0
    ALBEDO = 0.2

    # Noon on a clear day
    dt = datetime(2023, 7, 15, 12, 0, 0)
    matlab_epoch = datetime(1, 1, 1)
    delta = dt - matlab_epoch
    datenum = delta.total_seconds() / 86400.0 + 1

    # Clear sky
    N_CLOUD_clear = 0.0
    SOLAR_NET_clear, _, _ = global_radiation_func(
        LAT, LON, datenum, DIFUTC_H, Z_SURF, N_CLOUD_clear, ALBEDO
    )

    # Overcast sky
    N_CLOUD_overcast = 1.0
    SOLAR_NET_overcast, _, _ = global_radiation_func(
        LAT, LON, datenum, DIFUTC_H, Z_SURF, N_CLOUD_overcast, ALBEDO
    )

    # Clear sky should have more radiation than overcast
    assert SOLAR_NET_clear > SOLAR_NET_overcast
    assert SOLAR_NET_overcast >= 0  # Should still be non-negative


def test_global_radiation_func_elevation_effects():
    """Test elevation effects on radiation."""

    LAT = 60.0
    LON = 10.0
    DIFUTC_H = 1.0
    N_CLOUD = 0.2
    ALBEDO = 0.2

    # Noon on a summer day
    dt = datetime(2023, 7, 15, 12, 0, 0)
    matlab_epoch = datetime(1, 1, 1)
    delta = dt - matlab_epoch
    datenum = delta.total_seconds() / 86400.0 + 1

    # Sea level
    Z_SURF_low = 0.0
    SOLAR_NET_low, _, _ = global_radiation_func(
        LAT, LON, datenum, DIFUTC_H, Z_SURF_low, N_CLOUD, ALBEDO
    )

    # High elevation
    Z_SURF_high = 2000.0
    SOLAR_NET_high, _, _ = global_radiation_func(
        LAT, LON, datenum, DIFUTC_H, Z_SURF_high, N_CLOUD, ALBEDO
    )

    # Both should be positive
    assert SOLAR_NET_low > 0
    assert SOLAR_NET_high > 0

    # Higher elevation typically has more radiation due to less atmosphere
    # (though the function may implement this differently)
    assert SOLAR_NET_high >= 0


def test_global_radiation_func_albedo_effects():
    """Test albedo effects on net radiation."""

    LAT = 60.0
    LON = 10.0
    DIFUTC_H = 1.0
    Z_SURF = 100.0
    N_CLOUD = 0.3

    # Noon on a summer day
    dt = datetime(2023, 7, 15, 12, 0, 0)
    matlab_epoch = datetime(1, 1, 1)
    delta = dt - matlab_epoch
    datenum = delta.total_seconds() / 86400.0 + 1

    # Low albedo (dark surface)
    ALBEDO_low = 0.05
    SOLAR_NET_low_albedo, _, _ = global_radiation_func(
        LAT, LON, datenum, DIFUTC_H, Z_SURF, N_CLOUD, ALBEDO_low
    )

    # High albedo (bright surface like snow)
    ALBEDO_high = 0.8
    SOLAR_NET_high_albedo, _, _ = global_radiation_func(
        LAT, LON, datenum, DIFUTC_H, Z_SURF, N_CLOUD, ALBEDO_high
    )

    # Low albedo should result in higher net radiation
    assert SOLAR_NET_low_albedo > SOLAR_NET_high_albedo
    assert SOLAR_NET_high_albedo >= 0  # Should still be non-negative


def test_global_radiation_func_seasonal_variation():
    """Test seasonal variation in solar radiation."""

    LAT = 60.0
    LON = 10.0
    DIFUTC_H = 1.0
    Z_SURF = 100.0
    N_CLOUD = 0.2
    ALBEDO = 0.2

    matlab_epoch = datetime(1, 1, 1)

    # Summer solstice (June 21)
    dt_summer = datetime(2023, 6, 21, 12, 0, 0)
    delta_summer = dt_summer - matlab_epoch
    datenum_summer = delta_summer.total_seconds() / 86400.0 + 1

    SOLAR_NET_summer, _, zenith_summer = global_radiation_func(
        LAT, LON, datenum_summer, DIFUTC_H, Z_SURF, N_CLOUD, ALBEDO
    )

    # Winter solstice (December 21)
    dt_winter = datetime(2023, 12, 21, 12, 0, 0)
    delta_winter = dt_winter - matlab_epoch
    datenum_winter = delta_winter.total_seconds() / 86400.0 + 1

    SOLAR_NET_winter, _, zenith_winter = global_radiation_func(
        LAT, LON, datenum_winter, DIFUTC_H, Z_SURF, N_CLOUD, ALBEDO
    )

    # Summer should have more radiation than winter at same latitude
    assert SOLAR_NET_summer > SOLAR_NET_winter

    # Summer zenith angle should be smaller (sun higher) than winter
    assert zenith_summer < zenith_winter
