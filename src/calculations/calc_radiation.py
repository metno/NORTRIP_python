import numpy as np
import logging
from initialise import model_variables, time_config
from input_classes import (
    converted_data,
    input_metadata,
    input_initial,
    input_meteorology,
)
from config_classes import model_flags, model_parameters
from functions import (
    global_radiation_func,
    longwave_in_radiation_func,
    road_shading_func,
)
import constants

logger = logging.getLogger(__name__)


def calc_radiation(
    model_variables: model_variables,
    time_config: time_config,
    converted_data: converted_data,
    metadata: input_metadata,
    initial_data: input_initial,
    model_flags: model_flags,
    model_parameters: model_parameters,
    meteorology_data: input_meteorology,
) -> None:
    """
    Calculate the radiation and running mean temperature.

    This function replicates the MATLAB radiation calculation functionality,
    including solar radiation calculation, cloud cover estimation, longwave
    radiation calculation, and street canyon shading effects.

    Args:
        model_variables: Model variables containing all data arrays
        time_config: Time configuration with min_time and max_time
        converted_data: Consolidated input data
        metadata: Metadata containing location and other parameters
        initial_data: Initial conditions including offsets
        model_flags: Model flags including canyon settings
        model_parameters: Model parameters
    """
    logger.info("Calculating radiation and running mean temperature")

    # Set search window in hours for calculating cloud cover
    dti = 11

    # Set the short wave calculation offset
    dt_day_sw = 0  # -dt_hour/2/24 - set in metadata file

    # Extract flags and convert to boolean
    short_rad_in_available = meteorology_data.short_rad_in_available == 1
    cloud_cover_available = meteorology_data.cloud_cover_available == 1
    long_rad_in_available = meteorology_data.long_rad_in_available == 1
    canyon_shadow_flag = model_flags.canyon_shadow_flag
    canyon_long_rad_flag = model_flags.canyon_long_rad_flag

    # Initialize arrays for angles
    azimuth_ang = np.zeros(converted_data.n_date)
    zenith_ang = np.zeros(converted_data.n_date)
    shadow_fraction = np.zeros(converted_data.n_date)

    # Arrays for running means
    short_rad_net_rmean = np.zeros(converted_data.n_date)
    short_rad_net_clearsky_rmean = np.zeros(converted_data.n_date)
    f_short_rad = np.zeros(converted_data.n_date)
    long_rad_canyon = np.zeros(converted_data.n_date)

    # Loop over roads (currently only ro=0 is used)
    for ro in range(constants.n_roads):
        # Add RH and temperature offset
        for ti in range(time_config.min_time, time_config.max_time + 1):
            # Apply RH offset and clamp to 0-100%
            rh_value = (
                converted_data.meteo_data[constants.RH_index, ti, ro]
                + initial_data.RH_offset
            )
            converted_data.meteo_data[constants.RH_index, ti, ro] = max(
                0, min(100, rh_value)
            )

            # Apply temperature offset
            converted_data.meteo_data[constants.T_a_index, ti, ro] += (
                initial_data.T_2m_offset
            )

        # Set initial cloud cover to default value if no data available
        cloud_cover_default = 0.5
        if not (meteorology_data.cloud_cover_available == 1):
            for ti in range(time_config.min_time, time_config.max_time + 1):
                converted_data.meteo_data[constants.cloud_cover_index, ti, ro] = (
                    cloud_cover_default
                )

        # Calculate short wave net radiation when global radiation is available
        for ti in range(time_config.min_time, time_config.max_time + 1):
            if meteorology_data.short_rad_in_available == 1:
                short_rad_net_value = converted_data.meteo_data[
                    constants.short_rad_in_index, ti, ro
                ] * (1 - metadata.albedo_road)
                for tr in range(model_parameters.num_track):
                    model_variables.road_meteo_data[
                        constants.short_rad_net_index, ti, tr, ro
                    ] = short_rad_net_value

            # Calculate short wave net radiation when global radiation is not available
            if not (meteorology_data.short_rad_in_available == 1):
                datenum_value = (
                    converted_data.date_data[constants.datenum_index, ti, 0] + dt_day_sw
                )
                cloud_cover_value = converted_data.meteo_data[
                    constants.cloud_cover_index, ti, ro
                ]

                short_rad_net_temp, azimuth_ang[ti], zenith_ang[ti] = (
                    global_radiation_func(
                        metadata.LAT,
                        metadata.LON,
                        datenum_value,
                        metadata.DIFUTC_H,
                        metadata.Z_SURF,
                        cloud_cover_value,
                        metadata.albedo_road,
                    )
                )

                for tr in range(model_parameters.num_track):
                    model_variables.road_meteo_data[
                        constants.short_rad_net_index, ti, tr, ro
                    ] = short_rad_net_temp

            # Calculate clear sky short radiation
            datenum_value = (
                converted_data.date_data[constants.datenum_index, ti, 0] + dt_day_sw
            )

            short_rad_clearsky, azimuth_ang[ti], zenith_ang[ti] = global_radiation_func(
                metadata.LAT,
                metadata.LON,
                datenum_value,
                metadata.DIFUTC_H,
                metadata.Z_SURF,
                0.0,
                0.0,
            )
            converted_data.meteo_data[constants.short_rad_in_clearsky_index, ti, ro] = (
                short_rad_clearsky
            )

            short_rad_net_clearsky, _, _ = global_radiation_func(
                metadata.LAT,
                metadata.LON,
                datenum_value,
                metadata.DIFUTC_H,
                metadata.Z_SURF,
                0.0,
                metadata.albedo_road,
            )
            for tr in range(model_parameters.num_track):
                model_variables.road_meteo_data[
                    constants.short_rad_net_clearsky_index, ti, tr, ro
                ] = short_rad_net_clearsky

        # Calculate cloud cover when cloud cover is not available and global is available
        if not cloud_cover_available and short_rad_in_available:
            # Calculate running means to calculate cloud cover per hour
            for ti in range(time_config.min_time, time_config.max_time + 1):
                tr = 0  # Use first track for cloud cover calculation
                ti1 = max(ti - dti, time_config.min_time)
                ti2 = min(ti + dti, time_config.max_time)
                ti_num = ti2 - ti1 + 1

                short_rad_net_rmean[ti] = 0.0
                short_rad_net_clearsky_rmean[ti] = 0.0

                for tt in range(ti1, ti2 + 1):
                    short_rad_net_rmean[ti] += (
                        model_variables.road_meteo_data[
                            constants.short_rad_net_index, tt, tr, ro
                        ]
                        / ti_num
                    )
                    short_rad_net_clearsky_rmean[ti] += (
                        model_variables.road_meteo_data[
                            constants.short_rad_net_clearsky_index, tt, tr, ro
                        ]
                        / ti_num
                    )

                # Calculate cloud cover fraction
                if short_rad_net_clearsky_rmean[ti] > 0:
                    f_short_rad[ti] = (
                        short_rad_net_rmean[ti] / short_rad_net_clearsky_rmean[ti]
                    )
                else:
                    f_short_rad[ti] = 0.0

                f_short_rad[ti] = max(0.0, min(1.0, f_short_rad[ti]))
                converted_data.meteo_data[constants.cloud_cover_index, ti, ro] = min(
                    1.0, (1 - f_short_rad[ti]) / 0.9
                )

        elif not cloud_cover_available:
            for ti in range(time_config.min_time, time_config.max_time + 1):
                converted_data.meteo_data[constants.cloud_cover_index, ti, ro] = (
                    cloud_cover_default
                )

        # Calculate incoming long wave radiation
        if not long_rad_in_available:
            for ti in range(time_config.min_time, time_config.max_time + 1):
                T_air = converted_data.meteo_data[constants.T_a_index, ti, ro]
                RH = converted_data.meteo_data[constants.RH_index, ti, ro]
                cloud_cover_val = converted_data.meteo_data[
                    constants.cloud_cover_index, ti, ro
                ]

                longwave_in = longwave_in_radiation_func(
                    T_air, RH, cloud_cover_val, metadata.Pressure
                )
                converted_data.meteo_data[constants.long_rad_in_index, ti, ro] = (
                    longwave_in + initial_data.long_rad_in_offset
                )

        # Calculate the shadow fraction
        if canyon_shadow_flag:
            tau_cs_diffuse = 0.2
            h_canyon_temp = [
                max(0.001, h) for h in metadata.h_canyon
            ]  # Avoid division by 0

            for ti in range(time_config.min_time, time_config.max_time + 1):
                shadow_fraction[ti] = road_shading_func(
                    azimuth_ang[ti],
                    zenith_ang[ti],
                    metadata.ang_road,
                    metadata.b_road,
                    metadata.b_canyon,
                    h_canyon_temp,
                )

                cloud_cover_val = converted_data.meteo_data[
                    constants.cloud_cover_index, ti, ro
                ]
                tau_diffuse = tau_cs_diffuse + cloud_cover_val * (1 - tau_cs_diffuse)

                for tr in range(model_parameters.num_track):
                    current_rad = model_variables.road_meteo_data[
                        constants.short_rad_net_index, ti, tr, ro
                    ]
                    short_rad_direct = (
                        current_rad * (1 - tau_diffuse) * (1 - shadow_fraction[ti])
                    )
                    short_rad_diffuse = current_rad * tau_diffuse

                    model_variables.road_meteo_data[
                        constants.short_rad_net_index, ti, tr, ro
                    ] = short_rad_direct + short_rad_diffuse

        # Canyon building facade contribution to longwave radiation
        if canyon_long_rad_flag:
            # This is based on the integral of a cylinder of height h_canyon
            h_canyon_temp = max(
                0.001, float(np.mean(metadata.h_canyon))
            )  # Avoid division by 0
            theta = np.arctan(h_canyon_temp * 2 / metadata.b_canyon)
            canyon_fraction = (
                1 - np.cos(2 * theta / 2)
            ) / 2  # factor 2 for theta to get an average

            sigma = 5.67e-8
            T0C = 273.15

            for ti in range(time_config.min_time, time_config.max_time + 1):
                T_air = converted_data.meteo_data[constants.T_a_index, ti, ro]
                long_rad_canyon[ti] = sigma * (T0C + T_air) ** 4

                # Mix sky and canyon longwave radiation
                sky_fraction = 1 - canyon_fraction
                current_longwave = converted_data.meteo_data[
                    constants.long_rad_in_index, ti, ro
                ]
                converted_data.meteo_data[constants.long_rad_in_index, ti, ro] = (
                    current_longwave * sky_fraction
                    + long_rad_canyon[ti] * canyon_fraction
                )

    logger.info("Radiation calculation completed")
