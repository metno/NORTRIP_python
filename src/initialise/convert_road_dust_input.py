import constants
import numpy as np
import logging
from input_classes import (
    converted_data,
    input_traffic,
    input_meteorology,
    input_activity,
    input_airquality,
    input_data,
)

logger = logging.getLogger(__name__)


def convert_input_data_to_consolidated_structure(
    traffic_data: input_traffic,
    meteorology_data: input_meteorology,
    activity_data: input_activity,
    airquality_data: input_airquality,
    nodata: float = -99.0,
) -> converted_data:
    """
    Convert individual input data classes to consolidated data structure.

    This function replicates the MATLAB conversion logic that consolidates
    individual data arrays into a unified structure for the NORTRIP model.

    Args:
        traffic_data: input_traffic dataclass
        meteorology_data: input_meteorology dataclass
        activity_data: input_activity dataclass
        airquality_data: input_airquality dataclass
        nodata: Missing data value

    Returns:
        converted_data: Consolidated data structure
    """
    # Use traffic data to determine n_date (assuming it's the primary time series)
    n_date = traffic_data.n_traffic
    n_roads = constants.n_roads

    # Initialize the converted data structure
    converted = converted_data()
    converted.n_date = n_date
    converted.nodata = nodata

    # Initialize arrays with proper dimensions
    converted.date_data = np.full((constants.num_date_index, n_date, n_roads), nodata)
    converted.traffic_data = np.full(
        (constants.num_traffic_index, n_date, n_roads), nodata
    )
    converted.meteo_data = np.full((constants.num_meteo_index, n_date, n_roads), nodata)
    converted.activity_data = np.full(
        (constants.num_activity_index, n_date, n_roads), nodata
    )
    converted.activity_data_input = np.full(
        (constants.num_activity_index, n_date, n_roads), nodata
    )
    converted.f_conc = np.full((n_date, n_roads), nodata)
    converted.f_dis = np.full((n_date, n_roads), nodata)

    # Fill date data (using traffic data as reference)
    converted.date_data[constants.year_index, :n_date, 0] = traffic_data.year[:n_date]
    converted.date_data[constants.month_index, :n_date, 0] = traffic_data.month[:n_date]
    converted.date_data[constants.day_index, :n_date, 0] = traffic_data.day[:n_date]
    converted.date_data[constants.hour_index, :n_date, 0] = traffic_data.hour[:n_date]
    converted.date_data[constants.minute_index, :n_date, 0] = traffic_data.minute[
        :n_date
    ]
    converted.date_data[constants.datenum_index, :n_date, 0] = traffic_data.date_num[
        :n_date
    ]

    # Fill traffic data for all roads (currently only road 0)
    for ro in range(n_roads):
        converted.traffic_data[constants.N_total_index, :n_date, ro] = (
            traffic_data.N_total[:n_date]
        )
        # Fill vehicle type traffic volumes - each vehicle type separately
        for v in range(constants.num_veh):
            converted.traffic_data[constants.N_v_index[v], :n_date, ro] = (
                traffic_data.N_v[v, :n_date]
            )
            converted.traffic_data[constants.V_veh_index[v], :n_date, ro] = (
                traffic_data.V_veh[v, :n_date]
            )

        # Fill tyre-specific traffic data
        for t in range(constants.num_tyre):
            for v in range(constants.num_veh):
                tyre_veh_index = constants.N_t_v_index[(t, v)]
                converted.traffic_data[tyre_veh_index, :n_date, ro] = traffic_data.N[
                    t, v, :n_date
                ]

    # Fill meteorological data for all roads
    for ro in range(n_roads):
        # Ensure meteorology data has same length as traffic data
        meteo_n_date = min(n_date, meteorology_data.n_meteo)

        converted.meteo_data[constants.T_a_index, :meteo_n_date, ro] = (
            meteorology_data.T_a[:meteo_n_date]
        )
        converted.meteo_data[constants.T2_a_index, :meteo_n_date, ro] = (
            meteorology_data.T2_a[:meteo_n_date]
        )
        converted.meteo_data[constants.FF_index, :meteo_n_date, ro] = (
            meteorology_data.FF[:meteo_n_date]
        )
        converted.meteo_data[constants.DD_index, :meteo_n_date, ro] = (
            meteorology_data.DD[:meteo_n_date]
        )
        converted.meteo_data[constants.RH_index, :meteo_n_date, ro] = (
            meteorology_data.RH[:meteo_n_date]
        )
        converted.meteo_data[constants.Rain_precip_index, :meteo_n_date, ro] = (
            meteorology_data.Rain[:meteo_n_date]
        )
        converted.meteo_data[constants.Snow_precip_index, :meteo_n_date, ro] = (
            meteorology_data.Snow[:meteo_n_date]
        )
        converted.meteo_data[constants.short_rad_in_index, :meteo_n_date, ro] = (
            meteorology_data.short_rad_in[:meteo_n_date]
        )
        converted.meteo_data[constants.long_rad_in_index, :meteo_n_date, ro] = (
            meteorology_data.long_rad_in[:meteo_n_date]
        )
        converted.meteo_data[constants.cloud_cover_index, :meteo_n_date, ro] = (
            meteorology_data.cloud_cover[:meteo_n_date]
        )

        # Set short_rad_in_clearsky to nodata (not available in input)
        converted.meteo_data[constants.short_rad_in_clearsky_index, :n_date, ro] = (
            nodata
        )

        converted.meteo_data[
            constants.road_temperature_obs_input_index, :meteo_n_date, ro
        ] = meteorology_data.road_temperature_obs[:meteo_n_date]
        converted.meteo_data[
            constants.road_wetness_obs_input_index, :meteo_n_date, ro
        ] = meteorology_data.road_wetness_obs[:meteo_n_date]
        converted.meteo_data[constants.T_dewpoint_index, :meteo_n_date, ro] = (
            meteorology_data.T_dewpoint[:meteo_n_date]
        )
        converted.meteo_data[constants.pressure_index, :meteo_n_date, ro] = (
            meteorology_data.Pressure_a[:meteo_n_date]
        )
        converted.meteo_data[constants.T_sub_input_index, :meteo_n_date, ro] = (
            meteorology_data.T_sub[:meteo_n_date]
        )

    # Fill activity data for all roads
    for ro in range(n_roads):
        # Activity data may have different length, so pad or truncate as needed
        activity_n_date = min(n_date, len(activity_data.M_sanding))

        converted.activity_data[constants.M_sanding_index, :activity_n_date, ro] = (
            activity_data.M_sanding[:activity_n_date]
        )
        converted.activity_data[constants.t_ploughing_index, :activity_n_date, ro] = (
            activity_data.t_ploughing[:activity_n_date]
        )
        converted.activity_data[constants.t_cleaning_index, :activity_n_date, ro] = (
            activity_data.t_cleaning[:activity_n_date]
        )
        converted.activity_data[
            constants.g_road_wetting_index, :activity_n_date, ro
        ] = activity_data.g_road_wetting[:activity_n_date]
        converted.activity_data[constants.M_salting_index[0], :activity_n_date, ro] = (
            activity_data.M_salting[0, :activity_n_date]
        )
        converted.activity_data[constants.M_salting_index[1], :activity_n_date, ro] = (
            activity_data.M_salting[1, :activity_n_date]
        )
        converted.activity_data[constants.M_fugitive_index, :activity_n_date, ro] = (
            activity_data.M_fugitive[:activity_n_date]
        )

    # Copy activity_data to activity_data_input (input version before any processing)
    converted.activity_data_input = converted.activity_data.copy()

    # Fill dispersion factors for all roads
    airquality_n_date = min(n_date, airquality_data.n_date)
    for ro in range(n_roads):
        converted.f_dis[:airquality_n_date, ro] = airquality_data.f_dis_input[
            :airquality_n_date
        ]

    logger.info("Successfully converted input data to consolidated structure")

    return converted


def convert_road_dust_input(
    input_data: input_data, nodata: float = -99.0
) -> converted_data:
    """
    Converts the output from read_road_dust_input to consolidated structure.

    Args:
        input_data: input_data dataclass returned from read_road_dust_input containing:
            activity, airquality, meteorology, traffic, initial, and metadata dataclasses
        nodata: Missing data value

    Returns:
        converted_data: Consolidated data structure
    """
    return convert_input_data_to_consolidated_structure(
        input_data.traffic,
        input_data.meteorology,
        input_data.activity,
        input_data.airquality,
        nodata,
    )
