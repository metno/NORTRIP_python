"""
Main script for the NORTRIP Road Dust Model in Python.
"""

from importlib.metadata import version
from ospm import OSPM_Main
import constants
import numpy as np
from functions import running_mean_temperature_func
from read_files import (
    read_road_dust_parameters,
    read_road_dust_paths,
    read_road_dust_input,
)
from initialise import (
    road_dust_initialise_time,
    road_dust_initialise_variables,
    convert_road_dust_input,
)
from calculations import (
    calc_radiation,
    road_dust_surface_wetness,
    set_activity_data,
    activity_state,
    road_dust_emission_model,
    road_dust_dispersion,
    road_dust_concentrations,
    road_dust_convert_variables,
)
import logging
from model_args import create_arg_parser
from fortran import NORTRIP_fortran_control
from plots import plot_road_dust_result


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def main():
    args = create_arg_parser().parse_args()
    read_as_text = bool(args.text)
    print_results = bool(args.print)
    use_fortran = bool(args.fortran)

    print("-" * 33)
    print(f"Starting NORTRIP_python_v{version('nortrip-python')}...")
    print("-" * 33)

    logger.info(f"Read as inputs as text: {read_as_text}")
    logger.info(f"Print results to terminal: {print_results}")
    logger.info(f"Run fortran model: {use_fortran}")

    # args.paths is now positional and points to the model paths Excel file
    paths = read_road_dust_paths(read_as_text=read_as_text, paths_xlsx=args.paths)

    model_parameters, model_flags, model_activities = read_road_dust_parameters(
        paths.path_filename_inputparam, read_as_text=read_as_text
    )

    input_data = read_road_dust_input(
        paths.path_filename_inputdata,
        model_parameters,
        read_as_text,
        print_results,
    )

    activity_input = input_data.activity
    airquality_input = input_data.airquality
    meteorology_input = input_data.meteorology
    traffic_input = input_data.traffic
    initial_input = input_data.initial
    metadata_input = input_data.metadata

    converted_data = convert_road_dust_input(input_data, nodata=metadata_input.nodata)

    time_config = road_dust_initialise_time(
        converted_data=converted_data,
        metadata=metadata_input,
        use_fortran_flag=use_fortran,
    )

    if time_config.time_bad:
        logger.error("Time configuration failed - stopping execution")
        return

    # Initialize model variables
    model_variables = road_dust_initialise_variables(
        time_config=time_config,
        converted_data=converted_data,
        initial_data=initial_input,
        metadata=metadata_input,
        airquality_data=airquality_input,
        model_parameters=model_parameters,
        model_flags=model_flags,
    )

    if use_fortran:
        NORTRIP_fortran_control()

    # Initialize activity state for tracking maintenance activities
    state = activity_state()

    # Main model loop

    for ro in range(constants.n_roads):
        calc_radiation(
            model_variables=model_variables,
            time_config=time_config,
            converted_data=converted_data,
            metadata=metadata_input,
            initial_data=initial_input,
            model_flags=model_flags,
            model_parameters=model_parameters,
            meteorology_data=meteorology_input,
        )

        for tr in range(model_parameters.num_track):
            # Set the road meteo data
            if meteorology_input.T_sub_available:
                model_variables.road_meteo_data[constants.T_sub_index, :, tr, ro] = (
                    converted_data.meteo_data[constants.T_sub_input_index, :, ro]
                )
            else:
                model_variables.road_meteo_data[constants.T_sub_index, :, tr, ro] = (
                    running_mean_temperature_func(
                        converted_data.meteo_data[constants.T_a_index, :, ro],
                        model_parameters.sub_surf_average_time,
                        0,
                        time_config.max_time_inputdata,
                        time_config.dt,
                    )
                )

        logger.info("Starting time loop...")

        for tf in range(time_config.min_time, time_config.max_time + 1):
            if model_flags.forecast_hour == 0:
                forecast_index = 0
            else:
                forecast_index = max(
                    0, round(model_flags.forecast_hour / time_config.dt - 1)
                )

            # Print the date
            if converted_data.date_data[constants.hour_index, tf, ro] == 1:
                full_date_str = traffic_input.date_str[1, tf]
                date_str = full_date_str[6:12].strip()

                if forecast_index > 0:
                    logger.info(f"{date_str} F: {model_flags.forecast_hour}")

                else:
                    logger.info(date_str)

            # Forecast loop. This is not a loop if forecast_hour=0 or 1
            for ti in range(tf, tf + forecast_index + 1):
                if ti <= time_config.max_time:
                    # Use road maintenance activity rules to determine activities
                    set_activity_data(
                        ti=ti,
                        ro=ro,
                        time_config=time_config,
                        converted_data=converted_data,
                        model_variables=model_variables,
                        model_flags=model_flags,
                        model_activities=model_activities,
                        model_parameters=model_parameters,
                        state=state,
                    )

                    # Loop through the tracks. Future development since num_track=1
                    for tr in range(model_parameters.num_track):
                        # Calculate road surface conditions
                        road_dust_surface_wetness(
                            ti=ti,
                            tr=tr,
                            ro=ro,
                            time_config=time_config,
                            converted_data=converted_data,
                            model_variables=model_variables,
                            model_parameters=model_parameters,
                            model_flags=model_flags,
                            metadata=metadata_input,
                            input_activity=activity_input,
                            tf=tf,
                            meteorology_input=meteorology_input,
                        )

                        # Calculate road emissions and dust loading
                        road_dust_emission_model(
                            ti=ti,
                            tr=tr,
                            ro=ro,
                            time_config=time_config,
                            converted_data=converted_data,
                            model_variables=model_variables,
                            model_parameters=model_parameters,
                            model_flags=model_flags,
                            metadata=metadata_input,
                            initial_data=initial_input,
                            airquality_data=airquality_input,
                        )

            # Save the forecast surface temperature into the forecast index
            if (
                model_flags.forecast_type_flag > 0
                and tf > time_config.min_time
                and tf + forecast_index <= time_config.max_time
            ):
                forecast_ti = tf + forecast_index
                # Use the first track (tr=0) since num_track is typically 1
                tr_forecast = 0

                # Modelled forecast
                if model_flags.forecast_type_flag == 1:
                    model_variables.forecast_T_s[forecast_ti] = (
                        model_variables.road_meteo_data[
                            constants.T_s_index, forecast_ti, tr_forecast, ro
                        ]
                    )

                # Persistence forecast
                elif model_flags.forecast_type_flag == 2:
                    model_variables.forecast_T_s[forecast_ti] = (
                        model_variables.road_meteo_data[
                            constants.T_s_index, tf - 1, tr_forecast, ro
                        ]
                    )

                # Bias correction forecast. Set use_observed_temperature_init_flag=0
                elif model_flags.forecast_type_flag == 3:
                    model_variables.forecast_T_s[forecast_ti] = (
                        model_variables.road_meteo_data[
                            constants.T_s_index, forecast_ti, tr_forecast, ro
                        ]
                        - model_variables.original_bias_T_s
                    )

                # Linear extrapolation. Set use_observed_temperature_init_flag=0
                elif (
                    model_flags.forecast_type_flag == 4
                    and tf - 1 > time_config.min_time
                ):
                    # Get datenum values for interpolation
                    x_dates = converted_data.date_data[
                        constants.datenum_index, tf - 2 : tf, ro
                    ]
                    y_temps = model_variables.road_meteo_data[
                        constants.T_s_index, tf - 2 : tf, tr_forecast, ro
                    ]
                    target_date = converted_data.date_data[
                        constants.datenum_index, forecast_ti, ro
                    ]

                    # Linear extrapolation using numpy.interp
                    model_variables.forecast_T_s[forecast_ti] = np.interp(
                        target_date, x_dates, y_temps
                    )

            # Redistribute mass and moisture between tracks.
            # Not implemented yet

        # Put forecast surface temperature into the normal road temperature
        if model_flags.forecast_hour > 0:
            model_variables.road_meteo_data[
                constants.T_s_index, time_config.min_time : time_config.max_time, tr, ro
            ] = model_variables.forecast_T_s[
                time_config.min_time : time_config.max_time
            ]

        # Calculate dispersion factors using ospm or NOx
        if model_flags.use_ospm_flag:
            OSPM_Main()
        else:
            road_dust_dispersion(
                time_config=time_config,
                converted_data=converted_data,
                model_variables=model_variables,
                model_parameters=model_parameters,
                metadata=metadata_input,
                airquality_data=airquality_input,
                ro=ro,
            )

        # Calculate concentrations
        road_dust_concentrations(
            time_config=time_config,
            model_variables=model_variables,
            metadata=metadata_input,
            ro=ro,
        )

        # Put binned balance data into normal arrays
        road_dust_convert_variables(
            model_variables=model_variables,
            metadata=metadata_input,
            ro=ro,
        )

    time_config.min_time = time_config.min_time_save
    time_config.max_time = time_config.max_time_save

    # Generate plots
    try:
        plot_road_dust_result(
            time_config=time_config,
            converted_data=converted_data,
            initial_data=initial_input,
            metadata=metadata_input,
            airquality_data=airquality_input,
            model_parameters=model_parameters,
            model_flags=model_flags,
            model_variables=model_variables,
            meteo_input=meteorology_input,
            paths=paths,
            ro=0,
        )
    except Exception as e:
        logger.exception(f"Plotting failed: {e}")

    logger.info("End of NORTRIP_Control")


if __name__ == "__main__":
    main()
