"""
Main script for the NORTRIP Road Dust Model in Python.
"""

from importlib.metadata import version
import constants
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
)
import logging
from model_args import create_arg_parser
from fortran import NORTRIP_fortran_control


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

    paths = read_road_dust_paths(read_as_text=read_as_text)

    model_parameters, model_flags, model_activities = read_road_dust_parameters(
        paths.path_filename_inputparam, read_as_text=read_as_text
    )

    input_data = read_road_dust_input(
        paths.path_filename_inputdata,
        model_parameters,
        read_as_text,
        print_results,
    )

    (
        activity_input,
        airquality_input,
        meteorology_input,
        traffic_input,
        initial_input,
        metadata_input,
    ) = input_data

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
        # Does nothing for now
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

        for tf in range(time_config.min_time, time_config.max_time):
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
                        )

                        # Calculate road emissions and dust loading
                        # road_dust_emission_model_v2 - TODO: Implement this function

    logger.info("End of NORTRIP_Control")


if __name__ == "__main__":
    main()
