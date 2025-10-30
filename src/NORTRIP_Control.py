"""
Main script for the NORTRIP Road Dust Model in Python.
"""

from importlib.metadata import version
import os
from read_files import (
    read_road_dust_parameters,
    read_road_dust_paths,
    read_road_dust_input,
)
from initialise import (
    road_dust_initialise_time,
    convert_road_dust_input,
)
import logging
from model_args import create_arg_parser
from fortran import NORTRIP_fortran_control
from plots import plot_road_dust_result
from output import save_road_dust_results_average
from main_nortrip_loop import main_nortrip_loop

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def main():
    """
    Main function for the NORTRIP Road Dust Model.
    """
    args = create_arg_parser().parse_args()
    print_results = bool(args.print)
    use_fortran = bool(args.fortran)
    use_logging = bool(args.log)
    plot_figure = args.plot

    if not use_logging:
        logging.disable(25)

    print("-" * 33)
    print(f"Starting NORTRIP_python_v{version('nortrip-python')}...")
    print("-" * 33)

    print(f"Print results to terminal: {print_results}")
    print(f"Run fortran model: {use_fortran}")

    paths = read_road_dust_paths(paths_path=args.paths)

    model_parameters, model_flags, model_activities, parameter_sheets = (
        read_road_dust_parameters(paths.path_filename_inputparam)
    )

    input_data = read_road_dust_input(paths.path_filename_inputdata, model_parameters)

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

    if use_fortran:
        NORTRIP_fortran_control()

    # Main model loop extracted to function
    model_variables = main_nortrip_loop(
        time_config=time_config,
        converted_data=converted_data,
        metadata_input=metadata_input,
        initial_input=initial_input,
        model_flags=model_flags,
        model_parameters=model_parameters,
        meteorology_input=meteorology_input,
        traffic_input=traffic_input,
        activity_input=activity_input,
        airquality_input=airquality_input,
        model_activities=model_activities,
    )

    time_config.min_time = time_config.min_time_save
    time_config.max_time = time_config.max_time_save

    min_time_original = time_config.min_time
    max_time_original = time_config.max_time

    # Plots are saved if save_type_flag is 2 or 3
    save_plots = model_flags.save_type_flag in [2, 3]
    if save_plots:
        os.makedirs(paths.path_outputfig, exist_ok=True)
        logger.info(f"Creating and saving plots to {paths.path_outputfig}...")

    plot_road_dust_result(
        time_config=time_config,
        converted_data=converted_data,
        metadata=metadata_input,
        airquality_data=airquality_input,
        model_parameters=model_parameters,
        model_flags=model_flags,
        model_variables=model_variables,
        meteo_input=meteorology_input,
        input_activity=activity_input,
        paths=paths,
        ro=0,
        plot_figure=plot_figure,
        print_result=print_results,
        save_plots=save_plots,
    )

    time_config.min_time = min_time_original
    time_config.max_time = max_time_original

    save_data = model_flags.save_type_flag in [1, 3]

    if save_data:
        save_road_dust_results_average(
            time_config=time_config,
            converted_data=converted_data,
            metadata=metadata_input,
            airquality_data=airquality_input,
            model_parameters=model_parameters,
            model_flags=model_flags,
            model_variables=model_variables,
            input_activity=activity_input,
            av=[model_flags.plot_type_flag],
            paths=paths,
            parameter_sheets=parameter_sheets,
        )

    logger.info("End of NORTRIP_Control")
    input("Press Enter to close...")


if __name__ == "__main__":
    main()
