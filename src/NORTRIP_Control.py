"""
Main script for the NORTRIP Road Dust Model in Python.
"""

from importlib.metadata import version
import constants
from read_files import (
    read_road_dust_parameters,
    read_road_dust_paths,
    read_road_dust_input,
)
from input_classes.converted_data import convert_read_road_dust_input
from initialise import road_dust_initialise_time, road_dust_initialise_variables
from calculations import calc_radiation
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
    use_fortran = False

    print(f"Starting NORTRIP_python_v{version('nortrip-python')}...")
    print(f"Read as inputs as text: {read_as_text}")
    print(f"Print results to terminal: {print_results}")

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

    converted_data = convert_read_road_dust_input(
        input_data, nodata=metadata_input.nodata
    )

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

    # Main model loop

    for ro in range(constants.n_roads):
        logger.info("Calculating radiation and running mean temperature")
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

    logger.info("End of NORTRIP_Control")


if __name__ == "__main__":
    main()
