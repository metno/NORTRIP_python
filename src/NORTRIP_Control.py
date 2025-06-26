"""
Main script for the NORTRIP Road Dust Model in Python.
"""

from importlib.metadata import version
from read_files import (
    read_road_dust_parameters,
    read_road_dust_paths,
    read_road_dust_input,
)
from input_classes.converted_data import convert_read_road_dust_input_output
from initialise import road_dust_initialise_time
import logging
from model_args import create_arg_parser


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def main():
    args = create_arg_parser().parse_args()
    read_as_text = bool(args.text)
    print_results = bool(args.print)

    print(f"Starting NORTRIP_model_python_v{version('nortrip-python')}...")
    print(f"Read as inputs as text: {read_as_text}")
    print(f"Print results to terminal: {print_results}")

    # Loading model parameters and flags
    paths = read_road_dust_paths(read_as_text=read_as_text)

    model_parameters, model_flags, model_activities = read_road_dust_parameters(
        paths.path_filename_inputparam, read_as_text=read_as_text
    )

    # Read input data
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

    # Convert individual input data classes to consolidated structure (matching MATLAB format)
    converted_data = convert_read_road_dust_input_output(
        input_data, nodata=metadata_input.nodata
    )

    # Initialize time loop parameters
    time_config = road_dust_initialise_time(
        date_data=converted_data.date_data,
        n_date=converted_data.n_date,
        metadata=metadata_input,
        use_fortran_flag=False,  # Set based on model flags if needed
    )

    # Check for time configuration errors
    if time_config.time_bad:
        logger.error("Time configuration failed - stopping execution")
        return

    logger.info("End of NORTRIP_Control")


if __name__ == "__main__":
    main()
