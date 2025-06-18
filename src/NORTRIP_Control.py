"""
Main script for the NORTRIP Road Dust Model in Python.
"""

from importlib.metadata import version
from read_files import read_road_dust_parameters, read_road_dust_paths
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
    print(f"Read as text mode: {read_as_text}")
    print(f"Print results to terminal: {print_results}")
    # Loading model parameters and flags
    paths = read_road_dust_paths(read_as_text=read_as_text)

    model_parameters, model_flags, model_activities = read_road_dust_parameters(
        paths.path_filename_inputparam, read_as_text=read_as_text
    )

    logger.info(model_parameters.z0)


if __name__ == "__main__":
    main()
