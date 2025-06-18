"""
Main script for the NORTRIP Road Dust Model in Python.
"""

from importlib.metadata import version
from read_files import read_road_dust_parameters, read_road_dust_paths
import logging
import argparse

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="NORTRIP Road Dust Model")
    parser.add_argument(
        "-t",
        "--text",
        type=int,
        choices=[0, 1],
        default=0,
        help="Read as text mode (0=False, 1=True). Default is 0.",
    )

    args = parser.parse_args()
    read_as_text = bool(args.text)

    print(f"Starting NORTRIP_model_python_v{version('nortrip-python')}...")
    print(f"Read as text mode: {read_as_text}")
    # Loading model parameters and flags
    paths = read_road_dust_paths(read_as_text=read_as_text)

    model_parameters, model_flags, model_activities = read_road_dust_parameters(
        paths.path_filename_inputparam, read_as_text=read_as_text
    )

    logger.info(model_parameters.z0)


if __name__ == "__main__":
    main()
