from importlib.metadata import version
from read_files import read_road_dust_parameters, read_road_dust_paths
import constants
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def main():
    print(f"Starting NORTRIP_model_python_v{version('nortrip-python')}...")

    read_as_text = False
    # Loading model parameters and flags
    paths = read_road_dust_paths(read_as_text=read_as_text)

    model_parameters, model_flags, model_activities = read_road_dust_parameters(
        paths.path_filename_inputparam, read_as_text=read_as_text
    )

    logger.info(model_activities)
