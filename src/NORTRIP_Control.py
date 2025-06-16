from importlib.metadata import version
from read_files import read_road_dust_paths, read_model_flags, read_model_parameters
import pandas as pd
import constants
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def main():
    print(f"Starting NORTRIP_model_python_v{version('nortrip-python')}...")

    read_as_text = 0
    # Loading model parameters and flags
    paths = read_road_dust_paths(read_as_text=read_as_text)

    input_parameters_sheets = pd.read_excel(
        paths.path_filename_inputparam, sheet_name=None
    )

    model_flags = read_model_flags(input_parameters_sheets["Flags"])
