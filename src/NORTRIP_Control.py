from importlib.metadata import version
from read_files.read_road_dust_paths import read_road_dust_paths
from read_files.load_model_parameters import load_model_flags, load_model_parameters
import pandas as pd


def main():
    print(f"Starting NORTRIP_model_python_v{version('nortrip-python')}...")

    # Loading model parameters and flags
    paths = read_road_dust_paths()
    print(paths.path_filename_inputparam)

    flags_df = pd.read_excel(paths.path_filename_inputparam, sheet_name="Flags")
    model_flags = load_model_flags(flags_df)

    print(model_flags)
