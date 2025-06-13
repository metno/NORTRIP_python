from importlib.metadata import version
from read_files import read_road_dust_paths, read_model_flags, read_model_parameters
import pandas as pd


def main():
    print(f"Starting NORTRIP_model_python_v{version('nortrip-python')}...")

    read_as_text = 0
    # Loading model parameters and flags
    paths = read_road_dust_paths(read_as_text=read_as_text)

    flags_df = pd.read_excel(paths.path_filename_inputparam, sheet_name="Flags")
    model_flags = read_model_flags(flags_df)
