from importlib.metadata import version
from read_files.load_model_parameters import load_model_flags, load_model_parameters
import pandas as pd


def main():
    print(f"Starting NORTRIP_model_python_v{version('nortrip-python')}...")

    # Loading model parameters and flags
    paramater_path = "model_parameters/Road_dust_parameter_table_v11.xlsx"
    # model_paramters = load_model_parameters_xlsx(paramater_path)

    flags_df = pd.read_excel(paramater_path, sheet_name="Flags")
    model_flags = load_model_flags(flags_df)

    print(model_flags)
