from importlib.metadata import version
from load_model_parameters import load_model_flags_xlsx, load_model_parameters_xlsx


def main():
    package_version = version("nortrip-python")
    print(f"Starting NORTRIP_model_python_v{package_version}...")

    # Loading model parameters and flags
    paramater_path = "model_parameters/Road_dust_parameter_table_v11.xlsx"
    model_paramters = load_model_parameters_xlsx(paramater_path)
    model_flags = load_model_flags_xlsx(paramater_path)
