from importlib.metadata import version
from load.load_model_flags import load_model_flags_xlsx
import os

def main():
    package_version = version("nortrip-python")
    print(f"Starting NORTRIP_model_python_v{package_version}...")
    file_path = "model_parameters/Road_dust_parameter_table_v11.xlsx"
    model_flags = load_model_flags_xlsx(file_path)
    print(model_flags)
