from parameter_classes.model_flags import model_flags
from parameter_classes.model_parameters import model_parameters
import pandas as pd


def load_model_flags_xlsx(file_path: str) -> model_flags:
    """
    Load model flags from xlsx file and return an instance of model_flags.

    Args:
        file_path (str): Path to the file containing model flags.

    Returns:
        model_flags: An instance of model_flags with loaded values.
    """
    loaded_flags = model_flags()

    try:
        flags_df = pd.read_excel(file_path, sheet_name="Flags")
        for _, row in flags_df.iterrows():
            flag_name = str(row.tolist()[0]).strip()
            if hasattr(loaded_flags, flag_name):
                flag_value = int(row.tolist()[1])
                setattr(loaded_flags, flag_name, flag_value)
            else:
                # print(f"Warning: Flag '{flag_name}' not found in simulation_flags dataclass.")
                pass

    except Exception as e:
        print(f"Error loading model flags from {file_path}: {e}, using default values.")
        return model_flags()

    return loaded_flags


def load_model_parameters_xlsx(file_path: str) -> model_parameters:
    """
    Load model parameters from xlsx file and return an instance of model_parameters.
    Args:
        file_path (str): Path to the file containing model parameters.

    Returns:
        loaded_parameters (model paramters): An instance of model_parameters with loaded values.
    """
    loaded_parameters = model_parameters()



    return loaded_parameters