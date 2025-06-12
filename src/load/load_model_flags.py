import config_classes.simulation_flags as flags
import pandas as pd

def load_model_flags_xlsx(file_path: str) -> flags.simulation_flags:
    """
    Load model flags from xlsx file and return an instance of simulation_flags.
    
    Args:
        file_path (str): Path to the file containing model flags.
        
    Returns:
        simulation_flags: An instance of simulation_flags with loaded values.
    """
    loaded_flags = flags.simulation_flags()

    try:
        flags_df = pd.read_excel(file_path, sheet_name="Flags")
        for _, row in flags_df.iterrows():
            flag_name = str(row.tolist()[0]).strip()
            # Set the flag value in the loaded_flags dataclass
            if hasattr(loaded_flags, flag_name):
                flag_value = int(row.tolist()[1])
                setattr(loaded_flags, flag_name, flag_value)
            else:
                # print(f"Warning: Flag '{flag_name}' not found in simulation_flags dataclass.")
                pass

            
    except Exception as e:
        print(f"Error loading model flags from {file_path}: {e}")
        exit()
    
    
    return loaded_flags