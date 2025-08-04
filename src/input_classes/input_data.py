from dataclasses import dataclass
from .input_activity import input_activity
from .input_airquality import input_airquality
from .input_initial import input_initial
from .input_metadata import input_metadata
from .input_meteorology import input_meteorology
from .input_traffic import input_traffic


@dataclass
class input_data:
    """
    Dataclass containing all input data classes for NORTRIP model.
    
    This class consolidates all the individual input dataclasses that are
    read from the input files into a single container for better organization
    and type safety.
    """
    
    activity: input_activity
    airquality: input_airquality
    meteorology: input_meteorology
    traffic: input_traffic
    initial: input_initial
    metadata: input_metadata