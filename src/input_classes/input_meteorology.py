from dataclasses import dataclass, field
import numpy as np


@dataclass
class input_meteorology:
    """
    Dataclass containing parsed meteorological input data, matching the MATLAB structure and defaults.
    """
