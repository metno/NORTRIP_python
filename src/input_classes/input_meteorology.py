from dataclasses import dataclass, field
import numpy as np


@dataclass
class input_meteorology:
    """
    Dataclass containing parsed meteorological input data, matching the MATLAB structure and defaults.

    This class contains all meteorological variables that are read from the input file,
    along with their availability flags and missing data indices.
    """

    n_meteo: int = 0
