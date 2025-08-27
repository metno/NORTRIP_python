import numpy as np
import matplotlib.pyplot as plt

import constants
from functions.average_data_func import average_data_func
from config_classes import model_file_paths
from .shared_plot_data import shared_plot_data
from .helpers import (
    matlab_datenum_to_datetime_array,
    format_time_axis,
)


def plot_summary(shared: shared_plot_data, paths: model_file_paths) -> None:
    """
    Render the first panel of the summary chart (concentrations over time).

    This implements the MATLAB section under plot_figure(13) → first subplot,
    computing averaged series and plotting observed vs source contributions.
    """

    # Select size fraction and set labels
