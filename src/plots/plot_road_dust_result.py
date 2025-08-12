from typing import Optional

import constants
from config_classes.model_parameters import model_parameters as ModelParameters
from config_classes.model_flags import model_flags as ModelFlags
from config_classes.model_file_paths import model_file_paths as ModelFilePaths
from input_classes.converted_data import converted_data as ConvertedData
from input_classes.input_initial import input_initial as InputInitial
from input_classes.input_metadata import input_metadata as InputMetadata
from input_classes.input_airquality import input_airquality as InputAirquality
from initialise.road_dust_initialise_time import time_config as TimeConfig
from initialise.road_dust_initialise_variables import (
    model_variables as ModelVariables,
)
from .plot_summary import plot_summary


def plot_road_dust_result(
    *,
    time_config: TimeConfig,
    converted_data: ConvertedData,
    initial_data: InputInitial,
    metadata: InputMetadata,
    airquality_data: InputAirquality,
    model_parameters: ModelParameters,
    model_flags: ModelFlags,
    model_variables: ModelVariables,
    paths: Optional[ModelFilePaths] = None,
    ro: int = 0,
    plot_size_fraction: int = constants.pm_10,
):
    """
    Entry point for generating NORTRIP plots.

    All data/config classes required for plotting are passed in and then
    dispatched to specific plot functions depending on plot_figure flags.

    Args:
        time_config: Time setup used for averaging ranges etc.
        converted_data: Converted input data (traffic, meteo, activities, dates).
        initial_data: Initial state inputs.
        metadata: Metadata, including nodata and geometry.
        airquality_data: Air quality observations (PM, NOx, etc.).
        model_parameters: Model parameters (tracks, size distributions, etc.).
        model_flags: Model flags (plot_type_flag controls averaging).
        model_variables: Full model outputs (E, C, M arrays, meteo on road, etc.).
        paths: File/figure naming and title information.
        ro: Road index to plot.
        plot_size_fraction: Size fraction index to plot (defaults to PM10).
    """

    # Configure which figures to render (aligns with MATLAB order)
    # [1..14] where index 13 (0-based 12) is the Summary figure
    plot_figure = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]

    # Shared pre-compute could be added here (track-averaged temp arrays, etc.)

    # Summary figure
    if len(plot_figure) >= 13 and plot_figure[12] == 1:
        plot_summary(
            time_config=time_config,
            converted_data=converted_data,
            metadata=metadata,
            airquality_data=airquality_data,
            model_parameters=model_parameters,
            model_flags=model_flags,
            model_variables=model_variables,
            paths=paths,
            ro=ro,
            plot_size_fraction=plot_size_fraction,
        )
