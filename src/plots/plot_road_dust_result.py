import constants
from config_classes import model_parameters, model_flags, model_file_paths
from input_classes import (
    converted_data,
    input_activity,
    input_metadata,
    input_airquality,
    input_meteorology,
)
from initialise import time_config, model_variables
from .plot_summary import plot_summary
from .style import apply_plot_style
from .plot_traffic import plot_traffic
from .plot_meteorology import plot_meteorology
from .plot_emissions_mass import plot_emissions_mass
from .init_shared_data import init_shared_data
from .plot_other_factors import plot_other_factors
from .plot_wetness import plot_wetness
from .plot_energy_moisture_balance import plot_energy_moisture_balance
from .plot_concentrations import plot_concentrations
from .plot_scatter_qq import plot_scatter_qq
from .plot_ae import plot_ae
from .plot_scatter_temp_moisture import plot_scatter_temp_moisture
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)


def plot_road_dust_result(
    *,
    time_config: time_config,
    converted_data: converted_data,
    metadata: input_metadata,
    airquality_data: input_airquality,
    model_parameters: model_parameters,
    model_flags: model_flags,
    model_variables: model_variables,
    input_activity: input_activity,
    paths: model_file_paths,
    meteo_input: input_meteorology,
    ro: int = 0,
    plot_size_fraction: int = constants.pm_10,
    plot_figure: list[int] | None = None,
    print_result: bool,
    save_plots: bool,
):
    """
    Entry point for generating NORTRIP plots.

    All data/config classes required for plotting are passed in and then
    dispatched to specific plot functions depending on plot_figure flags.

    Args:
        time_config: Time setup used for averaging ranges etc.
        converted_data: Converted input data (traffic, meteo, activities, dates).
        metadata: Metadata, including nodata and geometry.
        airquality_data: Air quality observations (PM, NOx, etc.).
        model_parameters: Model parameters (tracks, size distributions, etc.).
        model_flags: Model flags (plot_type_flag controls averaging).
        model_variables: Full model outputs (E, C, M arrays, meteo on road, etc.).
        input_activity: Input activity data.
        paths: File/figure naming and title information.
        ro: Road index to plot.
        plot_size_fraction: Size fraction index to plot (defaults to PM10).
    """

    if plot_figure is None:
        plot_figure = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

    # Apply global plot style once per session
    try:
        apply_plot_style()
    except Exception:
        pass

    # Prepare shared data that all plots will consume
    shared = init_shared_data(
        time_config=time_config,
        converted_data=converted_data,
        metadata=metadata,
        airquality_data=airquality_data,
        meteo_input=meteo_input,
        model_parameters=model_parameters,
        model_flags=model_flags,
        model_variables=model_variables,
        input_activity=input_activity,
        ro=ro,
        plot_size_fraction=plot_size_fraction,
        print_result=print_result,
        save_plots=save_plots,
    )

    # Traffic figure (plot_figure index 0)
    if plot_figure[0]:
        logger.info("Plotting figure 1: traffic")
        plot_traffic(shared, paths)

    # Meteorology figure (plot_figure index 1)
    if plot_figure[1]:
        logger.info("Plotting figure 2: meteorology")
        plot_meteorology(shared, paths)

    # Emissions and mass balance figure (plot_figure index 2)
    if plot_figure[2]:
        logger.info("Plotting figure 3: emissions and mass balance")
        plot_emissions_mass(shared, paths)

    # Road wetness figure (plot_figure index 3)
    if plot_figure[3]:
        logger.info("Plotting figure 4: road wetness")
        plot_wetness(shared, paths)

    # Other factors figure (plot_figure index 4)
    if plot_figure[4]:
        logger.info("Plotting figure 5: other factors")
        plot_other_factors(shared, paths)

    # Energy and moisture balance figure (plot_figure index 5)
    if plot_figure[5]:
        logger.info("Plotting figure 6: energy and moisture balance")
        plot_energy_moisture_balance(shared, paths)

    # Concentrations figure (plot_figure index 6)
    if plot_figure[6]:
        logger.info("Plotting figure 7: concentrations")
        plot_concentrations(shared, paths)

    # AE plot (plot_figure index 7)
    if plot_figure[7]:
        logger.info("Plotting figure 8: AE")
        plot_ae(shared, paths)

    # Scatter/QQ plots (plot_figure index 8)
    if plot_figure[8]:
        logger.info("Plotting figure 9: scatter/QQ")
        plot_scatter_qq(shared, paths)

    # Scatter temperature and moisture figure (plot_figure index 9)
    if plot_figure[9]:
        logger.info("Plotting figure 10: scatter temperature and moisture")
        plot_scatter_temp_moisture(shared, paths)

    # Summary figure
    if plot_figure[10]:
        logger.info("Plotting figure 11: summary")
        plot_summary(shared, paths)

    plt.show(block=False)
