"""
NORTRIP model road dust dispersion calculation
Converts emissions to concentrations using NOX as a tracer
"""

import numpy as np
import logging

from initialise import time_config, model_variables
from input_classes import converted_data, input_metadata, input_airquality
from config_classes import model_parameters

logger = logging.getLogger(__name__)


def road_dust_dispersion(
    time_config: time_config,
    converted_data: converted_data,
    model_variables: model_variables,
    model_parameters: model_parameters,
    metadata: input_metadata,
    airquality_data: input_airquality,
    ro: int = 0,
):
    """
    Calculate dispersion factors using NOX as a tracer.

    This function calculates f_conc (concentration factor) by comparing
    NOX observations with NOX emissions. If dispersion factor is directly
    available, it uses that instead.

    Args:
        time_config: Time configuration with min_time, max_time, dt, etc.
        converted_data: Consolidated input data arrays
        model_variables: Model variables containing f_q array
        model_parameters: Model parameters including conc_min and emis_min
        metadata: Metadata including nodata value
        airquality_data: Air quality data with NOX observations and emissions
        ro: Road index (default 0)
    """

    logger.info("Calculating dispersion using NOX")

    # Extract NOX data from airquality input
    NOX_obs_net = airquality_data.NOX_obs_net
    NOX_emis = airquality_data.NOX_emis

    # Use dispersion factor if directly available
    if airquality_data.f_dis_available:
        f_dis = airquality_data.f_dis_input
        for ti in range(time_config.min_time, time_config.max_time + 1):
            if ti < len(f_dis):
                model_variables.f_conc[ti, ro] = f_dis[ti]
            else:
                model_variables.f_conc[ti, ro] = metadata.nodata
    else:
        # Calculate dispersion factor from NOX data
        for ti in range(time_config.min_time, time_config.max_time + 1):
            if (
                ti < len(NOX_obs_net)
                and ti < len(NOX_emis)
                and NOX_obs_net[ti] != metadata.nodata
                and NOX_emis[ti] != metadata.nodata
                and not np.isnan(NOX_obs_net[ti])
                and not np.isnan(NOX_emis[ti])
                and NOX_obs_net[ti] > model_parameters.conc_min
                and NOX_emis[ti] > model_parameters.emis_min
            ):
                model_variables.f_conc[ti, ro] = NOX_obs_net[ti] / NOX_emis[ti]
            else:
                model_variables.f_conc[ti, ro] = metadata.nodata
