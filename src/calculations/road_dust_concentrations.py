"""
NORTRIP model road dust concentration calculation
Converts emissions to concentrations using dispersion factors
"""

import logging

from initialise import time_config
from input_classes import input_metadata, model_variables

logger = logging.getLogger(__name__)


def road_dust_concentrations(
    time_config: time_config,
    model_variables: model_variables,
    metadata: input_metadata,
    ro: int = 0,
):
    """
    Convert emissions to concentrations using dispersion factors.

    This function multiplies the emission arrays (E_road_bin_data) by the
    dispersion factor (f_conc) to get concentration arrays (C_bin_data).

    Args:
        time_config: Time configuration with min_time, max_time, dt, etc.
        model_variables: Model variables containing emission and concentration arrays
        metadata: Metadata including nodata value
        ro: Road index (default 0)
    """

    logger.info("Calculating concentrations and converting variables")

    # Convert emissions to concentrations
    for ti in range(time_config.min_time, time_config.max_time + 1):
        if (
            ti < model_variables.f_conc.shape[0]
            and model_variables.f_conc[ti, ro] != metadata.nodata
        ):
            # Apply dispersion factor to convert emissions to concentrations
            # C_bin_data[source, size, process, time, track, road] =
            # E_road_bin_data[source, size, process, time, track, road] * f_conc[time, road]
            model_variables.C_bin_data[:, :, :, ti, :, ro] = (
                model_variables.E_road_bin_data[:, :, :, ti, :, ro]
                * model_variables.f_conc[ti, ro]
            )
        else:
            # Set to nodata if dispersion factor is not available
            model_variables.C_bin_data[:, :, :, ti, :, ro] = metadata.nodata
