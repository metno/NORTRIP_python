"""
NORTRIP model variable conversion
Converts binned data to integrated arrays
"""

import numpy as np
import constants
import logging

from initialise import model_variables
from input_classes import input_metadata

logger = logging.getLogger(__name__)


def road_dust_convert_variables(
    model_variables: model_variables,
    metadata: input_metadata,
    ro: int = 0,
):
    """
    Convert binned balance data into normal arrays.

    This function sums the binned data over size fractions to create
    integrated arrays for mass, emissions, and concentrations.

    Args:
        model_variables: Model variables containing binned and integrated arrays
        metadata: Metadata including nodata value
        ro: Road index (default 0)
    """

    logger.info("Converting binned variables to integrated arrays")

    # Put binned data in integrated data arrays
    # Sum from current size fraction to largest (x:num_size)
    for x in range(constants.num_size):
        # Mass data: M_road_data[source, size, time, track, road] =
        # sum(M_road_bin_data[source, size:num_size, time, track, road])
        model_variables.M_road_data[:, x, :, :, ro] = np.sum(
            model_variables.M_road_bin_data[:, x : constants.num_size, :, :, ro], axis=1
        )

        # Mass balance data: M_road_balance_data[source, size, process, time, track, road] =
        # sum(M_road_bin_balance_data[source, size:num_size, process, time, track, road])
        model_variables.M_road_balance_data[:, x, :, :, :, ro] = np.sum(
            model_variables.M_road_bin_balance_data[
                :, x : constants.num_size, :, :, :, ro
            ],
            axis=1,
        )

        # Emission data: E_road_data[source, size, process, time, track, road] =
        # sum(E_road_bin_data[source, size:num_size, process, time, track, road])
        model_variables.E_road_data[:, x, :, :, :, ro] = np.sum(
            model_variables.E_road_bin_data[:, x : constants.num_size, :, :, :, ro],
            axis=1,
        )

        # Concentration data: C_data[source, size, process, time, track, road] =
        # sum(C_bin_data[source, size:num_size, process, time, track, road])
        model_variables.C_data[:, x, :, :, :, ro] = np.sum(
            model_variables.C_bin_data[:, x : constants.num_size, :, :, :, ro], axis=1
        )

    nodata_indices = np.where(model_variables.f_conc[:, ro] == metadata.nodata)[0]

    # Set concentration data to nodata for those time indices
    for ti in nodata_indices:
        if ti < model_variables.C_data.shape[3]:
            model_variables.C_data[:, :, :, ti, :, ro] = metadata.nodata
