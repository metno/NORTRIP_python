import logging
from pathlib import Path

import numpy as np

import constants
from initialise import time_config
from input_classes import converted_data, input_metadata

logger = logging.getLogger(__name__)


def write_input_file(
    *,
    input_file: Path,
    converted: converted_data,
    metadata: input_metadata,
    time_config: time_config,
    ro: int,
) -> None:
    """Create the OSPM input data file."""
    input_file.parent.mkdir(parents=True, exist_ok=True)

    idx = range(time_config.min_time, time_config.max_time + 1)

    year = converted.date_data[constants.year_index, idx, ro]
    month = converted.date_data[constants.month_index, idx, ro]
    day = converted.date_data[constants.day_index, idx, ro]
    hour = converted.date_data[constants.hour_index, idx, ro]

    U_mast = (
        converted.meteo_data[constants.FF_index, idx, ro]
        / metadata.wind_speed_correction
    )
    wind_dir = converted.meteo_data[constants.DD_index, idx, ro]
    TK = converted.meteo_data[constants.T_a_index, idx, ro] + 273.15
    GlobalRad = converted.meteo_data[constants.short_rad_in_index, idx, ro]

    NNp = converted.traffic_data[constants.N_v_index[constants.li], idx, ro]
    NNt = converted.traffic_data[constants.N_v_index[constants.he], idx, ro]
    Vp = converted.traffic_data[constants.V_veh_index[constants.li], idx, ro]
    Vt = converted.traffic_data[constants.V_veh_index[constants.he], idx, ro]

    zeros = np.zeros_like(U_mast)
    qNOX = np.full_like(U_mast, 1.0 / 3.6, dtype=float)

    with input_file.open("w", encoding="utf-8") as file_handle:
        for i in range(len(idx)):
            file_handle.write(
                f"{int(year[i])}\t{int(month[i])}\t{int(day[i])}\t{int(hour[i])}\t"
                f"{U_mast[i]:6.2f}\t{wind_dir[i]:6.2f}\t{TK[i]:6.2f}\t{GlobalRad[i]:6.2f}\t"
                f"{zeros[i]:6.2f}\t{NNp[i]:6.2f}\t{NNt[i]:6.2f}\t{Vp[i]:6.2f}\t{Vt[i]:6.2f}\t{qNOX[i]:6.2f}\n"
            )

    logger.debug("Wrote OSPM input file to %s", input_file)
