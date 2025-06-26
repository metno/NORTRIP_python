from dataclasses import dataclass, field
import numpy as np
import datetime
import logging
import constants
from input_classes import input_metadata

logger = logging.getLogger(__name__)


@dataclass
class time_config:
    """
    Configuration dataclass for time loop parameters in NORTRIP model execution.
    """

    # Time loop indices
    min_time: int = 0
    max_time: int = 0
    max_time_inputdata: int = 0

    # Time step for iteration (hours)
    dt: np.float64 = np.float64(1.0)

    # Flag for incorrect start and stop times
    time_bad: int = 0

    # Save time indices
    min_time_save: int = 0
    max_time_save: int = 0

    # Subdate save time indices
    min_subtime_save: list = field(default_factory=list)
    max_subtime_save: list = field(default_factory=list)


def _parse_date_string(date_str: str) -> tuple[int, int, int, int, int, int]:
    """
    Parse date string to extract year, month, day, hour, minute, second.

    Args:
        date_str: Date string in format "YYYY-MM-DD HH:MM:SS" or similar

    Returns:
        tuple: (year, month, day, hour, minute, second)
    """
    if not date_str:
        return 0, 0, 0, 0, 0, 0

    # Strip whitespace from the date string
    date_str = date_str.strip()

    # List of date formats to try
    date_formats = [
        "%Y-%m-%d %H:%M:%S",  # 2023-03-15 14:30:00
        "%Y.%m.%d %H:%M:%S",  # 2023.03.15 14:30:00
        "%d.%m.%Y %H:%M",  # 01.10.2010 01:00
        "%d.%m.%Y %H:%M:%S",  # 01.10.2010 01:00:00
        "%Y-%m-%d %H:%M",  # 2023-03-15 14:30
        "%Y.%m.%d %H:%M",  # 2023.03.15 14:30
        "%Y-%m-%d",  # 2023-03-15
        "%Y.%m.%d",  # 2023.03.15
        "%d.%m.%Y",  # 01.10.2010
    ]

    # Handle date only formats by adding time (but be careful not to modify strings with time already)
    if len(date_str) <= 10:  # Date only format
        if " " not in date_str:
            date_str += " 00:00:00"

    for fmt in date_formats:
        try:
            dt = datetime.datetime.strptime(date_str, fmt)
            return dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second
        except ValueError:
            continue

    logger.error(f"Could not parse date string: {date_str}")
    return 0, 0, 0, 0, 0, 0


def _find_time_index(
    year: int, month: int, day: int, hour: int, date_data: np.ndarray
) -> int:
    """
    Find the index in date_data that matches the given date and time.

    Args:
        year, month, day, hour: Date/time components to find
        date_data: Date data array with shape [num_date_index, n_date, n_roads]

    Returns:
        int: Index if found, -1 if not found
    """
    # Extract date components from the first road (index 0)
    years = date_data[constants.year_index, :, 0]
    months = date_data[constants.month_index, :, 0]
    days = date_data[constants.day_index, :, 0]
    hours = date_data[constants.hour_index, :, 0]

    # Find matching indices
    matches = np.where(
        (years == year) & (months == month) & (days == day) & (hours == hour)
    )[0]

    if len(matches) == 1:
        return int(matches[0])
    elif len(matches) > 1:
        logger.warning(
            f"Multiple matches found for date {year}-{month:02d}-{day:02d} {hour:02d}"
        )
        return int(matches[0])
    else:
        return -1


def road_dust_initialise_time(
    date_data: np.ndarray,
    n_date: int,
    metadata: input_metadata,
    use_fortran_flag: bool = False,
) -> time_config:
    """
    Initialize time loop parameters for NORTRIP model execution.

    This function replicates the MATLAB road_dust_initialise_time functionality,
    setting time loop indices, time step, and parsing start/end dates from metadata.

    Args:
        date_data: Date data array with shape [num_date_index, n_date, n_roads]
        n_date: Number of time steps in the data
        metadata: Metadata containing date strings and other parameters
        use_fortran_flag: If True, always run all the data

    Returns:
        time_config: Configuration object with time parameters
    """
    # Initialize time configuration
    config = time_config()

    # Set time loop index
    config.min_time = 0
    config.max_time = n_date - 1  # 0-based indexing: last valid index is n_date - 1
    config.max_time_inputdata = n_date - 1

    # Set time step for iteration based on the first time step of the input data
    if n_date > 1:
        config.dt = np.round(
            (
                date_data[constants.datenum_index, 1, 0]
                - date_data[constants.datenum_index, 0, 0]
            )
            * 24,
            decimals=6,
        )
    else:
        config.dt = np.float64(1.0)  # Default to 1 hour if only one time step

    # Flag for incorrect start and stop times. Stops model run
    config.time_bad = 0

    # Set start and end dates based on date string (if specified in metadata)
    if metadata.start_date_str:
        Y, M, D, H, MN, S = _parse_date_string(metadata.start_date_str)
        rstart = _find_time_index(Y, M, D, H, date_data)
        if rstart != -1:
            config.min_time = rstart
        else:
            logger.error("Start date not found. Stopping")
            config.time_bad = 1
            return config

    if metadata.end_date_str:
        Y, M, D, H, MN, S = _parse_date_string(metadata.end_date_str)
        rend = _find_time_index(Y, M, D, H, date_data)
        if rend != -1:
            config.max_time = rend
        else:
            logger.error("End date not found. Stopping")
            config.time_bad = 1
            return config

    # Set start and end dates for plotting and saving based on date string (if specified)
    if metadata.start_date_save_str:
        Y, M, D, H, MN, S = _parse_date_string(metadata.start_date_save_str)
        rstart = _find_time_index(Y, M, D, H, date_data)
        if rstart != -1:
            config.min_time_save = rstart
        else:
            logger.error("Start save date not found. Stopping")
            config.time_bad = 1
            return config
    else:
        config.min_time_save = config.min_time

    if metadata.end_date_save_str:
        Y, M, D, H, MN, S = _parse_date_string(metadata.end_date_save_str)
        rend = _find_time_index(Y, M, D, H, date_data)
        if rend != -1:
            config.max_time_save = rend
        else:
            logger.error("End save date not found. Stopping")
            config.time_bad = 1
            return config
    else:
        config.max_time_save = config.max_time

    # Set the subdate start and end dates for plotting and saving based on date string (if specified)
    if metadata.n_save_subdate > 1:
        config.min_subtime_save = []
        config.max_subtime_save = []

        for i_subdate in range(metadata.n_save_subdate):
            # Handle start subdate
            if i_subdate < len(metadata.start_subdate_save_str):
                start_subdate_str = metadata.start_subdate_save_str[i_subdate]
                Y, M, D, H, MN, S = _parse_date_string(start_subdate_str)
                rstart = _find_time_index(Y, M, D, H, date_data)
                if rstart != -1:
                    config.min_subtime_save.append(rstart)
                else:
                    logger.error("Start save subdate not found. Stopping")
                    config.time_bad = 1
                    return config
            else:
                config.min_subtime_save.append(config.min_time_save)

            # Handle end subdate
            if i_subdate < len(metadata.end_subdate_save_str):
                end_subdate_str = metadata.end_subdate_save_str[i_subdate]
                Y, M, D, H, MN, S = _parse_date_string(end_subdate_str)
                rend = _find_time_index(Y, M, D, H, date_data)
                if rend != -1:
                    config.max_subtime_save.append(rend)
                else:
                    logger.error("End save subdate not found. Stopping")
                    config.time_bad = 1
                    return config
            else:
                config.max_subtime_save.append(config.max_time_save)

    # Always run all the data when using fortran
    if use_fortran_flag:
        config.min_time = 0
        config.max_time = n_date - 1

    logger.info("Time configuration initialized.")

    return config
