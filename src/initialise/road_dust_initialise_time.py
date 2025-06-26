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
    Dataclass containing time configuration for the NORTRIP model execution.

    This dataclass holds all time-related parameters computed during the
    time loop initialization, including start/end times, save times, and
    time step information.
    """

    # Basic time loop parameters
    min_time: int = 0  # Start time index (0-based)
    max_time: int = 0  # End time index (0-based, exclusive)
    max_time_inputdata: int = 0  # Maximum time from input data
    dt: float = 1.0  # Time step in hours

    # Error flag
    time_bad: int = 0  # Flag for incorrect start/stop times

    # Save time parameters
    min_time_save: int = 0  # Start time index for saving/plotting (0-based)
    max_time_save: int = 0  # End time index for saving/plotting (0-based, exclusive)

    # Subdate save parameters (for multiple save periods)
    min_subtime_save: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=int)
    )
    max_subtime_save: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=int)
    )

    # Date format for parsing
    date_format_str: str = "%Y-%m-%d %H:%M:%S"


def road_dust_initialise_time(
    date_data: np.ndarray,
    n_date: int,
    metadata: input_metadata,
    use_fortran_flag: bool = False,
) -> time_config:
    """
    Initialize time loop parameters for NORTRIP model execution.

    This function replicates the MATLAB time loop initialization logic,
    setting up time indices, calculating time steps, and validating
    start/end dates.

    Args:
        date_data: Date data array with shape [num_date_index, n_date, n_roads]
        n_date: Number of time points in data
        metadata: Input metadata object containing date strings and configuration
        use_fortran_flag: If True, always run all data (ignore date restrictions)

    Returns:
        time_config: Configuration object with all time parameters
    """
    # Initialize time configuration
    config = time_config()
    config.date_format_str = "%d.%m.%Y %H:%M"  # Default NORTRIP format

    # Set time loop index (0-based indexing)
    config.min_time = 0  # Start from index 0
    config.max_time = (
        n_date  # End at n_date (exclusive, so last valid index is n_date-1)
    )
    config.max_time_inputdata = n_date

    # Set time step for iteration based on the first time step of the input data
    if n_date > 1:
        # Convert from days to hours (MATLAB datenum difference * 24)
        config.dt = (
            date_data[constants.datenum_index, 1, 0]
            - date_data[constants.datenum_index, 0, 0]
        ) * 24
    else:
        config.dt = 1.0  # Default to 1 hour

    # Flag for incorrect start and stop times. Stops model run
    config.time_bad = 0

    # Set start and end dates based on date string (if specified)
    if metadata.start_date_str and metadata.start_date_str.strip():
        start_datetime = _parse_date_string(metadata.start_date_str)
        if start_datetime is not None:
            rstart = _find_matching_time_index(date_data, start_datetime, n_date)
            if rstart is not None:
                config.min_time = rstart  # Use 0-based indexing
            else:
                print("Start date not found. Stopping")
                config.time_bad = 1
                return config
        else:
            logger.error(f"Error parsing start date '{metadata.start_date_str}'")
            config.time_bad = 1
            return config

    if metadata.end_date_str and metadata.end_date_str.strip():
        end_datetime = _parse_date_string(metadata.end_date_str)
        if end_datetime is not None:
            rend = _find_matching_time_index(date_data, end_datetime, n_date)
            if rend is not None:
                config.max_time = rend + 1  # Exclusive end index (0-based)
            else:
                print("End date not found. Stopping")
                config.time_bad = 1
                return config
        else:
            logger.error(f"Error parsing end date '{metadata.end_date_str}'")
            config.time_bad = 1
            return config

    # Set start and end dates for plotting and saving based on date string (if specified)
    if metadata.start_date_save_str and metadata.start_date_save_str.strip():
        start_save_datetime = _parse_date_string(metadata.start_date_save_str)
        if start_save_datetime is not None:
            rstart = _find_matching_time_index(date_data, start_save_datetime, n_date)
            if rstart is not None:
                config.min_time_save = rstart  # Use 0-based indexing
            else:
                print("Start save date not found. Stopping")
                config.time_bad = 1
                return config
        else:
            logger.error(
                f"Error parsing start save date '{metadata.start_date_save_str}'"
            )
            config.time_bad = 1
            return config
    else:
        config.min_time_save = config.min_time

    if metadata.end_date_save_str and metadata.end_date_save_str.strip():
        end_save_datetime = _parse_date_string(metadata.end_date_save_str)
        if end_save_datetime is not None:
            rend = _find_matching_time_index(date_data, end_save_datetime, n_date)
            if rend is not None:
                config.max_time_save = rend + 1  # Exclusive end index (0-based)
            else:
                print("End save date not found. Stopping")
                config.time_bad = 1
                return config
        else:
            logger.error(f"Error parsing end save date '{metadata.end_date_save_str}'")
            config.time_bad = 1
            return config
    else:
        config.max_time_save = config.max_time

    # Set the subdate start and end dates for plotting and saving based on date string (if specified)
    if (
        metadata.n_save_subdate > 1
        and len(metadata.start_subdate_save_str) > 0
        and len(metadata.end_subdate_save_str) > 0
    ):
        min_subtime_list = []
        max_subtime_list = []

        for i_subdate in range(metadata.n_save_subdate):
            if i_subdate < len(metadata.start_subdate_save_str):
                start_sub_datetime = _parse_date_string(
                    metadata.start_subdate_save_str[i_subdate]
                )
                if start_sub_datetime is not None:
                    rstart = _find_matching_time_index(
                        date_data, start_sub_datetime, n_date
                    )
                    if rstart is not None:
                        min_subtime_list.append(rstart)  # Use 0-based indexing
                    else:
                        print("Start save subdate not found. Stopping")
                        config.time_bad = 1
                        return config
                else:
                    logger.error(
                        f"Error parsing start subdate {i_subdate + 1} '{metadata.start_subdate_save_str[i_subdate]}'"
                    )
                    config.time_bad = 1
                    return config

            if i_subdate < len(metadata.end_subdate_save_str):
                end_sub_datetime = _parse_date_string(
                    metadata.end_subdate_save_str[i_subdate]
                )
                if end_sub_datetime is not None:
                    rend = _find_matching_time_index(
                        date_data, end_sub_datetime, n_date
                    )
                    if rend is not None:
                        max_subtime_list.append(
                            rend + 1
                        )  # Exclusive end index (0-based)
                    else:
                        print("End save subdate not found. Stopping")
                        config.time_bad = 1
                        return config
                else:
                    logger.error(
                        f"Error parsing end subdate {i_subdate + 1} '{metadata.end_subdate_save_str[i_subdate]}'"
                    )
                    config.time_bad = 1
                    return config

        config.min_subtime_save = np.array(min_subtime_list, dtype=int)
        config.max_subtime_save = np.array(max_subtime_list, dtype=int)

    # Always run all the data when using fortran
    if use_fortran_flag:
        config.min_time = 0
        config.max_time = n_date
        logger.info(
            "Using Fortran flag: running all data regardless of date restrictions"
        )

    return config


def _parse_date_string(date_str: str) -> datetime.datetime | None:
    """
    Parse date string with multiple format attempts.

    Tries common date formats used in NORTRIP input files.

    Args:
        date_str: Date string to parse

    Returns:
        datetime.datetime or None: Parsed datetime or None if parsing failed
    """
    # Clean up the string
    date_str = date_str.strip()

    # List of possible date formats to try
    date_formats = [
        "%d.%m.%Y %H:%M",  # dd.mm.yyyy hh:mm (most common in NORTRIP)
        "%Y-%m-%d %H:%M:%S",  # yyyy-mm-dd hh:mm:ss
        "%Y-%m-%d %H:%M",  # yyyy-mm-dd hh:mm
        "%d.%m.%Y %H:%M:%S",  # dd.mm.yyyy hh:mm:ss
        "%d/%m/%Y %H:%M",  # dd/mm/yyyy hh:mm
        "%d/%m/%Y %H:%M:%S",  # dd/mm/yyyy hh:mm:ss
    ]

    for fmt in date_formats:
        try:
            return datetime.datetime.strptime(date_str, fmt)
        except ValueError:
            continue

    logger.error(f"Unable to parse date string '{date_str}' with any known format")
    return None


def _find_matching_time_index(
    date_data: np.ndarray, target_datetime: datetime.datetime, n_date: int
) -> int | None:
    """
    Find the index in date_data that matches the target datetime.

    Args:
        date_data: Date data array with shape [num_date_index, n_date, n_roads]
        target_datetime: Target datetime to find
        n_date: Number of time points

    Returns:
        int or None: 0-based index of matching time, or None if not found
    """
    # Extract date components from target
    target_year = target_datetime.year
    target_month = target_datetime.month
    target_day = target_datetime.day
    target_hour = target_datetime.hour

    # Search through all time points (using 0-based indexing for array access)
    for i in range(n_date):
        if (
            int(date_data[constants.year_index, i, 0]) == target_year
            and int(date_data[constants.month_index, i, 0]) == target_month
            and int(date_data[constants.day_index, i, 0]) == target_day
            and int(date_data[constants.hour_index, i, 0]) == target_hour
        ):
            return i

    return None
