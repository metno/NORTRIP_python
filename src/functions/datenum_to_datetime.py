from datetime import datetime, timedelta


def datenum_to_datetime(datenum: float) -> datetime:
    """Convert MATLAB datenum to Python datetime."""
    # MATLAB datenum 1 = January 1, year 0000
    # Python datetime reference is January 1, year 1
    # Need to subtract 366 days because year 0 was a leap year
    matlab_epoch = datetime(1, 1, 1)

    # Use high precision calculation to avoid floating point errors
    days_since_epoch = float(datenum) - 1.0 - 366.0  # Subtract 366 for year 0
    total_seconds = days_since_epoch * 86400.0

    # Round to nearest microsecond to handle floating point precision issues
    total_seconds_rounded = round(total_seconds, 6)

    dt = matlab_epoch + timedelta(seconds=total_seconds_rounded)
    return dt
