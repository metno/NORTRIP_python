from .read_input_activity import read_input_activity
from .read_input_airquality import read_input_airquality
from .read_input_meteorology import read_input_meteorology
from .read_input_traffic import read_input_traffic
from .read_input_initial import read_input_initial
from .read_input_metadata import read_input_metadata


def read_road_dust_input(
    input_file_path: str,
    read_as_text=False,
):
    """
    Read road dust input data from specified file.

    Args:
        input_file_path (str): Path to the input file.
        read_as_text (bool, optional): If True, read the file as text. Will reformat input_file_path to text format.

    Returns:

    """
