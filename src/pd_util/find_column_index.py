import pandas as pd
import logging

logger = logging.getLogger(__name__)


def find_column_index(
    search_text: str,
    header_text: pd.Series,
    print_results: bool = False,
    exact_match: bool = False,
) -> int:
    """
    Find column index for given search text, handling duplicates.

    Args:
        search_text (str): Text to search for in headers
        header_text (pd.Series): Series containing header text
        print_results (bool): Whether to print warning messages for duplicates
        exact_match (bool): If True, use exact string match. If False, use substring search.

    Returns:
        int: Column index if found, -1 if not found
    """
    if exact_match:
        # Use exact string match (case-insensitive)
        matches = header_text.str.lower() == search_text.lower()
    else:
        # Use substring search (case-insensitive, no regex)
        matches = header_text.str.contains(
            search_text, case=False, na=False, regex=False
        )

    if matches.sum() > 1 and print_results:
        logger.warning(
            f"Double occurrence of input data header '{search_text}': USING THE FIRST"
        )

    if matches.any():
        # Get the integer position, not the index label
        return matches.argmax()

    return -1
