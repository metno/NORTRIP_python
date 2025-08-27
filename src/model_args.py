import argparse


def create_arg_parser() -> argparse.ArgumentParser:
    """
    Create an argument parser for the NORTRIP Road Dust Model.

    Returns:
        argparse.ArgumentParser: Configured argument parser.
    """
    parser = argparse.ArgumentParser(description="NORTRIP Road Dust Model")

    def validate_xlsx_path(path_value: str) -> str:
        """Ensure provided path points to a .xlsx file (relative or absolute)."""
        if not isinstance(path_value, str) or not path_value.lower().endswith(".xlsx"):
            raise argparse.ArgumentTypeError(
                "Path must be a relative or absolute path to a .xlsx file"
            )
        return path_value

    parser.add_argument(
        "paths",
        type=validate_xlsx_path,
        help="Relative or absolute path to an .xlsx file containing the model file paths. (Eg. model_paths/model_paths.xlsx)",
    )
    parser.add_argument(
        "-t",
        "--text",
        action="store_true",
        help="Read paths as text file instead of Excel.",
    )
    parser.add_argument(
        "-p",
        "--print",
        action="store_true",
        help="Print model results to terminal.",
    )
    parser.add_argument(
        "-f",
        "--fortran",
        action="store_true",
        help="Run fortran model. (WIP)",
    )

    return parser
