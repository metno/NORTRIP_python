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
        help="Relative or absolute path to a .xlsx file. (Eg. model_paths/model_paths.xlsx)",
    )
    parser.add_argument(
        "-t",
        "--text",
        type=int,
        choices=[0, 1],
        default=0,
        help="Read as text mode (0=False, 1=True). Default is 0.",
    )
    parser.add_argument(
        "-p",
        "--print",
        type=int,
        choices=[0, 1],
        default=0,
        help="Print model results to terminal (0=False, 1=True). Default is 0.",
    )
    parser.add_argument(
        "-f",
        "--fortran",
        type=int,
        choices=[0, 1],
        default=0,
        help="Run fortran model (0=False, 1=True). Default is 0.",
    )

    return parser
