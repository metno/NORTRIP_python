import argparse


def create_arg_parser() -> argparse.ArgumentParser:
    """
    Create an argument parser for the NORTRIP Road Dust Model.

    Returns:
        argparse.ArgumentParser: Configured argument parser.
    """
    parser = argparse.ArgumentParser(description="NORTRIP Road Dust Model")
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
