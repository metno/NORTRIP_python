import argparse


def create_arg_parser() -> argparse.ArgumentParser:
    """
    Create an argument parser for the NORTRIP Road Dust Model.

    Returns:
        argparse.ArgumentParser: Configured argument parser.
    """
    parser = argparse.ArgumentParser(description="NORTRIP Road Dust Model")

    def parse_plot_figure_arg(value: str) -> list[int]:
        """Parse plot figure selection from CLI.

        Supported forms:
        - "all" -> enable all 14 plots
        - "none" -> disable all plots
        - "normal" -> enable all plots except temperature and moisture
        - "temperature" -> enable temperature and moisture plots
        - 14-character bitstring like "11110000000010"
        - Comma-separated 0/1 list like "1,1,1,1,1,0,0,0,0,0,0,0,1,0"
        """

        text = value.strip().lower()
        num_flags = 14

        if text == "all":
            return [1] * num_flags
        if text == "none":
            return [0] * num_flags
        if text == "normal":
            return [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0]
        if text == "temperature":
            return [0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1]

        if set(text).issubset({"0", "1"}):
            if len(text) != num_flags:
                raise argparse.ArgumentTypeError(
                    f"--plot bitstring must be length {num_flags}"
                )
            return [int(ch) for ch in text]

        raise argparse.ArgumentTypeError(
            "Invalid --plot format. Use 'all', 'none', 'normal', 'temperature', or a 14-digit 0/1 bitstring."
        )

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
    parser.add_argument(
        "-l",
        "--log",
        action="store_false",
        help="Disables logging.",
    )

    parser.add_argument(
        "-sp",
        "--save-plots",
        action="store_true",
        help="Save plots to output_figures directory.",
    )

    parser.add_argument(
        "-pl",
        "--plot",
        dest="plot",
        type=parse_plot_figure_arg,
        default=parse_plot_figure_arg("normal"),
        metavar="SPEC",
        help=(
            "Which plots to generate. One of: 'all', 'none', 'normal' (default), 'temperature', a 14-digit 0/1 bitstring. (eg. '11110000000010')"
        ),
    )

    return parser
