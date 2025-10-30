import argparse
import sys


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
        - "all" -> enable all 11 plots
        - "none" -> disable all plots
        - "normal" -> enable all plots except temperature and moisture
        - "temperature" -> enable temperature and moisture plots
        - 11-character bitstring like "11110000010"
        """

        text = value.strip().lower()
        num_flags = 11

        if text == "list":
            print("Available plots:")
            print("  1: Traffic")
            print("  2: Meteorology")
            print("  3: Emissions and mass balance")
            print("  4: Road wetness")
            print("  5: Other factors")
            print("  6: Energy and moisture balance")
            print("  7: Concentrations")
            print("  8: AE")
            print("  9: Scatter/QQ")
            print(" 10: Scatter temperature and moisture")
            print(" 11: Summary")
            print("\nUse --plot with one of:")
            print("  'all' - enable all plots")
            print("  'none' - disable all plots")
            print("  'normal' - enable plots 1-7,9,11 (default)")
            print("  'summary' - enable summary plot (figure 11)")
            print("  'temperature' - enable plots 2,4,6,10,11")
            print("  Or a 11-digit binary string (e.g., '11110000010')")
            sys.exit(0)
        if text == "all":
            return [1] * num_flags
        if text == "none":
            return [0] * num_flags
        if text == "normal":
            return [1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1]
        if text == "summary":
            return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        if text == "temperature":
            return [0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1]

        if set(text).issubset({"0", "1"}):
            if len(text) != num_flags:
                raise argparse.ArgumentTypeError(
                    f"--plot bitstring must be length {num_flags}"
                )
            return [int(ch) for ch in text]

        raise argparse.ArgumentTypeError(
            "Invalid --plot format. Use 'all', 'none', 'normal', 'summary', 'temperature', or a 11-digit 0/1 bitstring."
        )

    def validate_path_file(path_value: str) -> str:
        """Ensure provided path points to a .xlsx or .txt file (relative or absolute)."""
        if not isinstance(path_value, str) or not path_value.lower().endswith(
            (".xlsx", ".txt")
        ):
            raise argparse.ArgumentTypeError(
                "Path must be a relative or absolute path to a .xlsx or .txt file"
            )
        return path_value

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
        "paths",
        type=validate_path_file,
        help="Relative or absolute path to an .xlsx file containing the model file paths. (Eg. model_paths/model_paths.xlsx)",
    )
    parser.add_argument(
        "-pl",
        "--plot",
        dest="plot",
        type=parse_plot_figure_arg,
        default=parse_plot_figure_arg("normal"),
        metavar="<plot type>",
        help="""
            Which plots to generate. One of:
            'all',
            'none',
            'normal' (default),
            'summary',
            'temperature',
            a 11-digit 0/1 bitstring. (eg. '11110000010')
            or 'list' to see available plots.
        """,
    )

    return parser
