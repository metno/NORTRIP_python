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
        - 14-character bitstring like "11110000000010"
        - Comma-separated 0/1 list like "1,1,1,1,1,0,0,0,0,0,0,0,1,0"
        """
        if not isinstance(value, str):
            raise argparse.ArgumentTypeError("--plot-figure must be a string")

        text = value.strip().lower()
        num_flags = 14

        if text == "all":
            return [1] * num_flags
        if text == "none":
            return [0] * num_flags

        # Comma-separated list
        if "," in text:
            try:
                flags = [int(x.strip()) for x in text.split(",")]
            except ValueError as exc:
                raise argparse.ArgumentTypeError(
                    "--plot-figure must contain only 0 or 1"
                ) from exc
            if any(x not in (0, 1) for x in flags):
                raise argparse.ArgumentTypeError("--plot-figure values must be 0 or 1")
            if len(flags) != num_flags:
                raise argparse.ArgumentTypeError(
                    f"--plot-figure must have exactly {num_flags} values"
                )
            return flags

        # Bitstring of 0/1 characters
        if set(text).issubset({"0", "1"}):
            if len(text) != num_flags:
                raise argparse.ArgumentTypeError(
                    f"--plot-figure bitstring must be length {num_flags}"
                )
            return [int(ch) for ch in text]

        raise argparse.ArgumentTypeError(
            "Invalid --plot-figure format. Use 'all', 'none', a 14-digit 0/1 bitstring, "
            "or a comma-separated list of 14 zeros/ones."
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
        "-pl",
        "--plot-figure",
        dest="plot_figure",
        type=parse_plot_figure_arg,
        default=parse_plot_figure_arg("all"),
        metavar="SPEC",
        help=(
            "Which plots to generate. One of: 'all' (default), 'none', a 14-digit 0/1 bitstring, "
            "or a comma-separated list of 14 zeros/ones (e.g. 1,1,1,1,1,0,0,0,0,0,0,0,1,0)."
        ),
    )

    return parser
