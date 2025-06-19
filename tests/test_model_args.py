import argparse
from src.model_args import create_arg_parser


def test_create_arg_parser_returns_parser_instance():
    parser = create_arg_parser()
    assert isinstance(parser, argparse.ArgumentParser)


def test_create_arg_parser_has_expected_arguments():
    parser = create_arg_parser()
    args = parser.parse_args(
        []
    )  # Parse with no arguments to check default values and existence

    assert hasattr(args, "text")
    assert args.text == 0  # Default value
    assert hasattr(args, "print")
    assert args.print == 0  # Default value

    # Test if arguments can be set
    args_text = parser.parse_args(["-t", "1"])
    assert args_text.text == 1
    args_print = parser.parse_args(["-p", "1"])
    assert args_print.print == 1
