#!/usr/bin/env python3
# Copyright 2025 nCompass Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
nCompass CLI - Main entry point.

Usage:
    ncompass profile [OPTIONS] -- COMMAND  Run nsys profiling on a command
    ncompass convert [OPTIONS] INPUT_FILE  Convert nsys report to Chrome trace
    ncompass --version                     Show version
    ncompass --help                        Show help
"""

import argparse
import sys

from ncompass import __version__


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="ncompass",
        description="nCompass SDK - Profiling and trace analysis tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Profile a Python script with nsys
    ncompass profile -- python my_script.py

    # Profile with auto-conversion to Chrome trace
    ncompass profile --convert -- python my_script.py --epochs 10

    # Profile any executable
    ncompass profile -c -- ./my_cuda_app --config config.yaml

    # Convert an existing nsys report to Chrome trace
    ncompass convert my_profile.nsys-rep

    # Convert with custom output directory
    ncompass convert my_profile.nsys-rep --output-dir ./traces

For more information, visit: https://docs.ncompass.tech
        """,
    )

    parser.add_argument(
        "--version",
        "-V",
        action="version",
        version=f"ncompass {__version__}",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        title="commands",
        description="Available commands",
        help="Run 'ncompass <command> --help' for more information",
    )

    # Import and register subcommands
    from ncompass.cli.profile import add_profile_parser
    from ncompass.cli.convert import add_convert_parser

    add_profile_parser(subparsers)
    add_convert_parser(subparsers)

    return parser


def main(args: list[str] | None = None) -> int:
    """Main CLI entry point.

    Args:
        args: Command-line arguments (defaults to sys.argv[1:])

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    if args is None:
        args = sys.argv[1:]

    # Handle the -- separator for profile command
    # Split on -- to separate ncompass args from user command
    if "--" in args:
        separator_idx = args.index("--")
        nc_args = args[:separator_idx]
        user_command = args[separator_idx + 1:]
    else:
        nc_args = args
        user_command = []

    parser = create_parser()
    parsed_args = parser.parse_args(nc_args)

    # Attach user command for profile handler
    parsed_args.user_command = user_command

    if parsed_args.command is None:
        parser.print_help()
        return 0

    # Execute the appropriate command handler
    if hasattr(parsed_args, "func"):
        return parsed_args.func(parsed_args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())

