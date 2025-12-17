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
nCompass CLI - Convert command.

Converts nsys reports to Chrome trace format.
"""

import argparse
import logging
from pathlib import Path

from ncompass.trace.infra.utils import logger
from ncompass.trace.converters import convert_nsys_report, ConversionOptions


def add_convert_parser(
    subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    """Add the convert subcommand parser.

    Args:
        subparsers: Subparsers action from parent parser

    Returns:
        The convert subparser
    """
    parser = subparsers.add_parser(
        "convert",
        help="Convert nsys report to Chrome trace format",
        description="Convert an NVIDIA Nsight Systems report (.nsys-rep) to Chrome trace format (.json.gz).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic conversion
    ncompass convert my_profile.nsys-rep

    # Convert with custom output directory
    ncompass convert my_profile.nsys-rep --output-dir ./traces

    # Convert with custom output filename
    ncompass convert my_profile.nsys-rep --output custom_name

    # Keep intermediate SQLite file
    ncompass convert my_profile.nsys-rep --keep-sqlite

    # Select specific activity types
    ncompass convert my_profile.nsys-rep --activity-types kernel,nvtx,cuda-api
        """,
    )

    # Required arguments
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the .nsys-rep file to convert",
    )

    # Output options
    output_group = parser.add_argument_group("Output options")
    output_group.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output filename (without extension, auto-generated if not provided)",
    )
    output_group.add_argument(
        "--output-dir",
        "-d",
        type=str,
        default=None,
        help="Output directory (default: same as input file)",
    )
    output_group.add_argument(
        "--keep-sqlite",
        action="store_true",
        help="Keep intermediate SQLite file",
    )

    # Conversion options
    conv_group = parser.add_argument_group("Conversion options")
    conv_group.add_argument(
        "--activity-types",
        "-a",
        type=str,
        default="kernel,nvtx,nvtx-kernel,cuda-api,osrt,sched",
        help="Comma-separated activity types to include (default: kernel,nvtx,nvtx-kernel,cuda-api,osrt,sched)",
    )
    conv_group.add_argument(
        "--no-metadata",
        action="store_true",
        help="Exclude metadata events from output",
    )

    # Verbosity
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress non-error output",
    )
    
    parser.add_argument(
        "--python-fallback",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use Python backend for conversion (default: False)",
    )

    parser.set_defaults(func=run_convert_command)
    return parser


def run_convert_command(args: argparse.Namespace) -> int:
    """Execute the convert command.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    # Configure logging
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.ERROR)
    else:
        logger.setLevel(logging.INFO)

    # Validate input file
    input_path = Path(args.input_file).absolute()
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return 1

    if input_path.suffix != ".nsys-rep":
        logger.warning(f"Input file should be a .nsys-rep file. Got: {input_path}")

    # Determine output directory and filename
    if args.output_dir:
        output_dir = Path(args.output_dir).absolute()
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = input_path.parent

    if args.output:
        output_name = args.output
    else:
        output_name = input_path.stem

    output_path = output_dir / f"{output_name}.json.gz"

    # Parse activity types
    activity_types = [t.strip() for t in args.activity_types.split(",")]

    # Log configuration
    logger.info("=" * 80)
    logger.info("ncompass convert")
    logger.info("=" * 80)
    logger.info(f"  Input: {input_path}")
    logger.info(f"  Output: {output_path}")
    logger.info(f"  Activity types: {activity_types}")
    logger.info(f"  Include metadata: {not args.no_metadata}")
    logger.info("=" * 80)

    # Perform conversion
    try:
        options = ConversionOptions(
            activity_types=activity_types,
            include_metadata=not args.no_metadata,
        )

        convert_nsys_report(
            nsys_rep_path=str(input_path),
            output_path=str(output_path),
            options=options,
            keep_sqlite=args.keep_sqlite,
            use_rust=(not args.python_fallback),
        )

        logger.info("=" * 80)
        logger.info("Conversion complete!")
        logger.info(f"  Chrome trace: {output_path}")
        if args.keep_sqlite:
            sqlite_path = input_path.with_suffix(".sqlite")
            logger.info(f"  SQLite file: {sqlite_path}")
        logger.info("=" * 80)

        return 0

    except FileNotFoundError as e:
        logger.error(f"Conversion failed: {e}")
        return 1
    except RuntimeError as e:
        logger.error(f"Conversion failed: {e}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error during conversion: {e}")
        return 1

