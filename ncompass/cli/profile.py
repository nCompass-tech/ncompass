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
nCompass CLI - Profile command.

Runs nsys or ncu profiling on any command with nCompass instrumentation.
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path

from ncompass.profile import (
    check_nsys_available,
    check_ncu_available,
    create_trace_directory,
    run_nsys_profile,
    run_ncu_profile,
    convert_ncu_to_csv,
)
from ncompass.profile.nsys import NsysDefaults 
from ncompass.trace.converters import convert_nsys_report, ConversionOptions
from ncompass.trace.infra.utils import logger


def add_profile_parser(
    subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    """Add the profile subcommand parser.

    Args:
        subparsers: Subparsers action from parent parser

    Returns:
        The profile subparser
    """
    # Format default nsys args for help text
    defaults_help = "\n".join(f"        {k}={v}" for k, v in NsysDefaults().to_dict().items())

    parser = subparsers.add_parser(
        "profile",
        help="Run nsys or ncu profiling on any command",
        description="Profile any command using NVIDIA Nsight Systems (nsys) or Nsight Compute (ncu).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
    # Profile with nsys (default settings)
    ncompass profile --nsys -- python my_script.py

    # Profile with nsys and auto-convert to Chrome trace
    ncompass profile --nsys --convert -- python train.py

    # Profile with nsys and custom trace types (overrides default)
    ncompass profile --nsys --trace=cuda,nvtx -- python my_script.py

    # Profile with nsys and additional arguments
    ncompass profile --nsys --cuda-memory-usage=true -- python my_script.py

    # Profile with ncu
    ncompass profile --ncu -- python my_script.py

Default nsys arguments (can be overridden):
{defaults_help}

Note:
    - Either --nsys or --ncu is required
    - All ncompass options must appear BEFORE the -- separator
    - Everything after -- is the command to profile
    - Any nsys/ncu arguments can be passed and will override defaults
        """,
    )

    # Profiler selection (mutually exclusive, one required)
    profiler_group = parser.add_mutually_exclusive_group(required=True)
    profiler_group.add_argument(
        "--nsys",
        action="store_true",
        help="Profile with NVIDIA Nsight Systems (nsys)",
    )
    profiler_group.add_argument(
        "--ncu",
        action="store_true",
        help="Profile with NVIDIA Nsight Compute (ncu)",
    )

    # Output options
    output_group = parser.add_argument_group("Output options")
    output_group.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Base name for output files (auto-generated if not provided)",
    )
    output_group.add_argument(
        "--output-dir",
        "-d",
        type=str,
        default=None,
        help="Directory to store output files (default: .nsys_traces/<timestamp>)",
    )
    output_group.add_argument(
        "--convert",
        "-c",
        action="store_true",
        help="Auto-convert nsys report to Chrome trace format (.json.gz)",
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

    parser.set_defaults(func=run_profile_command)

    return parser


def _configure_logging(args: argparse.Namespace) -> None:
    """Configure logging level based on verbosity flags."""
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.ERROR)
    else:
        logger.setLevel(logging.INFO)


def _check_profiler_availability(use_ncu: bool) -> bool:
    """Check if the required profiler (nsys or ncu) is available."""
    if use_ncu:
        if not check_ncu_available():
            logger.error(
                "ncu command not found. Please ensure NVIDIA Nsight Compute is installed "
                "and available in your PATH."
            )
            logger.error("Download from: https://developer.nvidia.com/nsight-compute")
            return False
    else:
        if not check_nsys_available():
            logger.error(
                "nsys command not found. Please ensure NVIDIA Nsight Systems is installed "
                "and available in your PATH."
            )
            logger.error("Download from: https://developer.nvidia.com/nsight-systems")
            return False
    return True


def _resolve_session_paths(
    args: argparse.Namespace, user_command: list[str]
) -> tuple[Path, Path, str]:
    """Resolve working directory, trace directory, and output filename."""
    working_dir = Path.cwd()

    # Determine output directory
    if args.output_dir:
        base_dir = Path(args.output_dir).absolute()
        trace_dir = base_dir
        trace_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        trace_dir, timestamp = create_trace_directory(working_dir)

    # Generate output name
    if args.output:
        output_name = args.output
    else:
        first_cmd = Path(user_command[0]).stem
        output_name = f"{first_cmd}_profile_{timestamp}"

    return working_dir, trace_dir, output_name


def _execute_ncu_session(
    user_command: list[str],
    output_name: str,
    trace_dir: Path,
    working_dir: Path,
    extra_args: list[str],
) -> int:
    """Execute an NCU profiling session."""
    logger.info("=" * 80)
    logger.info("Starting ncompass NCU profile session")
    logger.info("=" * 80)
    logger.info(f"  Command: {' '.join(user_command)}")
    logger.info(f"  Output: {output_name}")
    logger.info(f"  Trace directory: {trace_dir}")
    if extra_args:
        logger.info(f"  Extra args: {' '.join(extra_args)}")
    logger.info("=" * 80)

    ncu_rep_file = run_ncu_profile(
        command=user_command,
        output_name=output_name,
        trace_dir=trace_dir,
        working_dir=working_dir,
        extra_args=extra_args,
    )

    if ncu_rep_file is None:
        logger.error("Profiling failed!")
        return 1

    # Convert to CSV
    logger.info("-" * 80)
    logger.info("Converting NCU report to CSV...")
    try:
        csv_file = trace_dir / f"{output_name}.csv"
        convert_ncu_to_csv(ncu_rep_file, csv_file)
    except Exception as e:
        logger.warning(f"CSV conversion failed: {e}")
        csv_file = None

    # Log summary
    logger.info("=" * 80)
    logger.info("Session complete!")
    logger.info(f"  ncu report: {ncu_rep_file}")
    if csv_file:
        logger.info(f"  CSV file: {csv_file}")
    logger.info("=" * 80)

    return 0


def _execute_nsys_session(
    args: argparse.Namespace,
    user_command: list[str],
    output_name: str,
    trace_dir: Path,
    working_dir: Path,
    extra_args: list[str],
) -> int:
    """Execute an nsys profiling session."""
    logger.info("=" * 80)
    logger.info("Starting ncompass nsys profile session")
    logger.info("=" * 80)
    logger.info(f"  Command: {' '.join(user_command)}")
    logger.info(f"  Output: {output_name}")
    logger.info(f"  Trace directory: {trace_dir}")
    if extra_args:
        logger.info(f"  Extra args: {' '.join(extra_args)}")
    logger.info("=" * 80)

    nsys_rep_file = run_nsys_profile(
        command=user_command,
        output_name=output_name,
        trace_dir=trace_dir,
        working_dir=working_dir,
        extra_args=extra_args,
    )

    if nsys_rep_file is None:
        logger.error("Profiling failed!")
        return 1

    logger.info("-" * 80)
    logger.info("Profiling complete!")
    logger.info(f"  nsys report: {nsys_rep_file}")

    # Handle conversion
    json_file = None
    if args.convert:
        logger.info("-" * 80)
        logger.info("Converting to Chrome trace format...")

        try:
            json_file = trace_dir / f"{output_name}.json.gz"
            options = ConversionOptions(
                activity_types=[
                    "kernel",
                    "nvtx",
                    "nvtx-kernel",
                    "cuda-api",
                    "osrt",
                    "sched",
                ],
                include_metadata=True,
            )
            convert_nsys_report(
                nsys_rep_path=str(nsys_rep_file),
                output_path=str(json_file),
                options=options,
                keep_sqlite=False,
            )
            logger.info(f"Generated Chrome trace: {json_file}")
        except Exception as e:
            logger.warning(f"Conversion failed: {e}")
            json_file = None

    # Log summary
    logger.info("=" * 80)
    logger.info("Session complete!")
    logger.info(f"  nsys report: {nsys_rep_file}")
    if json_file:
        logger.info(f"  Chrome trace: {json_file}")
    else:
        logger.info("  Run with --convert to generate Chrome trace JSON")
    logger.info("=" * 80)

    return 0


def run_profile_command(args: argparse.Namespace) -> int:
    """Execute the profile command.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    _configure_logging(args)

    # Get user command and extra args from args (set by main.py after parsing)
    user_command: list[str] = getattr(args, "user_command", [])
    extra_args: list[str] = getattr(args, "extra_args", [])

    # Validate command is provided
    if not user_command:
        logger.error("No command specified. Usage: ncompass profile --nsys|--ncu [options] -- <command>")
        return 1

    # Check profiler availability
    if not _check_profiler_availability(args.ncu):
        return 1

    # Determine paths and names
    working_dir, trace_dir, output_name = _resolve_session_paths(args, user_command)

    # Run appropriate session
    if args.ncu:
        return _execute_ncu_session(
            user_command, output_name, trace_dir, working_dir, extra_args
        )
    else:
        return _execute_nsys_session(
            args, user_command, output_name, trace_dir, working_dir, extra_args
        )
