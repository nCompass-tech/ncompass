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

Runs nsys profiling on any command with nCompass instrumentation.
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path

from ncompass.profile import (
    check_nsys_available,
    create_trace_directory,
    run_nsys_profile,
)
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
    parser = subparsers.add_parser(
        "profile",
        help="Run nsys profiling on any command",
        description="Profile any command using NVIDIA Nsight Systems with nCompass instrumentation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic profiling of a Python script
    ncompass profile -- python my_script.py

    # Profile with auto-conversion to Chrome trace
    ncompass profile --convert -- python train.py --epochs 10

    # Profile with custom trace types
    ncompass profile --trace-types cuda,nvtx -- python my_script.py

    # Profile any executable (not just Python)
    ncompass profile -c -- ./my_cuda_app --config config.yaml

    # Profile a shell script
    ncompass profile -- bash run_training.sh

    # Enable NVTX range capture mode
    ncompass profile --with-range -- python my_script.py

    # Run with sudo (for full system profiling features)
    ncompass profile --sudo -- mpirun -np 4 python distributed.py

Note: All ncompass options must appear BEFORE the -- separator.
      Everything after -- is the command to profile.
        """,
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
        help="Directory to store output files (default: .traces/<timestamp> in current directory)",
    )
    output_group.add_argument(
        "--convert",
        "-c",
        action="store_true",
        help="Auto-convert nsys report to Chrome trace format (.json.gz)",
    )

    # Trace options
    trace_group = parser.add_argument_group("Trace options")
    trace_group.add_argument(
        "--trace-types",
        "-t",
        type=str,
        default="cuda,nvtx,osrt,cudnn,cublas,opengl,cudla",
        help="Comma-separated trace types (default: cuda,nvtx,osrt,cudnn,cublas,opengl,cudla)",
    )
    trace_group.add_argument(
        "--no-nc-range",
        action="store_true",
        help="Disable profiling only within nc_start_capture NVTX range",
    )
    trace_group.add_argument(
        "--python-tracing",
        action="store_true",
        help="Disable Python/PyTorch tracing (enabled by default)",
    )
    trace_group.add_argument(
        "--cuda-graph-trace",
        type=str,
        default="node",
        choices=["node", "graph"],
        help="CUDA graph trace mode (default: node)",
    )

    # Advanced options
    advanced_group = parser.add_argument_group("Advanced options")
    advanced_group.add_argument(
        "--sample",
        type=str,
        default="process-tree",
        help="Sampling mode (default: process-tree)",
    )
    advanced_group.add_argument(
        "--session-name",
        type=str,
        default="nc0",
        help="Name for profiling session (default: nc0)",
    )
    advanced_group.add_argument(
        "--no-force",
        action="store_true",
        help="Don't overwrite existing output files",
    )
    advanced_group.add_argument(
        "--no-gpu-ctx-switch",
        action="store_true",
        help="Disable GPU context switch tracing",
    )
    advanced_group.add_argument(
        "--no-cuda-memory-usage",
        action="store_true",
        help="Disable CUDA memory usage tracking",
    )
    advanced_group.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory for nCompass cache (default: .cache in current directory)",
    )
    advanced_group.add_argument(
        "--no-sudo",
        action="store_true",
        help="Run nsys with sudo (enables full system profiling features)",
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


def run_profile_command(args: argparse.Namespace) -> int:
    """Execute the profile command.

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

    # Get user command from args (set by main.py after parsing)
    user_command: list[str] = getattr(args, "user_command", [])

    # Validate command is provided
    if not user_command:
        logger.error("No command specified. Usage: ncompass profile [options] -- <command>")
        return 1

    # Check nsys availability
    if not check_nsys_available():
        logger.error(
            "nsys command not found. Please ensure NVIDIA Nsight Systems is installed "
            "and available in your PATH."
        )
        logger.error("Download from: https://developer.nvidia.com/nsight-systems")
        return 1

    # Determine working directory - use current directory
    working_dir = Path.cwd()

    # Determine output directory
    if args.output_dir:
        base_dir = Path(args.output_dir).absolute()
        trace_dir = base_dir
        trace_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        trace_dir, timestamp = create_trace_directory(working_dir)

    # Generate output name from first command element or use provided
    if args.output:
        output_name = args.output
    else:
        # Extract base name from first command element
        first_cmd = Path(user_command[0]).stem
        output_name = f"{first_cmd}_profile_{timestamp}"

    # Log configuration
    logger.info("=" * 80)
    logger.info("Starting ncompass profile session")
    logger.info("=" * 80)
    logger.info(f"  Command: {' '.join(user_command)}")
    logger.info(f"  Output: {output_name}")
    logger.info(f"  Trace directory: {trace_dir}")
    logger.info(f"  Trace types: {args.trace_types}")
    logger.info(f"  Python tracing: {args.python_tracing}")
    logger.info(f"  Auto-convert: {args.convert}")
    logger.info(f"  Using sudo: {not args.no_sudo}")
    logger.info("=" * 80)

    # Run profiling
    nsys_rep_file = run_nsys_profile(
        command=user_command,
        output_name=output_name,
        trace_dir=trace_dir,
        working_dir=working_dir,
        trace_types=args.trace_types,
        force_overwrite=not args.no_force,
        sample=args.sample,
        session_name=args.session_name,
        gpuctxsw=not args.no_gpu_ctx_switch,
        cuda_graph_trace=args.cuda_graph_trace,
        cuda_memory_usage=not args.no_cuda_memory_usage,
        with_range=not args.no_nc_range,
        python_tracing=args.python_tracing,
        use_sudo=not args.no_sudo,
        cache_dir=args.cache_dir,
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

