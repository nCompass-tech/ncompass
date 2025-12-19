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
nCompass Profiling - Nsight Compute (ncu) integration.

Provides functions for running ncu profiling on any command.
"""

import subprocess
from pathlib import Path

from ncompass.trace.infra.utils import logger
from ncompass.profile.config import config


def check_ncu_available() -> bool:
    """Check if ncu CLI is available in PATH.

    Returns:
        True if ncu is found and executable, False otherwise.
    """
    try:
        result = subprocess.run(
            ["ncu", "--version"], capture_output=True, text=True, check=True
        )
        logger.info(f"Found ncu: {result.stdout.strip()}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def query_ncu_metrics(ncu_bin: str = "ncu") -> set[str]:
    """Query available base metrics from NCU.

    Args:
        ncu_bin: Path to ncu binary

    Returns:
        Set of available base metric names
    """
    try:
        result = subprocess.run(
            [ncu_bin, "--query-metrics"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=True,
        )
        # Parse output - first column is metric name
        metrics = set()
        for line in result.stdout.splitlines():
            line = line.strip()
            if line:
                # Get first word (metric name)
                metric = line.split()[0]
                metrics.add(metric)
        return metrics
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.warning("Failed to query NCU metrics")
        return set()


def filter_available_metrics(
    metrics_list: list[str], available_base_metrics: set[str]
) -> tuple[list[str], list[str]]:
    """Filter metrics list based on available base metrics.

    Args:
        metrics_list: List of metrics to filter
        available_base_metrics: Set of available base metric names

    Returns:
        Tuple of (filtered_metrics, missing_metrics)
    """
    filtered_metrics = []
    missing_metrics = []

    for metric in metrics_list:
        # Extract base metric name (before first dot)
        base = metric.split(".")[0]
        if base in available_base_metrics:
            filtered_metrics.append(metric)
        else:
            missing_metrics.append(metric)

    return filtered_metrics, missing_metrics


def build_ncu_command(
    ncu_bin: str,
    kernel_name: str,
    nvtx_include: str,
    metrics_str: str,
    command: list[str],
) -> list[str]:
    """Build the NCU command line.

    Args:
        ncu_bin: Path to ncu binary
        kernel_name: Kernel name filter (empty for all kernels)
        nvtx_include: NVTX range filter (empty for no filter)
        metrics_str: Comma-separated metrics string
        command: Command to profile

    Returns:
        Complete NCU command as list of strings
    """
    ncu_cmd = [
        ncu_bin,
        "--kernel-name", kernel_name,
        "--target-processes", "all",
        "--nvtx",
    ]

    # Add nvtx-include if specified
    if nvtx_include:
        ncu_cmd.extend(["--nvtx-include", nvtx_include])

    ncu_cmd.extend([
        "--metrics", metrics_str,
        "--csv",
        "--force-overwrite",
    ])

    # Add the command to profile
    ncu_cmd.extend(command)

    return ncu_cmd


def run_ncu_and_parse_output(
    ncu_cmd: list[str], working_dir: Path, output_csv: Path
) -> None:
    """Run NCU command and parse CSV output.

    Args:
        ncu_cmd: Complete NCU command
        working_dir: Working directory to run command in
        output_csv: Path to save CSV output

    Returns:
        True if successful, False otherwise
    """
    try:
        # Run NCU and capture output
        result = subprocess.run(
            ncu_cmd,
            cwd=working_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        # Filter output to extract only CSV data
        # Look for the CSV header line starting with "ID","Process ID"
        lines = result.stdout.splitlines()
        csv_started = False
        csv_lines = []

        for line in lines:
            if not csv_started and line.startswith('"ID","Process ID"'):
                csv_started = True
            if csv_started:
                csv_lines.append(line)

        if not csv_lines:
            logger.error("No valid CSV data found in NCU output")
            logger.error(f"NCU output:\n{result.stdout}")
            raise ValueError(f"No valid CSV data found in NCU output: {result.stdout}")

        # Write CSV to file
        with open(output_csv, "w") as f:
            f.write("\n".join(csv_lines))

        logger.info(f"NCU profiling complete: {output_csv}")

    except Exception as e:
        raise RuntimeError(f"NCU execution failed: {e}")


def get_metrics_str(metrics_list: list[str], ncu_bin: str = "ncu") -> str:
    """Get the metrics string for the given metrics list.

    Args:
        metrics_list: List of metrics
        ncu_bin: Path to ncu binary

    Returns:
        Metrics string
    """
    logger.info("Querying available base metrics from NCU...")
    available_base_metrics = query_ncu_metrics(ncu_bin)

    filtered_metrics, missing_metrics = filter_available_metrics(
        metrics_list, available_base_metrics
    )

    if missing_metrics:
        logger.debug("Skipping missing base metrics:")
        for metric in missing_metrics:
            logger.debug(f"  {metric}")

    if not filtered_metrics:
        raise ValueError("No valid metrics found in NCU matching the given list: {metrics_list}")

    metrics_str = ",".join(filtered_metrics)
    logger.info(f"Using {len(filtered_metrics)} metrics")
    return metrics_str

def run_ncu_profile(
    command: list[str],
    output_name: str,
    trace_dir: Path,
    working_dir: Path,
    kernel_name: str = "",
    nvtx_include: str = "",
    ncu_bin: str = "ncu",
) -> Path:
    """Run NCU profiling on the given command.

    Args:
        command: Command to profile
        output_name: Base name for output file
        trace_dir: Directory to store output files
        working_dir: Working directory to run command in
        kernel_name: Kernel name filter (empty for all kernels)
        nvtx_include: NVTX range filter (empty for no filter)
        ncu_bin: Path to ncu binary

    Returns:
        Path to output CSV file, or None if profiling failed
    """
    # Query available metrics and filter
    metrics_str = get_metrics_str(config.ncu_metrics, ncu_bin)

    # Build NCU command
    ncu_cmd = build_ncu_command(
        ncu_bin=ncu_bin,
        kernel_name=kernel_name,
        nvtx_include=nvtx_include,
        metrics_str=metrics_str,
        command=command,
    )

    # Output CSV file
    output_csv = trace_dir / f"{output_name}.csv"

    logger.info(f"Running NCU command: {' '.join(ncu_cmd)}")
    logger.info(f"Working directory: {working_dir}")

    # Run NCU and parse output
    run_ncu_and_parse_output(ncu_cmd, working_dir, output_csv)

    return output_csv