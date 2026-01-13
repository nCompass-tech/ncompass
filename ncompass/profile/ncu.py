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
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ncompass.trace.infra.utils import logger
from ncompass.profile.config import config


@dataclass
class NcuDefaults:
    """Default ncu arguments for ncompass profiling."""

    target_processes: str = "all"
    nvtx_include: str     = "regex:user_annotated:.*/"
    replay_mode: str      = "application"

    def to_dict(self) -> dict[str, str]:
        """Convert to dictionary with ncu argument format (--key).

        Note: Boolean flags (--nvtx, --force-overwrite) are handled separately
        in _build_ncu_command since they don't take values.
        """
        return {
            "--target-processes": self.target_processes,
            "--nvtx-include": self.nvtx_include,
            "--replay-mode": self.replay_mode,
        }


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


def _parse_ncu_args(args: list[str]) -> dict[str, str]:
    """Parse ncu arguments into a dictionary.

    Handles both --key=value and --key value formats.

    Args:
        args: List of argument strings

    Returns:
        Dictionary mapping argument names to values
    """
    parsed: dict[str, str] = {}
    i = 0
    while i < len(args):
        arg = args[i]
        if arg.startswith("--"):
            if "=" in arg:
                # --key=value format
                key, value = arg.split("=", 1)
                parsed[key] = value
            elif i + 1 < len(args) and not args[i + 1].startswith("-"):
                # --key value format
                parsed[arg] = args[i + 1]
                i += 1
            else:
                # Boolean flag (--key with no value)
                parsed[arg] = "true"
        i += 1
    return parsed


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


def _build_ncu_command(
    output_path: Path,
    metrics_str: str,
    extra_args: list[str],
    command: list[str],
) -> list[str]:
    """Build the ncu profile command.

    Starts with defaults, then applies extra_args (which can override defaults).

    Args:
        output_path: Path for output file (without extension)
        metrics_str: Comma-separated metrics string
        extra_args: Additional ncu arguments (can override defaults)
        command: The command to profile

    Returns:
        Complete ncu command as list of strings
    """
    # Start with defaults
    args_dict = NcuDefaults().to_dict()

    # Add output path and metrics
    args_dict["--export"] = str(output_path)
    args_dict["--metrics"] = metrics_str

    # Parse and apply extra args (overrides defaults)
    extra_parsed = _parse_ncu_args(extra_args)
    args_dict.update(extra_parsed)

    # Build command - start with boolean flags (no value)
    cmd = ["ncu", "--nvtx", "--force-overwrite"]

    # Add key=value arguments
    for key, value in args_dict.items():
        cmd.append(f"{key}={value}")

    # Add the user command
    cmd.extend(command)

    return cmd


def convert_ncu_to_csv(ncu_rep_path: Path, output_csv: Path) -> None:
    """Convert .ncu-rep file to CSV format.

    Uses ncu --import to read the report and output CSV.

    Args:
        ncu_rep_path: Path to the .ncu-rep file
        output_csv: Path to save CSV output
    """
    try:
        # Import the ncu-rep file and export as CSV
        result = subprocess.run(
            ["ncu", "--import", str(ncu_rep_path), "--csv"],
            capture_output=True,
            text=True,
            check=True,
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

        logger.info(f"Converted NCU report to CSV: {output_csv}")

    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"NCU CSV conversion failed: {e.stderr}")
    except Exception as e:
        raise RuntimeError(f"NCU CSV conversion failed: {e}")


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
        raise ValueError(f"No valid metrics found in NCU matching the given list: {metrics_list}")

    metrics_str = ",".join(filtered_metrics)
    logger.info(f"Using {len(filtered_metrics)} metrics")
    return metrics_str

def run_ncu_profile(
    command: list[str],
    output_name: str,
    trace_dir: Path,
    working_dir: Optional[Path] = None,
    extra_args: Optional[list[str]] = None,
) -> Optional[Path]:
    """Run ncu profile on any command.

    Uses ncompass defaults for ncu arguments. Any extra_args will be passed
    through to ncu and can override the defaults.

    Default ncu arguments:
        --nvtx (boolean flag)
        --force-overwrite (boolean flag)
        --target-processes=all
        --nvtx-include=regex:user_annotated:.*/

    Args:
        command: Command and arguments to profile (e.g., ["python", "script.py"]).
        output_name: Base name for output files.
        trace_dir: Directory to store trace output.
        working_dir: Working directory for the command (defaults to current directory).
        extra_args: Additional ncu arguments (can override defaults).

    Returns:
        Path to the generated .ncu-rep file, or None if profiling failed.
    """
    output_path = trace_dir / output_name

    # Query available metrics and filter
    metrics_str = get_metrics_str(list(config.ncu_metrics))

    # Build the ncu command
    cmd = _build_ncu_command(
        output_path=output_path,
        metrics_str=metrics_str,
        extra_args=extra_args or [],
        command=command,
    )

    logger.info("Running ncu profile command:")
    logger.info(f"  {' '.join(cmd)}")

    # Use provided working directory or current directory
    cwd = working_dir if working_dir else Path.cwd()

    try:
        subprocess.run(
            cmd,
            check=True,
            cwd=cwd,
        )

        ncu_rep_file = trace_dir / f"{output_name}.ncu-rep"
        if ncu_rep_file.exists():
            logger.info(f"Generated ncu report: {ncu_rep_file}")
            return ncu_rep_file
        else:
            logger.error(f"Expected output file not found: {ncu_rep_file}")
            return None

    except subprocess.CalledProcessError as e:
        logger.error(f"ncu profile failed with return code {e.returncode}")
        return None
