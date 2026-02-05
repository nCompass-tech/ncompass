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

import json
import os
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
    clock_control: str    = "none"

    def to_dict(self) -> dict[str, str]:
        """Convert to dictionary with ncu argument format (--key)."""
        return {
            "--target-processes": self.target_processes,
            "--nvtx-include":     self.nvtx_include,
            "--clock-control":    self.clock_control,
        }


def load_ncu_kernel_targets(cache_dir: Optional[Path] = None, trace_file_name: Optional[str] = None) -> list[dict]:
    """Load NCU kernel targets from profile config.

    Reads the config.json file from the NCU profile directory.
    Uses the same directory structure as injector profiles:
    .cache/ncompass/profiles/{profile}/NCU/{trace_file}/current/config.json

    Args:
        cache_dir: Base directory containing .cache/ncompass/profiles.
                   Defaults to current directory or NCOMPASS_CACHE_DIR env var.
        trace_file_name: Specific trace file to load targets for.
                        If None, tries NCOMPASS_TRACE_NAME env var first,
                        then returns targets from all trace files combined.

    Returns:
        List of kernel target dicts with 'kernel_name' and 'instance_number' keys.
        Empty list if no targets file exists.
    """
    if cache_dir is None:
        # Try environment variable first, then current directory
        cache_dir = Path(os.environ.get("NCOMPASS_CACHE_DIR", "."))

    # If no trace_file_name provided, try NCOMPASS_TRACE_NAME env var
    if trace_file_name is None:
        trace_file_name = os.environ.get("NCOMPASS_TRACE_NAME")

    ncu_base = cache_dir / ".cache" / "ncompass" / "profiles" / ".default" / "NCU"

    if trace_file_name:
        # Load targets for specific trace file
        targets_path = ncu_base / trace_file_name / "current" / "config.json"
        if not targets_path.exists():
            return []
        try:
            with open(targets_path, 'r') as f:
                data = json.load(f)
            return data.get("targets", [])
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to load NCU kernel targets: {e}")
            return []
    else:
        # Load targets from all trace files
        all_targets = []
        if ncu_base.exists():
            for trace_dir in ncu_base.iterdir():
                if trace_dir.is_dir():
                    targets_path = trace_dir / "current" / "config.json"
                    if targets_path.exists():
                        try:
                            with open(targets_path, 'r') as f:
                                data = json.load(f)
                            all_targets.extend(data.get("targets", []))
                        except (json.JSONDecodeError, OSError):
                            pass  # Skip corrupted files
        return all_targets


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


def build_kernel_id_regex(kernel_targets: list[dict]) -> Optional[str]:
    """Build a single --kernel-id regex that matches all kernel targets.

    NCU doesn't support multiple --kernel-id flags to target different
    kernel/instance combinations. Instead, we build a single regex that
    matches the cross-product of all kernel names and instance numbers.

    Format: ::regex:^(kernel_a|kernel_b)$:(1|2|3)

    This allows profiling multiple specific kernel instances in a single run.

    Args:
        kernel_targets: List of kernel targets, each with 'kernel_name' and
                       'instance_number' keys.

    Returns:
        Kernel-id regex string for use with --kernel-id flag, or None if
        no targets provided.

    Example:
        >>> targets = [
        ...     {"kernel_name": "gemm", "instance_number": 1},
        ...     {"kernel_name": "gemm", "instance_number": 3},
        ...     {"kernel_name": "conv", "instance_number": 2},
        ... ]
        >>> build_kernel_id_regex(targets)
        "::regex:^(conv|gemm)$:(1|2|3)"
    """
    if not kernel_targets:
        return None

    # Collect unique kernel names and instance numbers
    kernel_names = set()
    instance_numbers = set()

    for target in kernel_targets:
        kernel_name = target.get("kernel_name", "")
        instance_number = target.get("instance_number", 1)
        if kernel_name:
            kernel_names.add(kernel_name)
            instance_numbers.add(str(instance_number))

    if not kernel_names or not instance_numbers:
        return None

    # Build regex pattern
    # Format: ::regex:^(kernel1|kernel2)$:(1|2|3)
    kernel_pattern = "|".join(sorted(kernel_names))
    instance_pattern = "|".join(sorted(instance_numbers, key=int))

    return f"::regex:^({kernel_pattern})$:({instance_pattern})"


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
    kernel_targets: Optional[list[dict]] = None,
) -> list[str]:
    """Build the ncu profile command.

    Starts with defaults, then applies extra_args (which can override defaults).
    If kernel_targets are provided, builds a single --kernel-id regex flag that
    matches all specified kernel/instance combinations.

    Args:
        output_path: Path for output file (without extension)
        metrics_str: Comma-separated metrics string
        extra_args: Additional ncu arguments (can override defaults)
        command: The command to profile
        kernel_targets: Optional list of kernel targets, each with 'kernel_name'
                       and 'instance_number' keys. NCU will profile only these
                       specific kernel instances using a cross-product regex.

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

    # Add kernel targets if specified using cross-product regex
    # NCU format: --kernel-id ::regex:^(kernel_a|kernel_b)$:(1|2)
    if kernel_targets:
        kernel_id_regex = build_kernel_id_regex(kernel_targets)
        if kernel_id_regex:
            cmd.append(f"--kernel-id={kernel_id_regex}")

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
        # --page raw outputs columnar format (metrics as columns, one row per kernel)
        # Without it, NCU outputs row-based format (each metric as separate row)
        result = subprocess.run(
            ["ncu", "--import", str(ncu_rep_path), "--csv", "--page", "raw"],
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
    use_kernel_targets: bool = True,
) -> Optional[Path]:
    """Run ncu profile on any command.

    Uses ncompass defaults for ncu arguments. Any extra_args will be passed
    through to ncu and can override the defaults.

    Default ncu arguments:
        --force-overwrite (boolean flag)
        --target-processes=all
        --profile-from-start=off
        --clock-control=none

    If use_kernel_targets is True and kernel targets are configured in the
    profile, only those specific kernel instances will be profiled using
    NCU's --kernel-id flag.

    Args:
        command: Command and arguments to profile (e.g., ["python", "script.py"]).
        output_name: Base name for output files.
        trace_dir: Directory to store trace output.
        working_dir: Working directory for the command (defaults to current directory).
        extra_args: Additional ncu arguments (can override defaults).
        use_kernel_targets: Whether to load and apply kernel targets from config.
                           Set to False to profile all kernels. Defaults to True.

    Returns:
        Path to the generated .ncu-rep file, or None if profiling failed.
    """
    output_path = trace_dir / output_name

    # Query available metrics and filter
    metrics_str = get_metrics_str(list(config.ncu_metrics))

    # Load kernel targets if enabled
    kernel_targets = None
    if use_kernel_targets:
        kernel_targets = load_ncu_kernel_targets(working_dir)
        if kernel_targets:
            logger.info(f"Using {len(kernel_targets)} kernel target(s) from profile config:")
            for target in kernel_targets:
                logger.info(f"  - {target.get('kernel_name')} (instance #{target.get('instance_number')})")

    # Build the ncu command
    cmd = _build_ncu_command(
        output_path=output_path,
        metrics_str=metrics_str,
        extra_args=extra_args or [],
        command=command,
        kernel_targets=kernel_targets,
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
