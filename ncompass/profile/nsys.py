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
nCompass Profiling - Nsight Systems (nsys) integration.

Provides functions for running nsys profiling on any command.
"""

import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from ncompass.trace.infra.utils import logger


@dataclass
class NsysDefaults:
    """Default nsys arguments for ncompass profiling."""

    trace: str                  = "cuda,nvtx,osrt,cudnn,cublas,opengl,cudla"
    sample: str                 = "process-tree"
    gpuctxsw: str               = "true"
    cuda_graph_trace: str       = "node"
    stop_on_exit: str           = "true"
    trace_fork_before_exec: str = "true"
    force_overwrite: str        = "true"
    capture_range: str          = "cudaProfilerApi"
    capture_range_end: str      = "repeat"

    def to_dict(self) -> dict[str, str]:
        """Convert to dictionary with nsys argument format (--key)."""
        return {
            "--trace": self.trace,
            "--sample": self.sample,
            "--gpuctxsw": self.gpuctxsw,
            "--cuda-graph-trace": self.cuda_graph_trace,
            "--stop-on-exit": self.stop_on_exit,
            "--trace-fork-before-exec": self.trace_fork_before_exec,
            "--force-overwrite": self.force_overwrite,
            "--capture-range": self.capture_range,
            "--capture-range-end": self.capture_range_end,
        }


def detect_nsys_sudo_needed() -> bool:
    """Quick test to detect if nsys profiling requires sudo.

    Runs a minimal nsys profile on the 'true' command to check if
    elevated privileges are needed. Uses the same default trace settings
    as actual profiling for an accurate test.

    Returns:
        True if sudo appears to be needed, False otherwise.
    """
    import tempfile

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            test_output = Path(tmpdir) / "ncompass_sudo_test"
            cmd = _build_nsys_command(
                output_path=test_output,
                extra_args=["--sample", "none"],
                command=["true"],
                sudo=False,
            )
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=15,
            )
            if result.returncode == 0:
                return False

            err = (result.stderr + result.stdout).lower()
            permission_keywords = [
                "permission",
                "denied",
                "privilege",
                "not allowed",
                "root",
                "requires root",
            ]
            if any(kw in err for kw in permission_keywords):
                logger.info("nsys requires elevated privileges for profiling")
                return True

            return False

    except subprocess.TimeoutExpired:
        logger.debug("nsys sudo detection timed out")
        return False
    except (FileNotFoundError, OSError):
        return False


def check_nsys_available(sudo: bool = False) -> bool:
    """Check if nsys CLI is available in PATH.

    Args:
        sudo: If True, prepend sudo to the command.

    Returns:
        True if nsys is found and executable, False otherwise.
    """
    try:
        cmd = ["nsys", "--version"]
        if sudo:
            cmd = ["sudo"] + cmd
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True
        )
        logger.info(f"Found nsys: {result.stdout.strip()}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def create_trace_directory(base_dir: Path) -> tuple[Path, str]:
    """Create a timestamped trace directory.

    Args:
        base_dir: Base directory for traces

    Returns:
        Tuple of (trace_directory_path, timestamp_string)
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    trace_dir = base_dir / ".nsys_traces" / timestamp
    trace_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created trace directory: {trace_dir}")
    return trace_dir, timestamp


def _parse_nsys_args(args: list[str]) -> dict[str, str]:
    """Parse nsys arguments into a dictionary.

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


def _build_nsys_command(
    output_path: Path,
    extra_args: list[str],
    command: list[str],
    sudo: bool = False,
) -> list[str]:
    """Build the nsys profile command.

    Starts with defaults, then applies extra_args (which can override defaults).

    Args:
        output_path: Path for output file (without extension)
        extra_args: Additional nsys arguments (can override defaults)
        command: The command to profile
        sudo: If True, prepend sudo to the command.

    Returns:
        Complete nsys command as list of strings
    """
    # Start with defaults
    args_dict = NsysDefaults().to_dict()

    # Add output path
    args_dict["--output"] = str(output_path)

    # Parse and apply extra args (overrides defaults)
    extra_parsed = _parse_nsys_args(extra_args)
    args_dict.update(extra_parsed)

    # Build command
    cmd = ["nsys", "profile"]
    for key, value in args_dict.items():
        cmd.append(f"{key}={value}")

    # Add the user command
    cmd.extend(command)

    if sudo:
        cmd = ["sudo"] + cmd

    return cmd


def run_nsys_profile(
    command: list[str],
    output_name: str,
    trace_dir: Path,
    working_dir: Optional[Path] = None,
    extra_args: Optional[list[str]] = None,
    sudo: bool = False,
) -> Optional[Path]:
    """Run nsys profile on any command.

    Uses ncompass defaults for nsys arguments. Any extra_args will be passed
    through to nsys and can override the defaults.

    Default nsys arguments:
        --trace=cuda,nvtx,osrt,cudnn,cublas,opengl,cudla
        --sample=process-tree
        --gpuctxsw=true
        --cuda-graph-trace=node
        --stop-on-exit=true
        --trace-fork-before-exec=true
        --force-overwrite=true
        --capture-range=nvtx
        --nvtx-capture=ncompass_nsys_range
        --capture-range-end=repeat

    Args:
        command: Command and arguments to profile (e.g., ["python", "script.py"]).
        output_name: Base name for output files.
        trace_dir: Directory to store trace output.
        working_dir: Working directory for the command (defaults to current directory).
        extra_args: Additional nsys arguments (can override defaults).
        sudo: If True, run nsys with sudo.

    Returns:
        Path to the generated .nsys-rep file, or None if profiling failed.
    """
    output_path = trace_dir / output_name

    # Build the nsys command
    cmd = _build_nsys_command(
        output_path=output_path,
        extra_args=extra_args or [],
        command=command,
        sudo=sudo,
    )

    logger.info("Running nsys profile command:")
    logger.info(f"  {' '.join(cmd)}")

    # Use provided working directory or current directory
    cwd = working_dir if working_dir else Path.cwd()

    try:
        subprocess.run(
            cmd,
            check=True,
            cwd=cwd,
        )

        nsys_rep_file = trace_dir / f"{output_name}.nsys-rep"
        if nsys_rep_file.exists():
            logger.info(f"Generated nsys report: {nsys_rep_file}")
            return nsys_rep_file
        else:
            logger.error(f"Expected output file not found: {nsys_rep_file}")
            return None

    except subprocess.CalledProcessError as e:
        logger.error(f"nsys profile failed with return code {e.returncode}")
        return None
