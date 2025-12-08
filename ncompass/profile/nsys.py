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

import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional

from ncompass.trace.infra.utils import logger


def check_nsys_available() -> bool:
    """Check if nsys CLI is available in PATH.

    Returns:
        True if nsys is found and executable, False otherwise.
    """
    try:
        result = subprocess.run(
            ["nsys", "--version"], capture_output=True, text=True, check=True
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


def run_nsys_profile(
    command: list[str],
    output_name: str,
    trace_dir: Path,
    working_dir: Optional[Path],
    trace_types: str,
    force_overwrite: bool,
    sample: str,
    session_name: str,
    gpuctxsw: bool,
    cuda_graph_trace: str,
    cuda_memory_usage: bool,
    with_range: bool,
    python_tracing: bool,
    use_sudo: bool,
    cache_dir: Optional[str],
) -> Optional[Path]:
    """Run nsys profile on any command.

    Args:
        command: Command and arguments to profile (e.g., ["python", "script.py", "--arg"]).
        output_name: Base name for output files.
        trace_dir: Directory to store trace output.
        working_dir: Working directory for the command (defaults to current directory).
        trace_types: Comma-separated trace types (e.g., "cuda,nvtx,osrt").
        force_overwrite: Whether to overwrite existing output files.
        sample: Sampling mode (e.g., "process-tree").
        session_name: Name for the profiling session.
        gpuctxsw: Enable GPU context switch tracing.
        cuda_graph_trace: CUDA graph trace mode ("node" or "graph").
        cuda_memory_usage: Enable CUDA memory usage tracking.
        with_range: Enable NVTX range capture mode.
        python_tracing: Enable Python/PyTorch tracing.
        use_sudo: Run nsys with sudo.
        cache_dir: Directory for nCompass cache.

    Returns:
        Path to the generated .nsys-rep file, or None if profiling failed.
    """
    output_path = trace_dir / output_name

    # Build the nsys profile command
    cmd: list[str] = []
    if use_sudo:
        cmd.extend(["sudo", "-E"])

    cmd.extend(
        [
            "nsys",
            "profile",
            f"--trace={trace_types}",
            f"--output={output_path}",
            f"--sample={sample}",
            f"--session-new={session_name}",
            f"--gpuctxsw={str(gpuctxsw).lower()}",
            f"--cuda-graph-trace={cuda_graph_trace}",
            "--show-output=true",
            "--stop-on-exit=true",
            "--gpu-metrics-devices=all",
            f"--cuda-memory-usage={str(cuda_memory_usage).lower()}",
            "--trace-fork-before-exec=true",
        ]
    )

    if force_overwrite:
        cmd.append("--force-overwrite=true")

    # NVTX range capture mode
    if with_range:
        cmd.extend(
            [
                "--capture-range=nvtx",
                "--nvtx-capture=nc_start_capture",
                "--env-var=NSYS_NVTX_PROFILER_REGISTER_ONLY=0",
                "--capture-range-end=repeat",
            ]
        )

    # Python/PyTorch tracing
    if python_tracing:
        cmd.extend(
            [
                "--cudabacktrace=kernel",
                "--python-backtrace=cuda",
                "--pytorch=functions-trace",
                "--python-sampling=true",
            ]
        )

    # Set environment variable for nCompass cache if specified
    if cache_dir:
        os.environ["NCOMPASS_CACHE_DIR"] = cache_dir

    # Add the user command
    cmd.extend(command)

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

