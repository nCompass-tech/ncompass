#!/usr/bin/env python3
"""
Nsight Systems profiling example for PyTorch neural network training.

This example demonstrates how to:
1. Profile PyTorch training using NVIDIA Nsight Systems (nsys)
2. Generate .nsys-rep profiling reports
3. Optionally convert traces to Chrome trace format (.json.gz) for visualization

Usage:
    # Basic profiling (generates .nsys-rep file)
    python main.py

    # Profile with custom parameters
    python main.py --epochs 20 --hidden-size 1024 --output my_profile

    # Profile and auto-convert to Chrome trace (gzip-compressed JSON)
    python main.py --convert

    # Specify trace types to capture
    python main.py --trace-types cuda,nvtx,osrt,cudnn,cublas

    # Enable NVTX range capture mode
    python main.py --with-range

    # Disable Python/PyTorch tracing (enabled by default)
    python main.py --no-python-tracing

    # Full profiling with all options (python tracing enabled by default)
    python main.py --with-range --convert
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from ncompass.trace.core.rewrite import enable_rewrites
from ncompass.trace.core.pydantic import RewriteConfig
from ncompass.trace.infra.utils import logger
from ncompass.trace.converters import convert_nsys_report as ncompass_convert_nsys_report, ConversionOptions

logger.setLevel(logging.DEBUG)


def check_nsys_available() -> bool:
    """Check if nsys CLI is available in PATH."""
    try:
        result = subprocess.run(
            ["nsys", "--version"],
            capture_output=True,
            text=True,
            check=True
        )
        logger.info(f"Found nsys: {result.stdout.strip()}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def run_nsys_profile(
    script_path: Path,
    output_name: str,
    trace_dir: Path,
    trace_types: str = "cuda,nvtx,osrt,cudnn,cublas,opengl,cudla",
    epochs: int = 10,
    hidden_size: int = 512,
    force_overwrite: bool = True,
    sample: str = "process-tree",
    session_new: str = "nc0",
    gpuctxsw: bool = True,
    cuda_graph_trace: str = "node",
    show_output: bool = True,
    stop_on_exit: bool = True,
    gpu_metrics_devices: str = "all",
    cuda_memory_usage: bool = True,
    trace_fork_before_exec: bool = True,
    with_range: bool = False,
    python_tracing: bool = True,
    cache_dir: Optional[str] = None,
) -> Optional[Path]:
    """
    Run nsys profile on the training script.
    
    Args:
        script_path: Path to the training script (simplenet.py)
        output_name: Base name for output file (without extension)
        trace_dir: Directory to store trace output files
        trace_types: Comma-separated trace types (cuda,nvtx,osrt,cudnn,cublas,opengl,cudla)
        epochs: Number of training epochs
        hidden_size: Hidden layer size for the network
        force_overwrite: Whether to overwrite existing output files
        sample: Sampling mode (e.g., 'process-tree')
        session_new: Name for new profiling session
        gpuctxsw: Enable GPU context switch tracing
        cuda_graph_trace: CUDA graph trace mode ('node' or 'graph')
        show_output: Show profiled application output
        stop_on_exit: Stop profiling when application exits
        gpu_metrics_devices: GPU devices for metrics collection ('all' or specific)
        cuda_memory_usage: Enable CUDA memory usage tracking
        trace_fork_before_exec: Trace forked processes before exec
        with_range: Enable NVTX range capture mode
        python_tracing: Enable Python/PyTorch tracing (default: True)
        cache_dir: Directory for nCompass cache (default: .cache in script directory)
    
    Returns:
        Path to the generated .nsys-rep file, or None if profiling failed
    """
    # Build the nsys profile command with output path in trace directory
    output_path = trace_dir / output_name
    cmd = [
        "sudo", "nsys", "profile",
        f"--trace={trace_types}",
        f"--output={output_path}",
        f"--sample={sample}",
        f"--session-new={session_new}",
        f"--gpuctxsw={str(gpuctxsw).lower()}",
        f"--cuda-graph-trace={cuda_graph_trace}",
        f"--show-output={str(show_output).lower()}",
        f"--stop-on-exit={str(stop_on_exit).lower()}",
        f"--gpu-metrics-devices={gpu_metrics_devices}",
        f"--cuda-memory-usage={str(cuda_memory_usage).lower()}",
        f"--trace-fork-before-exec={str(trace_fork_before_exec).lower()}",
    ]
    
    if force_overwrite:
        cmd.append("--force-overwrite=true")
    
    # NVTX range capture mode
    if with_range:
        cmd.extend([
            "--capture-range=nvtx",
            "--nvtx-capture=nc_start_capture",
            "--env-var=NSYS_NVTX_PROFILER_REGISTER_ONLY=0",
            "--capture-range-end=repeat",
        ])
    
    # Python/PyTorch tracing
    if python_tracing:
        cmd.extend([
            "--cudabacktrace=kernel",
            "--python-backtrace=cuda",
            "--pytorch=functions-trace",
            "--python-sampling=true",
        ])
    
    # Run the runner script that loads rewrites and executes simplenet
    runner_script = script_path.parent / "runners" / "run_simplenet.py"
    runner_args = [
        sys.executable,
        str(runner_script),
        f"--epochs={epochs}",
        f"--hidden-size={hidden_size}",
    ]
    if with_range:
        runner_args.append("--with-nvtx-capture")
    if cache_dir:
        runner_args.append(f"--cache-dir={cache_dir}")
    
    cmd.extend(runner_args)
    
    logger.info(f"Running nsys profile command:")
    logger.info(f"  {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            cwd=script_path.parent,
        )
        
        # Find the generated .nsys-rep file in trace directory
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


def convert_nsys_report(
    nsys_rep_file: Path,
    output_json: Optional[str] = None,
    output_dir: Optional[Path] = None,
) -> Optional[Path]:
    """
    Convert nsys report to Chrome trace JSON using ncompass library.
    
    Args:
        nsys_rep_file: Path to the .nsys-rep file
        output_json: Optional output JSON filename (without extension)
        output_dir: Optional output directory for generated files
    
    Returns:
        Path to the generated .json.gz file, or None if conversion failed
    """
    # Use the nsys-rep file's directory as output dir if not specified
    if output_dir is None:
        output_dir = nsys_rep_file.parent
    
    # Determine output filename
    output_base = output_json if output_json else nsys_rep_file.stem
    json_file = output_dir / f"{output_base}.json.gz"
    
    logger.info(f"Converting nsys report to Chrome trace...")
    logger.info(f"  Input: {nsys_rep_file}")
    logger.info(f"  Output: {json_file}")
    
    try:
        # Create conversion options with common activity types
        options = ConversionOptions(
            activity_types=["kernel", "nvtx", "nvtx-kernel", "cuda-api", "osrt", "sched"],
            include_metadata=True
        )
        
        # Use the ncompass library's unified conversion function
        ncompass_convert_nsys_report(
            nsys_rep_path=str(nsys_rep_file),
            output_path=str(json_file),
            options=options,
            keep_sqlite=False,
        )
        
        if json_file.exists():
            logger.info(f"Conversion completed successfully!")
            logger.info(f"Generated Chrome trace: {json_file}")
            return json_file
        else:
            logger.error(f"Expected JSON file not found: {json_file}")
            return None
            
    except FileNotFoundError as e:
        logger.error(f"Conversion failed: {e}")
        return None
    except RuntimeError as e:
        logger.error(f"Conversion failed: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error during conversion: {e}")
        return None


def setup_ncompass_rewrites(cache_dir: Optional[str] = None) -> bool:
    """
    Set up nCompass SDK rewrites if config exists.
    
    Args:
        cache_dir: Directory for nCompass cache (default: .cache in current directory)
    
    Returns:
        True if rewrites were enabled, False otherwise
    """
    cache_base = cache_dir if cache_dir else f"{os.getcwd()}/.cache"
    rewrite_config = Path(
        f"{cache_base}/ncompass/profiles/.default/NVTX/current/config.json"
    )
    
    if rewrite_config.exists():
        logger.info("Enabling nCompass rewrites...")
        try:
            with rewrite_config.open("r") as f:
                cfg = json.load(f)
                enable_rewrites(config=RewriteConfig.from_dict(cfg))
            return True
        except Exception as e:
            logger.warning(f"Failed to enable rewrites: {e}")
            return False
    else:
        logger.info("No nCompass rewrite config found, skipping rewrites")
        return False


def validate_environment() -> tuple[bool, Optional[Path]]:
    """
    Validate the profiling environment.
    
    Checks nsys availability and locates the training script.
    
    Returns:
        Tuple of (success, script_path). If success is False, script_path is None.
    """
    if not check_nsys_available():
        logger.error(
            "nsys command not found. Please ensure NVIDIA Nsight Systems is installed "
            "and available in your PATH."
        )
        logger.error("Download from: https://developer.nvidia.com/nsight-systems")
        return False, None
    
    script_dir = Path(__file__).parent.absolute()
    script_path = script_dir / "simplenet.py"
    
    if not script_path.exists():
        logger.error(f"Training script not found: {script_path}")
        return False, None
    
    return True, script_path


def create_trace_directory(script_dir: Path) -> tuple[Path, str]:
    """
    Create a timestamped trace directory under .traces/.
    
    Args:
        script_dir: Directory where the script is located
        
    Returns:
        Tuple of (trace_directory_path, timestamp_string)
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    trace_dir = script_dir / ".traces" / timestamp
    trace_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created trace directory: {trace_dir}")
    return trace_dir, timestamp


def generate_output_name(base_name: Optional[str] = None, timestamp: Optional[str] = None) -> str:
    """
    Generate output filename for profiling session.
    
    Args:
        base_name: Optional user-provided base name
        timestamp: Optional timestamp string to use if base_name not provided
        
    Returns:
        Output name (with timestamp if base_name not provided)
    """
    if base_name is not None:
        return base_name
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"simplenet_profile_{timestamp}"


def log_session_config(
    epochs: int,
    hidden_size: int,
    output: str,
    trace_dir: Path,
    trace_types: str,
    convert: bool,
) -> None:
    """
    Log profiling session configuration.
    
    Args:
        epochs: Number of training epochs
        hidden_size: Hidden layer size for the network
        output: Base name for output files
        trace_dir: Directory to store trace output files
        trace_types: Comma-separated trace types to capture
        convert: Whether to auto-convert to Chrome trace JSON
    """
    logger.info("=" * 80)
    logger.info("Starting nsys profiling session")
    logger.info("=" * 80)
    logger.info(f"  Epochs: {epochs}")
    logger.info(f"  Hidden size: {hidden_size}")
    logger.info(f"  Output: {output}")
    logger.info(f"  Trace directory: {trace_dir}")
    logger.info(f"  Trace types: {trace_types}")
    logger.info(f"  Auto-convert: {convert}")
    logger.info("=" * 80)


def run_profiling_session(
    script_path: Path,
    output: str,
    trace_dir: Path,
    trace_types: str,
    epochs: int,
    hidden_size: int,
    force: bool,
    with_range: bool,
    python_tracing: bool,
    cache_dir: Optional[str] = None,
) -> Optional[Path]:
    """
    Execute the nsys profiling session.
    
    Args:
        script_path: Path to the training script
        output: Base name for output files
        trace_dir: Directory to store trace output files
        trace_types: Comma-separated trace types to capture
        epochs: Number of training epochs
        hidden_size: Hidden layer size for the network
        force: Whether to overwrite existing output files
        with_range: Enable NVTX range capture mode
        python_tracing: Enable Python/PyTorch tracing
        cache_dir: Directory for nCompass cache (default: .cache in script directory)
        
    Returns:
        Path to the generated .nsys-rep file, or None if profiling failed
    """
    nsys_rep_file = run_nsys_profile(
        script_path=script_path,
        output_name=output,
        trace_dir=trace_dir,
        trace_types=trace_types,
        epochs=epochs,
        hidden_size=hidden_size,
        force_overwrite=force,
        with_range=with_range,
        python_tracing=python_tracing,
        cache_dir=cache_dir,
    )
    
    if nsys_rep_file is None:
        logger.error("Profiling failed!")
        return None
    
    logger.info("-" * 80)
    logger.info("Profiling complete!")
    logger.info(f"  nsys report: {nsys_rep_file}")
    
    return nsys_rep_file


def handle_conversion(
    nsys_rep_file: Path,
    output: str,
    trace_dir: Path,
    convert: bool,
) -> Optional[Path]:
    """
    Handle optional conversion to Chrome trace format.
    
    Args:
        nsys_rep_file: Path to the .nsys-rep file
        output: Base name for output files
        trace_dir: Directory to store trace output files
        convert: Whether to perform conversion
        
    Returns:
        Path to the generated .json.gz file, or None if not converted or failed
    """
    if not convert:
        return None
    
    logger.info("-" * 80)
    json_file = convert_nsys_report(nsys_rep_file, output, output_dir=trace_dir)
    if json_file is None:
        logger.warning("Warning: Conversion to Chrome trace failed")
    return json_file


def log_session_summary(
    nsys_rep_file: Path,
    json_file: Optional[Path],
) -> None:
    """
    Log final session summary.
    
    Args:
        nsys_rep_file: Path to the generated .nsys-rep file
        json_file: Path to the generated JSON file, or None
    """
    logger.info("=" * 80)
    logger.info("Session complete!")
    logger.info(f"  nsys report: {nsys_rep_file}")
    if json_file:
        logger.info(f"  Chrome trace: {json_file}")
    else:
        logger.info("  Run with --convert to generate Chrome trace JSON")
    logger.info("=" * 80)


def convert_only(
    input_file: str,
    output_dir: Optional[str] = None,
) -> int:
    """
    Convert-only mode: convert existing .nsys-rep file without profiling.
    
    Args:
        input_file: Path to existing .nsys-rep file
        output_dir: Optional output directory for converted file
        
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    script_dir = Path(__file__).parent.absolute()
    
    # Resolve input file path
    input_path = Path(input_file)
    if not input_path.is_absolute():
        input_path = script_dir / input_file
    
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return 1
    
    if not input_path.suffix == ".nsys-rep":
        logger.warning(f"Input file should be a .nsys-rep file. Got: {input_path}")
    
    # Resolve output directory
    if output_dir:
        out_dir = Path(output_dir)
        if not out_dir.is_absolute():
            out_dir = script_dir / output_dir
    else:
        out_dir = input_path.parent
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("Convert-only mode")
    logger.info("=" * 80)
    logger.info(f"  Input: {input_path}")
    logger.info(f"  Output dir: {out_dir}")
    
    # Convert using existing function
    json_file = convert_nsys_report(input_path, output_dir=out_dir)
    
    if json_file is None:
        logger.error("Conversion failed!")
        return 1
    
    logger.info("=" * 80)
    logger.info("Conversion complete!")
    logger.info(f"  Chrome trace: {json_file}")
    logger.info("=" * 80)
    
    return 0


def main(
    epochs: int = 10,
    hidden_size: int = 512,
    output: Optional[str] = None,
    trace_types: str = "cuda,nvtx,osrt,cudnn,cublas,opengl,cudla",
    convert: bool = False,
    force: bool = True,
    with_range: bool = False,
    python_tracing: bool = True,
    cache_dir: Optional[str] = None,
) -> int:
    """
    Run nsys profiling on SimpleNet training.
    
    Args:
        epochs: Number of training epochs
        hidden_size: Hidden layer size for the network
        output: Base name for output files (auto-generated if not provided)
        trace_types: Comma-separated trace types to capture
        convert: Whether to auto-convert to Chrome trace JSON
        force: Whether to overwrite existing output files
        with_range: Enable NVTX range capture mode
        python_tracing: Enable Python/PyTorch tracing (default: True)
        cache_dir: Directory for nCompass cache (default: .cache in script directory)
        
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    # Validate environment
    valid, script_path = validate_environment()
    if not valid:
        return 1
    
    # Note: nCompass rewrites are now loaded by runners/run_simplenet.py
    # so they apply in the subprocess that nsys profiles
    
    # Create timestamped trace directory
    trace_dir, timestamp = create_trace_directory(script_path.parent)
    
    # Generate output name (use timestamp for consistency)
    output = generate_output_name(output, timestamp)
    
    # Log session configuration
    log_session_config(epochs, hidden_size, output, trace_dir, trace_types, convert)
    
    # Run profiling
    nsys_rep_file = run_profiling_session(
        script_path=script_path,
        output=output,
        trace_dir=trace_dir,
        trace_types=trace_types,
        epochs=epochs,
        hidden_size=hidden_size,
        force=force,
        with_range=with_range,
        python_tracing=python_tracing,
        cache_dir=cache_dir,
    )
    
    if nsys_rep_file is None:
        return 1
    
    # Handle conversion
    json_file = handle_conversion(nsys_rep_file, output, trace_dir, convert)
    
    # Log summary
    log_session_summary(nsys_rep_file, json_file)
    
    return 0


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Profile PyTorch neural network training with NVIDIA Nsight Systems",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic profiling
  python main.py

  # Profile with custom parameters
  python main.py --epochs 20 --hidden-size 1024

  # Profile and auto-convert to Chrome trace
  python main.py --convert

  # Specify output name and trace types
  python main.py --output my_profile --trace-types cuda,nvtx,osrt,cudnn

  # Enable NVTX range capture mode
  python main.py --with-range

  # Disable Python/PyTorch tracing (enabled by default)
  python main.py --no-python-tracing

  # Full profiling with all options (python tracing enabled by default)
  python main.py --with-range --convert

  # Convert-only mode: convert existing .nsys-rep file without profiling
  python main.py --convert-only --input my_profile.nsys-rep --output-dir ./output
        """
    )
    
    parser.add_argument(
        "--epochs", type=int, default=10,
        help="Number of training epochs (default: 10)"
    )
    parser.add_argument(
        "--hidden-size", type=int, default=512,
        help="Hidden layer size for the network (default: 512)"
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Base name for output files (auto-generated if not provided)"
    )
    parser.add_argument(
        "--trace-types", "-t", type=str, default="cuda,nvtx,osrt,cudnn,cublas,opengl,cudla",
        help="Comma-separated trace types: cuda,nvtx,osrt,cudnn,cublas,opengl,cudla (default: cuda,nvtx,osrt,cudnn,cublas,opengl,cudla)"
    )
    parser.add_argument(
        "--convert", "-c", action="store_true",
        help="Auto-convert nsys report to Chrome trace format (.json.gz)"
    )
    parser.add_argument(
        "--no-force", action="store_true",
        help="Don't overwrite existing output files"
    )
    parser.add_argument(
        "--with-range", action="store_true",
        help="Enable NVTX range capture mode (--capture-range=nvtx, --nvtx-capture=nc_start_capture)"
    )
    parser.add_argument(
        "--no-python-tracing", action="store_true",
        help="Disable Python/PyTorch tracing (enabled by default)"
    )
    parser.add_argument(
        "--convert-only", action="store_true",
        help="Convert-only mode: convert existing .nsys-rep file without profiling"
    )
    parser.add_argument(
        "--input", "-i", type=str, default=None,
        help="Input .nsys-rep file path (required for --convert-only mode)"
    )
    parser.add_argument(
        "--output-dir", "-d", type=str, default=None,
        help="Output directory for converted files (used with --convert-only)"
    )
    parser.add_argument(
        "--cache-dir", type=str, default=None,
        help="Directory for nCompass cache (default: .cache in script directory)"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Handle convert-only mode
    if args.convert_only:
        if not args.input:
            logger.error("--convert-only requires --input to specify the .nsys-rep file")
            sys.exit(1)
        sys.exit(convert_only(
            input_file=args.input,
            output_dir=args.output_dir,
        ))
    
    # Normal profiling mode
    sys.exit(main(
        epochs=args.epochs,
        hidden_size=args.hidden_size,
        output=args.output,
        trace_types=args.trace_types,
        convert=args.convert,
        force=not args.no_force,
        with_range=args.with_range,
        python_tracing=not args.no_python_tracing,
        cache_dir=args.cache_dir,
    ))
