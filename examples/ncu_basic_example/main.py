#!/usr/bin/env python3
"""
Nsight Compute (NCU) profiling example for PyTorch neural network inference.

This example demonstrates how to:
1. Add NVTX markers via the nCompass VSCode extension (no code changes)
2. Profile PyTorch inference using NVIDIA Nsight Compute (ncu)
3. Generate CSV profiling reports with kernel-level metrics

Prerequisites:
    1. Add NVTX markers using the nCompass VSCode extension
    2. Set environment variables:
       - NCOMPASS_CACHE_DIR=<path to .cache dir created when adding NVTX markers>
       - NCOMPASS_PROFILER_TYPE=NVTX
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from ncompass.profile import check_ncu_available, run_ncu_profile as ncu_profile

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def build_simplenet_command(
    runner_path: Path,
    batch: int,
    iters: int,
    warmup: int,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    precision: str,
    python_bin: str = sys.executable,
) -> list[str]:
    """
    Build the command to run the simplenet inference script via the rewriter runner.

    Args:
        runner_path: Path to the runner script (run_simplenet.py)
        batch: Batch size for inference
        iters: Number of profiling iterations
        warmup: Number of warmup iterations
        input_dim: Input dimension for the model
        hidden_dim: Hidden layer dimension
        output_dim: Output dimension
        precision: Precision (fp32 or fp16)
        python_bin: Path to python binary

    Returns:
        Command as list of strings
    """
    return [
        python_bin,
        str(runner_path),
        f"--batch={batch}",
        f"--iters={iters}",
        f"--warmup={warmup}",
        f"--input-dim={input_dim}",
        f"--hidden-dim={hidden_dim}",
        f"--output-dim={output_dim}",
        f"--precision={precision}",
    ]


def setup_ncompass_path(ncompass_dir: str) -> None:
    """
    Add ncompass directory to PYTHONPATH environment variable.

    Args:
        ncompass_dir: Path to ncompass package directory
    """
    ncompass_path = Path(ncompass_dir).resolve()
    if not ncompass_path.exists():
        raise FileNotFoundError(f"ncompass directory not found: {ncompass_path}")

    current_pythonpath = os.environ.get("PYTHONPATH", "")
    if current_pythonpath:
        os.environ["PYTHONPATH"] = f"{ncompass_path}:{current_pythonpath}"
    else:
        os.environ["PYTHONPATH"] = str(ncompass_path)

    logger.info(f"Added ncompass to PYTHONPATH: {ncompass_path}")


def validate_environment() -> tuple[bool, Optional[Path]]:
    """
    Validate the profiling environment.

    Checks ncu availability and locates the inference script.

    Returns:
        Tuple of (success, script_path). If success is False, script_path is None.
    """
    if not check_ncu_available():
        logger.error(
            "ncu command not found. Please ensure NVIDIA Nsight Compute is installed "
            "and available in your PATH."
        )
        logger.error("Download from: https://developer.nvidia.com/nsight-compute")
        return False, None

    script_dir = Path(__file__).parent.absolute()
    script_path = script_dir / "simplenet.py"

    if not script_path.exists():
        logger.error(f"Inference script not found: {script_path}")
        return False, None

    return True, script_path


def setup_output_directory(script_dir: Path) -> tuple[Path, str]:
    """
    Prepare the .ncu_traces/ directory and return the timestamp.
    
    Deletes any existing files in .ncu_traces/ before returning.
    
    Args:
        script_dir: Directory where the script is located
        
    Returns:
        Tuple of (output_directory_path, timestamp_string)
    """
    output_dir = script_dir / ".ncu_traces"
    
    # Create directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Delete existing files in the directory
    for item in output_dir.iterdir():
        if item.is_file():
            item.unlink()
        elif item.is_dir():
            import shutil
            shutil.rmtree(item)
            
    logger.info(f"Cleaned output directory: {output_dir}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_dir, timestamp


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
    return f"simplenet_ncu_{timestamp}"


def main(
    batch: int = 512,
    iters: int = 20,
    warmup: int = 5,
    input_dim: int = 1024,
    hidden_dim: int = 2048,
    output_dim: int = 512,
    precision: str = "fp32",
    output: Optional[str] = None,
    kernel_name: str = "",
) -> int:
    """
    Run NCU profiling on SimpleNet inference.

    Args:
        batch: Batch size for inference
        iters: Number of profiling iterations
        warmup: Number of warmup iterations
        input_dim: Input dimension for the model
        hidden_dim: Hidden layer dimension
        output_dim: Output dimension
        precision: Precision (fp32 or fp16)
        output: Base name for output files (auto-generated if not provided)
        kernel_name: Optional regex pattern to filter kernel names

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    # Validate environment
    valid, script_path = validate_environment()
    if not valid or script_path is None:
        return 1

    # Setup output directory and get timestamp
    output_dir, timestamp = setup_output_directory(script_path.parent)

    # Generate output name
    output_name = generate_output_name(output, timestamp)

    # Use the runner script to enable rewrites
    runner_path = script_path.parent / "runners" / "run_simplenet.py"
    if not runner_path.exists():
        logger.error(f"Runner script not found: {runner_path}")
        return 1

    # Build the command to profile
    command = build_simplenet_command(
        runner_path=runner_path,
        batch=batch,
        iters=iters,
        warmup=warmup,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        precision=precision,
    )

    # Log session configuration
    logger.info("=" * 80)
    logger.info("Starting NCU profiling session")
    logger.info("=" * 80)
    logger.info(f"  Batch size: {batch}")
    logger.info(f"  Iterations: {iters} (warmup: {warmup})")
    logger.info(f"  Model dims: input={input_dim}, hidden={hidden_dim}, output={output_dim}")
    logger.info(f"  Precision: {precision}")
    logger.info(f"  Output: {output_name}")
    logger.info(f"  Output dir: {output_dir}")
    logger.info("=" * 80)

    # Run profiling using ncompass library
    try:
        t1 = time.time()
        csv_file = ncu_profile(
            command=command,
            output_name=output_name,
            trace_dir=output_dir,
            working_dir=script_path.parent,
            kernel_name=kernel_name,
            nvtx_include="regex:@user_annotated:.*/", # Includes all nvtx ranges
        )
        t2 = time.time()
        logger.info(f"Profiling time: {(t2 - t1):.1f} seconds")
    except Exception as e:
        logger.error(f"Profiling failed: {e}")
        return 1

    # Log summary
    logger.info("=" * 80)
    logger.info("Session complete!")
    logger.info(f"  NCU report:\n    {csv_file}\n")
    logger.info("=" * 80)

    return 0


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Profile PyTorch neural network inference with NVIDIA Nsight Compute",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic profiling 
  python main.py

  # Profile with custom parameters
  python main.py --iters 30 --hidden-dim 4096

  # Profile with fp16 precision
  python main.py --precision fp16

  # Filter specific kernels
  python main.py --kernel-name "regex:.*gemm.*"
        """
    )

    parser.add_argument(
        "--batch", type=int, default=512,
        help="Batch size for inference (default: 512)"
    )
    parser.add_argument(
        "--iters", type=int, default=20,
        help="Number of profiling iterations (default: 20)"
    )
    parser.add_argument(
        "--warmup", type=int, default=5,
        help="Number of warmup iterations (default: 5)"
    )
    parser.add_argument(
        "--input-dim", type=int, default=1024,
        help="Input dimension (default: 1024)"
    )
    parser.add_argument(
        "--hidden-dim", type=int, default=2048,
        help="Hidden layer dimension (default: 2048)"
    )
    parser.add_argument(
        "--output-dim", type=int, default=512,
        help="Output dimension (default: 512)"
    )
    parser.add_argument(
        "--precision", type=str, default="fp32", choices=["fp32", "fp16"],
        help="Precision (default: fp32)"
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Base name for output files (auto-generated if not provided)"
    )
    parser.add_argument(
        "--kernel-name", "-k", type=str, default="",
        help="Regex pattern to filter kernel names (e.g., 'regex:.*gemm.*')"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sys.exit(main(
        batch=args.batch,
        iters=args.iters,
        warmup=args.warmup,
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        precision=args.precision,
        output=args.output,
        kernel_name=args.kernel_name,
    ))
