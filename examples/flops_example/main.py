"""
FLOP counting example for PyTorch neural network training with nCompass SDK integration.

This example demonstrates how to:
1. Integrate nCompass SDK for tracing and instrumentation
2. Train a simple neural network with FLOP counting
3. Count FLOPs (floating point operations) for PyTorch code
4. Use TorchFlopCounterContext for parametric FLOP analysis
5. Display FLOP counts for different tensor sizes

Usage:
    python main.py
    python main.py --size-multipliers "0.5,1,2" --steps 5
    python main.py --steps 3 --epochs 5
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
import json
import logging

from ncompass.trace.core.rewrite import enable_rewrites
from ncompass.trace.core.pydantic import RewriteConfig
from ncompass.trace.infra.utils import logger
from ncompass.trace.profile.torch import TorchFlopCounterContext
from simplenet import train_simple_network

logger.setLevel(logging.DEBUG)

# PROFILING_TARGETS defines which functions should be instrumented with trace markers.
# This configuration tells ncompass to automatically wrap specific code regions with
# profiling contexts that will appear in PyTorch profiler traces.

def profile(
    label: Optional[str] = None,
    steps: int = 3,
    schedule: Optional[Dict[str, int]] = None,
    record_shapes: bool = True,
    profile_memory: bool = False,
    with_stack: bool = True,
    print_rows: int = 0,
    profiling_targets: Optional[Dict[str, Any]] = None,
    trace_dir: str = ".traces",
    link_annotations: bool = True,
    verbose: bool = False,
    cache_dir: Optional[str] = None,
    size_multipliers: Optional[list] = None,
    scale_dims: Optional[tuple] = None,
    **kwargs,
):
    """
    Count FLOPs for neural network training using TorchFlopCounterContext with parametric sizing.
    
    Args:
        label: Optional label for the run (note: not used for output directory with FlopCounter)
        steps: Number of training steps per size configuration (default: 3)
        schedule: Not used with FlopCounter (kept for API compatibility)
        record_shapes: Not used with FlopCounter (kept for API compatibility)
        profile_memory: Not used with FlopCounter (kept for API compatibility)
        with_stack: Not used with FlopCounter (kept for API compatibility)
        print_rows: Number of rows to print in summary (limited functionality with FlopCounter)
        profiling_targets: Custom profiling targets config (uses PROFILING_TARGETS by default)
        trace_dir: Not used with FlopCounter (kept for API compatibility)
        cache_dir: Directory for nCompass cache (default: .cache in current directory)
        size_multipliers: List of scaling factors for tensor dimensions
        scale_dims: Tuple of dimension indices to scale (default: (0,) for batch dimension only)
                   Note: For neural networks, typically only scale batch dimension to avoid 
                   breaking model input/output shapes
        **kwargs: Arguments to pass to train_simple_network (epochs, hidden_size)
    """
    logger.info("Starting parametric FLOP counting session...")
    
    # Initialize nCompass SDK with profiling targets
    cache_base = cache_dir if cache_dir else f"{os.getcwd()}/.cache"
    rewrite_config = \
            Path(f"{cache_base}/ncompass/profiles/.default/Torch/current/config.json")
    if rewrite_config.exists():
        logger.info("Enabling nCompass rewrites...")
        with rewrite_config.open("r") as f:
            cfg = json.load(f)
            enable_rewrites(config=RewriteConfig.from_dict(cfg))
    
    # Log device availability
    import torch
    if torch.cuda.is_available():
        logger.info("CUDA available - FLOP counting will include CUDA operations")
    else:
        logger.info("Running on CPU - FLOP counting will include CPU operations")
    
    # Prepare training data
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 128
    X = torch.randn(batch_size, 784, device=device)
    y = torch.randint(0, 10, (batch_size,), device=device)
    
    logger.info(f"Running parametric FLOP counting with {steps} steps per size configuration...")
    
    # Run parametric profiling with FlopCounter
    # Note: Only scale batch dimension (0) by default to avoid breaking model input shape
    result = None
    actual_scale_dims = scale_dims if scale_dims is not None else (0,)
    with TorchFlopCounterContext(
        name="train_simple_network",
        var_name="X",
        size_multipliers=size_multipliers,
        scale_dims=actual_scale_dims
    ) as ctx:
        # Run the training function multiple times with different tensor sizes
        # Generate new y tensor matching X's batch size (works for both smaller and larger batches)
        ctx.run(lambda: train_simple_network(
            X=X, 
            y=torch.randint(0, 10, (X.shape[0],), device=X.device), 
            **kwargs
        ))
    
    logger.info("Parametric FLOP counting complete!")
    
    # Note: TorchFlopCounterContext does not generate trace files like torch.profiler
    # FLOP counts are displayed directly to stdout during execution
    logger.info("FLOP counting does not generate trace files - results are displayed above")
    
    return None, result


def main(
    label: Optional[str] = None,
    steps: int = 3,
    record_shapes: bool = True,
    profile_memory: bool = False,
    with_stack: bool = True,
    print_rows: int = 10,
    trace_dir: str = ".traces",
    epochs: int = 10,
    hidden_size: int = 512,
    custom_config_path: Optional[str] = None,
    no_link: bool = False,
    verbose: bool = False,
    link_only: Optional[str] = None,
    cache_dir: Optional[str] = None,
    size_multipliers: Optional[str] = None,
    scale_dims: Optional[str] = None,
):
    """
    Run parametric FLOP counting from the command line.
    
    Args:
        label: Optional label (not used with FlopCounter)
        steps: Number of training steps per size configuration
        record_shapes: Not used with FlopCounter (kept for API compatibility)
        profile_memory: Not used with FlopCounter (kept for API compatibility)
        with_stack: Not used with FlopCounter (kept for API compatibility)
        print_rows: Number of rows to print in summary (limited with FlopCounter)
        trace_dir: Not used with FlopCounter (kept for API compatibility)
        epochs: Number of training epochs per step
        hidden_size: Hidden layer size for the neural network
        custom_config_path: Path to custom profiling targets JSON config
        no_link: Not used with FlopCounter (kept for API compatibility)
        verbose: Verbose output
        link_only: Not supported with FlopCounter
        cache_dir: Directory for nCompass cache (default: .cache in current directory)
        size_multipliers: Comma-separated list of size multipliers (e.g., "0.5,1,2")
        scale_dims: Comma-separated list of dimension indices to scale (e.g., "0")
                   Default is "0" (batch dimension only) to avoid breaking model shapes
    
    Example usage:
        python main.py
        python main.py --steps 5
        python main.py --epochs 20
        python main.py --hidden-size 1024
        python main.py --size-multipliers "0.5,1,2"
        python main.py --scale-dims "0"  # Only scale batch dimension (default)
    """
    # Handle link-only mode (not supported with FlopCounter)
    if link_only:
        logger.error("--link-only is not supported with FlopCounter (no trace files generated)")
        return None, None
    
    # Load custom profiling targets if provided
    profiling_targets = None
    if custom_config_path:
        logger.info(f"Loading custom config from: {custom_config_path}")
        with open(custom_config_path, 'r') as f:
            profiling_targets = json.load(f)
    
    # Parse size multipliers if provided
    parsed_multipliers = None
    if size_multipliers:
        parsed_multipliers = [float(x.strip()) for x in size_multipliers.split(',')]
        logger.info(f"Using custom size multipliers: {parsed_multipliers}")
    
    # Parse scale dims if provided
    parsed_scale_dims = None
    if scale_dims:
        parsed_scale_dims = tuple(int(x.strip()) for x in scale_dims.split(','))
        logger.info(f"Scaling dimensions: {parsed_scale_dims}")
    
    # Run FLOP counting
    trace_path, result = profile(
        label=label,
        steps=steps,
        record_shapes=record_shapes,
        profile_memory=profile_memory,
        with_stack=with_stack,
        print_rows=print_rows,
        profiling_targets=profiling_targets,
        trace_dir=trace_dir,
        epochs=epochs,
        hidden_size=hidden_size,
        link_annotations=not no_link,
        verbose=verbose,
        cache_dir=cache_dir,
        size_multipliers=parsed_multipliers,
        scale_dims=parsed_scale_dims,
    )
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Parametric FLOP counting session complete!")
    if result:
        logger.info(f"Final loss: {result.get('final_loss', 'N/A')}")
        logger.info(f"Epochs: {result.get('epochs', 'N/A')}")
        logger.info(f"Batch size: {result.get('batch_size', 'N/A')}")
    logger.info(f"{'='*60}")
    
    return trace_path, result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Count FLOPs for PyTorch neural network training with parametric tensor sizing"
    )
    parser.add_argument("--label", type=str, default=None, help="Label (not used with FlopCounter)")
    parser.add_argument("--steps", type=int, default=5, help="Number of training steps per size configuration")
    parser.add_argument("--record-shapes", action="store_true", default=True, help="Not used with FlopCounter (kept for compatibility)")
    parser.add_argument("--profile-memory", action="store_true", help="Not used with FlopCounter (kept for compatibility)")
    parser.add_argument("--with-stack", action="store_true", default=True, help="Not used with FlopCounter (kept for compatibility)")
    parser.add_argument("--print-rows", type=int, default=10, help="Number of rows to print in summary (limited with FlopCounter)")
    parser.add_argument("--trace-dir", type=str, default=".traces", help="Not used with FlopCounter (kept for compatibility)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs per step")
    parser.add_argument("--hidden-size", type=int, default=512, help="Hidden layer size")
    parser.add_argument("--custom-config-path", type=str, default=None, help="Path to custom profiling config JSON")
    parser.add_argument("--no-link", action="store_true", help="Not used with FlopCounter (kept for compatibility)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--link-only", type=str, default=None, help="Not supported with FlopCounter (no trace files generated)")
    parser.add_argument("--cache-dir", type=str, default=None, help="Directory for nCompass cache (default: .cache in current directory)")
    parser.add_argument("--size-multipliers", type=str, default=None, 
                       help="Comma-separated list of size multipliers (e.g., '0.5,1,2'). Default: 0.1,0.3,0.5,0.9,1,1.1,2,3,10")
    parser.add_argument("--scale-dims", type=str, default=None,
                       help="Comma-separated list of dimension indices to scale (e.g., '0' for batch only). Default: 0 (batch dimension only to avoid breaking model shapes)")
    
    args = parser.parse_args()
    
    main(
        label=args.label,
        steps=args.steps,
        record_shapes=args.record_shapes,
        profile_memory=args.profile_memory,
        with_stack=args.with_stack,
        print_rows=args.print_rows,
        trace_dir=args.trace_dir,
        epochs=args.epochs,
        hidden_size=args.hidden_size,
        custom_config_path=args.custom_config_path,
        no_link=args.no_link,
        verbose=args.verbose,
        link_only=args.link_only,
        cache_dir=args.cache_dir,
        size_multipliers=args.size_multipliers,
        scale_dims=args.scale_dims,
    )


