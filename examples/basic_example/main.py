"""
Local profiling example for PyTorch neural network training with nCompass SDK integration.

This example demonstrates how to:
1. Integrate nCompass SDK for tracing and instrumentation
2. Train a simple neural network with instrumentation
3. Profile GPU-accelerated PyTorch code locally
4. Use custom profiling targets configuration
5. Save profiling traces locally

Based on the Modal torch_profiling example but runs entirely locally.

Usage:
    python modal_replica.py
    python modal_replica.py --label "baseline" --steps 5
    python modal_replica.py --print-rows 20
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
import torch
import torch.profiler
import json
import logging
from datetime import datetime
from uuid import uuid4

from ncompass.trace.core.rewrite import enable_rewrites
from ncompass.trace.core.pydantic import RewriteConfig
from ncompass.trace.infra.utils import logger
from ncompass.trace.converters import link_user_annotation_to_kernels
from simplenet import SimpleNet, train_simple_network

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
    **kwargs,
):
    """
    Profile the neural network training using PyTorch's built-in profiler.
    
    Args:
        label: Optional label for the profiling run
        steps: Number of profiling steps (default: 3)
        schedule: Custom profiling schedule (default: wait=1, warmup=1, active=steps-2)
        record_shapes: Record tensor shapes during profiling
        profile_memory: Profile memory usage
        with_stack: Include Python stack traces
        print_rows: Number of rows to print in summary table
        profiling_targets: Custom profiling targets config (uses PROFILING_TARGETS by default)
        trace_dir: Directory to save traces to
        cache_dir: Directory for nCompass cache (default: .cache in current directory)
        **kwargs: Arguments to pass to train_simple_network (epochs, hidden_size)
    """
    logger.info("Starting profiling session...")
    
    # Initialize nCompass SDK with profiling targets
    cache_base = cache_dir if cache_dir else f"{os.getcwd()}/.cache"
    rewrite_config = \
            Path(f"{cache_base}/ncompass/profiles/.default/Torch/current/config.json")
    if rewrite_config.exists():
        logger.info("Enabling nCompass rewrites...")
        with rewrite_config.open("r") as f:
            cfg = json.load(f)
            enable_rewrites(config=RewriteConfig.from_dict(cfg))
    
    # Create output directory for this profiling run
    function_name = "train_simple_network"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = str(uuid4())[:8]
    
    output_dir = Path(trace_dir) / (
        function_name + (f"_{label}" if label else "") + f"_{timestamp}_{run_id}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # Set up profiling schedule
    if schedule is None:
        if steps < 3:
            raise ValueError("Steps must be at least 3 when using default schedule")
        schedule = {"wait": 1, "warmup": 1, "active": steps - 2, "repeat": 0}
    schedule_obj = torch.profiler.schedule(**schedule)
    
    # Determine profiler activities based on device availability
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
        logger.info("CUDA profiling enabled")
    
    # Run profiling
    logger.info(f"Running profiling for {steps} steps...")
    with torch.profiler.profile(
        activities=activities,
        schedule=schedule_obj,
        record_shapes=record_shapes,
        profile_memory=profile_memory,
        with_stack=with_stack,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(output_dir)),
    ) as prof:
        for step_idx in range(steps):
            logger.info(f"Profiling step {step_idx + 1}/{steps}")
            result = train_simple_network(**kwargs)
            prof.step()
    
    logger.info("Profiling complete!")
    
    # Print summary table if requested
    if print_rows:
        logger.info(f"\nTop {print_rows} operations by {'CUDA' if torch.cuda.is_available() else 'CPU'} time:")
        print(
            prof.key_averages().table(
                sort_by="cuda_time_total" if torch.cuda.is_available() else "cpu_time_total",
                row_limit=print_rows
            )
        )
    
    # Find the trace file
    trace_files = list(output_dir.glob("**/*.pt.trace.json"))
    if trace_files:
        trace_path = sorted(
            trace_files,
            key=lambda pth: pth.stat().st_mtime,
            reverse=True,
        )[0]
        logger.info(f"Trace saved to: {trace_path}")
        
        # Link user_annotation events to kernels if requested
        if link_annotations:
            logger.info("Linking user_annotation events to kernels...")
            try:
                linked_events = link_user_annotation_to_kernels(str(trace_path), verbose)
                # Write the complete linked events back to the trace file
                with open(trace_path, 'w') as f:
                    json.dump(linked_events, f, indent=2)
                
                logger.info(f"Trace updated: {trace_path}")
            except Exception as e:
                logger.warning(f"Failed to link user_annotation events: {e}")
                logger.warning("Original trace file preserved")
        
        return trace_path, result
    else:
        logger.warning("No trace files found!")
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
):
    """
    Run profiling from the command line.
    
    Args:
        label: Optional label for the profiling run
        steps: Number of profiling steps
        record_shapes: Record tensor shapes during profiling
        profile_memory: Profile memory usage
        with_stack: Include Python stack traces
        print_rows: Number of rows to print in summary table
        trace_dir: Directory to save traces to
        epochs: Number of training epochs per profiling step
        hidden_size: Hidden layer size for the neural network
        custom_config_path: Path to custom profiling targets JSON config
        link_only: If provided, only link the specified trace file and exit
        cache_dir: Directory for nCompass cache (default: .cache in current directory)
    
    Example usage:
        python modal_replica.py
        python modal_replica.py --label "baseline" --steps 5
        python modal_replica.py --print-rows 20 --epochs 20
        python modal_replica.py --hidden-size 1024 --custom-config-path my_config.json
        python modal_replica.py --link-only path/to/trace.pt.trace.json
    """
    # Handle link-only mode
    if link_only:
        input_path = Path(link_only)
        if not input_path.exists():
            logger.error(f"Trace file not found: {input_path}")
            return None, None
        
        if not input_path.name.endswith('.pt.trace.json'):
            logger.error(f"Input file must end with .pt.trace.json, got: {input_path.name}")
            return None, None
        
        logger.info(f"Linking user_annotation events to kernels in: {input_path}")
        try:
            linked_events = link_user_annotation_to_kernels(str(input_path), verbose)
            
            # Generate output filename by replacing .pt.trace.json with .linked.pt.trace.json
            output_path = input_path.parent / input_path.name.replace('.pt.trace.json', '.linked.pt.trace.json')
            
            # Write the linked events to the new file
            with open(output_path, 'w') as f:
                json.dump(linked_events, f, indent=2)
            
            logger.info(f"Linked trace saved to: {output_path}")
            logger.info(f"\n{'='*60}")
            logger.info(f"Link operation complete!")
            logger.info(f"Output file: {output_path}")
            logger.info(f"{'='*60}")
            
            return str(output_path), None
        except Exception as e:
            logger.error(f"Failed to link user_annotation events: {e}")
            return None, None
    
    # Load custom profiling targets if provided
    profiling_targets = None
    if custom_config_path:
        logger.info(f"Loading custom config from: {custom_config_path}")
        with open(custom_config_path, 'r') as f:
            profiling_targets = json.load(f)
    
    # Run profiling
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
    )
    
    if trace_path:
        logger.info(f"\n{'='*60}")
        logger.info(f"Profiling session complete!")
        logger.info(f"Trace location: {trace_path}")
        logger.info(f"Final loss: {result.get('final_loss', 'N/A')}")
        logger.info(f"Epochs: {result.get('epochs', 'N/A')}")
        logger.info(f"{'='*60}")
    
    return trace_path, result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Profile PyTorch neural network training with nCompass SDK integration"
    )
    parser.add_argument("--label", type=str, default=None, help="Label for the profiling run")
    parser.add_argument("--steps", type=int, default=3, help="Number of profiling steps")
    parser.add_argument("--record-shapes", action="store_true", default=True, help="Record tensor shapes")
    parser.add_argument("--profile-memory", action="store_true", help="Profile memory usage")
    parser.add_argument("--with-stack", action="store_true", default=True, help="Include Python stack traces")
    parser.add_argument("--print-rows", type=int, default=10, help="Number of rows to print in summary")
    parser.add_argument("--trace-dir", type=str, default=".traces", help="Directory to save traces")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--hidden-size", type=int, default=512, help="Hidden layer size")
    parser.add_argument("--custom-config-path", type=str, default=None, help="Path to custom profiling config JSON")
    parser.add_argument("--no-link", action="store_true", help="Disable linking user_annotation events to kernels (linking is enabled by default)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print detailed statistics when linking annotations")
    parser.add_argument("--link-only", type=str, default=None, help="Only link the specified trace file (must end with .pt.trace.json) and output to .linked.pt.trace.json")
    parser.add_argument("--cache-dir", type=str, default=None, help="Directory for nCompass cache (default: .cache in current directory)")
    
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
    )

