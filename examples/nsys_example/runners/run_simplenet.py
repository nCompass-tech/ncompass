#!/usr/bin/env python3
"""
Runner script for simplenet that loads nCompass rewrites before execution.

This script is the target for nsys profiling - it loads rewrites in-process
so they apply to the simplenet module when it's imported.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Add parent directory to path so we can import simplenet
script_dir = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(script_dir))

from typing import Optional

import torch

from ncompass.trace.infra.utils import logger
from ncompass.trace.core.rewrite import enable_rewrites
from ncompass.trace.core.pydantic import RewriteConfig

logger.setLevel(logging.DEBUG)


def load_ncompass_rewrites(cache_dir: Optional[str] = None) -> bool:
    """
    Load nCompass rewrites from config file.
    
    Args:
        cache_dir: Directory for nCompass cache (default: .cache in script directory)
    
    Returns:
        True if rewrites were enabled, False otherwise
    """
    # Look for config relative to the nsys_example directory (parent of runners/)
    if cache_dir:
        cache_base = Path(cache_dir)
    else:
        cache_base = script_dir / ".cache"
    rewrite_config = cache_base / "ncompass/profiles/.default/NVTX/current/config.json"
    
    if rewrite_config.exists():
        logger.info(f"Loading nCompass rewrites from: {rewrite_config}")
        try:
            with rewrite_config.open("r") as f:
                cfg = json.load(f)
                enable_rewrites(config=RewriteConfig.from_dict(cfg))
            logger.info("nCompass rewrites enabled successfully")
            return True
        except Exception as e:
            logger.warning(f"Failed to enable rewrites: {e}")
            return False
    else:
        logger.info(f"No nCompass rewrite config found at: {rewrite_config}")
        return False


def run_training(epochs: int, hidden_size: int, with_nvtx_capture: bool = False) -> dict:
    """
    Run simplenet training with optional NVTX capture markers.
    
    Args:
        epochs: Number of training epochs
        hidden_size: Hidden layer size for the network
        with_nvtx_capture: If True, wrap training in NVTX range for capture mode
        
    Returns:
        Training result dictionary
    """
    # Import simplenet AFTER rewrites are loaded
    from simplenet import train_simple_network
    
    if with_nvtx_capture:
        torch.cuda.nvtx.range_push("nc_start_capture")
        try:
            result = train_simple_network(epochs=epochs, hidden_size=hidden_size)
        finally:
            torch.cuda.nvtx.range_pop()
    else:
        result = train_simple_network(epochs=epochs, hidden_size=hidden_size)
    
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run simplenet with nCompass rewrites")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--hidden-size", type=int, default=512, help="Hidden layer size")
    parser.add_argument("--with-nvtx-capture", action="store_true", 
                        help="Wrap training in NVTX capture range")
    parser.add_argument("--cache-dir", type=str, default=None,
                        help="Directory for nCompass cache (default: .cache in script directory)")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Step 1: Load nCompass rewrites BEFORE importing simplenet
    load_ncompass_rewrites(cache_dir=args.cache_dir)
    
    # Step 2: Run training (imports simplenet after rewrites are active)
    result = run_training(
        epochs=args.epochs,
        hidden_size=args.hidden_size,
        with_nvtx_capture=args.with_nvtx_capture
    )
    
    logger.info(f"Training result: {result}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

