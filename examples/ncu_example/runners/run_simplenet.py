#!/usr/bin/env python3
"""
Runner script for simplenet that loads nCompass rewrites before execution.

This script is the target for ncu profiling - it loads rewrites in-process
so they apply to the simplenet module when it's imported.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

# Add parent directory to path so we can import simplenet
script_dir = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(script_dir))

import torch

from ncompass.trace.infra.utils import logger
from ncompass.trace.core.rewrite import enable_rewrites
from ncompass.trace.core.pydantic import RewriteConfig

logger.setLevel(logging.DEBUG)


def load_ncompass_rewrites(config_path: Optional[Path] = None) -> bool:
    """
    Load nCompass rewrites from config file.
    
    Args:
        config_path: Path to nCompass rewrite config (default: config.json in script directory)
    
    Returns:
        True if rewrites were enabled, False otherwise
    """
    if config_path is None:
        config_path = script_dir / "config.json"
    
    if config_path.exists():
        logger.info(f"Loading nCompass rewrites from: {config_path}")
        try:
            with config_path.open("r") as f:
                cfg = json.load(f)
                enable_rewrites(config=RewriteConfig.from_dict(cfg))
            logger.info("nCompass rewrites enabled successfully")
            return True
        except Exception as e:
            logger.warning(f"Failed to enable rewrites: {e}")
            return False
    else:
        logger.info(f"No nCompass rewrite config found at: {config_path}")
        return False


def run_inference() -> None:
    """
    Run simplenet inference.
    """
    # Import simplenet AFTER rewrites are loaded
    import simplenet
    
    # Run the main function of simplenet
    # Note: simplenet.main() will parse sys.argv[1:]
    simplenet.main()


def main():
    parser = argparse.ArgumentParser(description="Run simplenet with nCompass rewrites")
    parser.add_argument("--config", type=str, default=None, help="Path to rewrite config")
    # Capture all other arguments to pass to simplenet
    args, unknown = parser.parse_known_args()
    
    config_path = Path(args.config) if args.config else None
    
    # Step 1: Load nCompass rewrites BEFORE importing simplenet
    load_ncompass_rewrites(config_path=config_path)
    
    # Step 2: Run inference (imports simplenet after rewrites are active)
    # The unknown args are already in sys.argv, so simplenet.main() will see them
    run_inference()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
