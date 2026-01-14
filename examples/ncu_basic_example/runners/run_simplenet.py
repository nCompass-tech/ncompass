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
    
    run_inference()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
