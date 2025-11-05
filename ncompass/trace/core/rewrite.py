"""
Description: Top level utils for AST rewriting.
"""

import sys
from typing import Optional
from ncompass.trace.core.finder import RewritingFinder
from ncompass.trace.core.pydantic import RewriteConfig
from ncompass.trace.infra.utils import logger

def enable_rewrites(config: Optional[RewriteConfig] = None):
    """Enable all AST rewrites.
    Args:
        config: Optional configuration for the AST rewrites. RewriteConfig instance.
    """
    # Convert RewriteConfig to dict if needed
    config_dict = None
    if config is not None:
        if isinstance(config, RewriteConfig):
            config_dict = config.to_dict()
        else:
            raise TypeError(f"config must be a dict or RewriteConfig, got {type(config)}")
    
    # Check if finder already exists
    existing_finder = None
    for f in sys.meta_path:
        if isinstance(f, RewritingFinder):
            existing_finder = f
            break

    # Remove existing finder if present
    if existing_finder:
        sys.meta_path.remove(existing_finder)
    # Add new finder
    sys.meta_path.insert(0, RewritingFinder(config=config_dict))
    logger.info(f"NC profiling enabled.")


def enable_full_trace_mode():
    """Enable minimal profiling for full trace capture.
    
    This mode injects only a top-level profiler context to capture
    everything for AI analysis.
    """
    config = RewriteConfig(
        targets={},
        ai_analysis_targets=[],
        full_trace_mode=True
    )
    
    # For full trace mode, we want minimal markers
    # The AI analyzer will skip detailed analysis
    logger.info(f"NC full trace mode enabled.")
    
    enable_rewrites(config=config)


def disable_rewrites():
    """Disable AST rewrites by removing the finder from sys.meta_path."""
    for f in sys.meta_path[:]:
        if isinstance(f, RewritingFinder):
            sys.meta_path.remove(f)
            logger.info("NC profiling disabled.")
            return
    logger.debug("No active profiling to disable.")