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
Description: Top level utils for AST rewriting.
"""

import sys
from typing import Optional

from ncompass.trace.core.finder import RewritingFinder
from ncompass.trace.core.pydantic import RewriteConfig, ModuleConfig
from ncompass.trace.infra.utils import logger
from ncompass.trace.core.utils import clear_cached_modules, reimport_modules

def enable_rewrites(config: Optional[RewriteConfig] = None) -> None:
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
            raise TypeError(f"config must be a RewriteConfig instance, got {type(config)}")
    
    # Create finder first - canonicalization happens in __init__
    new_finder = RewritingFinder(config=config_dict)
    
    # Use the finder's canonicalized merged_configs for clear/reimport
    old_modules = {}
    canonicalized_targets = {}
    if config is not None:
        # Convert finder's merged_configs (dict) back to Dict[str, ModuleConfig]
        canonicalized_targets = {
            name: ModuleConfig(**cfg)
            for name, cfg in new_finder.merged_configs.items()
        }
        # Clear modules and get old references using canonicalized names
        old_modules = clear_cached_modules(canonicalized_targets)
    
    # Remove existing finder if present
    for f in sys.meta_path[:]:
        if isinstance(f, RewritingFinder):
            sys.meta_path.remove(f)
            break

    # Add the new finder to sys.meta_path
    sys.meta_path.insert(0, new_finder)
    
    # Reimport with canonicalized names
    if config is not None:
        reimport_modules(canonicalized_targets, old_modules)
    logger.info(f"NC profiling enabled.")


def enable_full_trace_mode() -> None:
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


def disable_rewrites() -> None:
    """Disable AST rewrites by removing the finder from sys.meta_path."""
    for f in sys.meta_path[:]:
        if isinstance(f, RewritingFinder):
            sys.meta_path.remove(f)
            logger.info("NC profiling disabled.")
            return
    logger.debug("No active profiling to disable.")
