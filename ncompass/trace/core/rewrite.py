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
from typing import Optional, Dict, Any
import importlib

from ncompass.trace.core.finder import RewritingFinder
from ncompass.trace.core.pydantic import RewriteConfig
from ncompass.trace.infra.utils import logger
from ncompass.trace.core.utils import clear_cached_modules, update_module_references
from ncompass.trace.core.pydantic import ModuleConfig


def _reimport_modules(targets: Dict[str, ModuleConfig], old_modules: Dict[str, Any]) -> None:
    """Reimport modules and update references."""
    import importlib.util
    import os
    
    # Get the RewritingFinder from sys.meta_path
    rewriting_finder = None
    for finder in sys.meta_path:
        if isinstance(finder, RewritingFinder):
            rewriting_finder = finder
            break
    
    for module_name, module_config in targets.items():
        try:
            # First try standard import (which will go through the RewritingFinder)
            importlib.import_module(module_name)
            logger.debug(f"Re-imported module with rewrites enabled: {module_name}")
        except Exception as e:
            # If standard import fails, try using the file path with the RewritingFinder
            # This handles cases where the module was imported locally and doesn't
            # have a proper package structure
            
            # First check if old module has a __file__ attribute we can use
            file_path = None
            if module_name in old_modules:
                old_mod = old_modules[module_name]
                if hasattr(old_mod, '__file__') and old_mod.__file__:
                    file_path = old_mod.__file__
                    logger.debug(f"Using file path from old module: {file_path}")
            
            # Fall back to config file path if available
            if not file_path or not os.path.exists(file_path):
                config_file_path = module_config.file_path if hasattr(module_config, 'file_path') else None
                if config_file_path and os.path.exists(config_file_path):
                    file_path = config_file_path
                    logger.debug(f"Using file path from config: {file_path}")
            
            if file_path and os.path.exists(file_path):
                try:
                    # Create the spec with rewriting if finder is available
                    if rewriting_finder:
                        # Use the RewritingFinder to create a rewriting spec
                        from ncompass.trace.replacers.utils import create_replacer_from_config
                        from ncompass.trace.core.loader import RewritingLoader
                        
                        # Get the merged config for this module
                        merged_config = rewriting_finder.merged_configs.get(module_name)
                        if merged_config:
                            replacer = create_replacer_from_config(module_name, merged_config)
                            spec = importlib.util.spec_from_loader(
                                module_name,
                                RewritingLoader(module_name, file_path, replacer),
                                origin=file_path
                            )
                        else:
                            # No config, use regular spec
                            spec = importlib.util.spec_from_file_location(module_name, file_path)
                    else:
                        # No finder, use regular spec
                        spec = importlib.util.spec_from_file_location(module_name, file_path)
                    
                    if spec and spec.loader:
                        new_module = importlib.util.module_from_spec(spec)
                        sys.modules[module_name] = new_module
                        spec.loader.exec_module(new_module)
                        logger.debug(f"Re-imported module from file path with rewrites enabled: {module_name}")
                    else:
                        logger.warning(f"Failed to create spec for module {module_name} from {file_path}")
                except Exception as e2:
                    logger.warning(f"Failed to re-import module {module_name} from file path: {e2}")
            else:
                logger.warning(f"Failed to re-import module {module_name}: {e}, and no valid file path available")
    
    # Update references in all loaded modules' namespaces
    update_module_references(old_modules)


def enable_rewrites(config: Optional[RewriteConfig] = None) -> None:
    """Enable all AST rewrites.
    Args:
        config: Optional configuration for the AST rewrites. RewriteConfig instance.
    """
    # Convert RewriteConfig to dict if needed
    config_dict = None
    old_modules = {}
    if config is not None:
        if isinstance(config, RewriteConfig):
            config_dict = config.to_dict()
            # Clear modules and get old references
            old_modules = clear_cached_modules(config.targets)
        else:
            raise TypeError(f"config must be a RewriteConfig instance, got {type(config)}")
    
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
    if config is not None and isinstance(config, RewriteConfig):
        _reimport_modules(config.targets, old_modules)
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