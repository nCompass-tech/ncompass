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
Description: Configuration management for incremental profiling.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple
from copy import deepcopy
from ncompass.types.trait import Trait
from ncompass.types.immutable import mutate
from enum import Enum
from ncompass.trace.infra.utils import logger, deep_merge


class ListSetMode(Enum):
    REPLACE = "replace"
    APPEND = "append"
    PREPEND = "prepend"


class DictSetMode(Enum):
    REPLACE = "replace"
    SET = "set"
    DELETE = "delete"

class ConfigManager(Trait):
    """Manage incremental configuration for AST rewrites."""
    
    def __init__(self, cache_dir: str):
        """Initialize configuration manager.
        
        Args:
            cache_dir: Directory to save summaries
        """
        self.cache_dir = cache_dir
        
        self.configs: List[Dict[str, Any]] = []  # History of configs
        self.current_config: Dict[str, Any] = {}
        self.iteration = 0
    

    @mutate
    def _mutate_configs(
        self,
        configs: List[Dict[str, Any]],
        mode: ListSetMode
    ) -> None:
        """Set the configurations.
        
        Args:
            configs: The configurations to set
            mode: The mode to set the configurations
        """
        if mode == ListSetMode.REPLACE:
            self.configs = configs
        elif mode == ListSetMode.APPEND:
            self.configs.extend(configs)
        elif mode == ListSetMode.PREPEND:
            self.configs = configs + self.configs
        else:
            raise ValueError(f"Invalid mode: {mode}")


    @mutate
    def _mutate_current_config(
        self,
        value: Union[Dict[str, Any], Tuple[Any, Any], Any],
        mode: DictSetMode
    ) -> None:
        """Set the current configuration.
        
        Args:
            value: The value to set(dict if replace, tuple of (key, value) if set or delete)
            mode: The mode to set the value
        """
        if mode == DictSetMode.REPLACE:
            assert isinstance(value, dict), "Value must be a dict for REPLACE mode"
            self.current_config = value
        elif mode == DictSetMode.SET:
            assert isinstance(value, tuple) and len(value) == 2, "Value must be a tuple of (key, value) for SET mode"
            k, v = value
            self.current_config[k] = v
        elif mode == DictSetMode.DELETE:
            assert isinstance(value, str), "Value must be a string key for DELETE mode"
            self.current_config.pop(value, None)
        else:
            raise ValueError(f"Invalid mode: {mode}")
    
    @mutate
    def _mutate_iteration(self, value: int) -> None:
        """Set the iteration counter.
        
        Args:
            value: The iteration value to set
        """
        self.iteration = value
    
    def add_config(self, new_config: Dict[str, Any], merge: bool = True) -> Dict[str, Any]:
        """Add a new configuration.
        
        Args:
            new_config: New configuration to add
            merge: If True, merge with existing config; if False, replace
            
        Returns:
            The resulting merged configuration
        """
        self._mutate_iteration(self.iteration + 1)
        
        if merge and self.current_config:
            # Merge new config with current
            merged = self._merge_configs(self.current_config, new_config)
        else:
            # Replace current config
            merged = deepcopy(new_config)
        
        # Store in history
        self._mutate_configs(
            configs=[{
                'iteration': self.iteration,
                'config': deepcopy(merged),
                'added': deepcopy(new_config)
            }],
            mode=ListSetMode.APPEND
        )
        
        self._mutate_current_config(merged, DictSetMode.REPLACE)
        
        logger.debug(f"[ConfigManager] Iteration {self.iteration}: Added config with {len(new_config.get('targets', {}))} targets")
        
        return deepcopy(self.current_config)

    def _merge_configs(self, base: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
        """Merge two configurations intelligently.
        
        Args:
            base: Base configuration
            new: New configuration to merge in
            
        Returns:
            Merged configuration
        """
        merged = deepcopy(base)
        
        # Merge top-level fields
        for key in ['ai_analysis_targets', 'ai_use_discovery']:
            if key in new:
                # For lists, extend; for others, replace
                if isinstance(new[key], list):
                    existing = merged.get(key, [])
                    merged[key] = existing + [t for t in new[key] if t not in existing]
                else:
                    merged[key] = new[key]
        
        # Merge targets
        if 'targets' in new:
            if 'targets' not in merged:
                merged['targets'] = {}
            
            for module_path, module_config in new['targets'].items():
                if module_path in merged['targets']:
                    # Merge module-level configs
                    merged['targets'][module_path] = deep_merge(
                        merged['targets'][module_path],
                        module_config
                    )
                else:
                    merged['targets'][module_path] = deepcopy(module_config)
        
        return merged
    
    def get_current_config(self) -> Dict[str, Any]:
        """Get the current merged configuration.
        
        Returns:
            Current configuration dictionary
        """
        return deepcopy(self.current_config)
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get the full configuration history.
        
        Returns:
            List of historical configurations
        """
        return deepcopy(self.configs)
    
    def reset(self) -> None:
        """Reset to empty configuration."""
        self._mutate_configs(
            configs=[],
            mode=ListSetMode.REPLACE
        )
        self._mutate_current_config({}, DictSetMode.REPLACE)
        self._mutate_iteration(0)
        logger.debug("[ConfigManager] Reset to empty configuration")
    
    def save_to_file(self, filepath: str) -> None:
        """Save current configuration to a JSON file.
        
        Args:
            filepath: Path to save configuration
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump({
                'iteration': self.iteration,
                'current_config': self.current_config,
                'history': self.configs
            }, f, indent=2)
        
        logger.debug(f"[ConfigManager] Saved configuration to {filepath}")
    
    def load_from_file(self, filepath: str) -> None:
        """Load configuration from a JSON file.
        
        Args:
            filepath: Path to load configuration from
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self._mutate_iteration(data.get('iteration', 0))
        self._mutate_current_config(data.get('current_config', {}), DictSetMode.REPLACE)
        self._mutate_configs(configs=data.get('history', []), mode=ListSetMode.REPLACE)
        
        logger.debug(f"[ConfigManager] Loaded configuration from {filepath} (iteration {self.iteration})")
    
    def validate_config(self, config: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """Validate a configuration.
        
        Args:
            config: Configuration to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check structure
        if not isinstance(config, dict):
            return False, "Configuration must be a dictionary"
        
        # Check targets structure if present
        if 'targets' in config:
            if not isinstance(config['targets'], dict):
                return False, "'targets' must be a dictionary"
            
            for module_path, module_config in config['targets'].items():
                if not isinstance(module_config, dict):
                    return False, f"Config for '{module_path}' must be a dictionary"
                
                # Check for valid config keys
                valid_keys = {
                    'class_replacements',
                    'class_func_replacements',
                    'class_func_context_wrappings',
                    'func_line_range_wrappings'
                }
                
                for key in module_config.keys():
                    if key not in valid_keys:
                        logger.warning(f"Unknown config key '{key}' in module '{module_path}'")
        
        return True, None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about current configuration.
        
        Returns:
            Dictionary with configuration statistics
        """
        stats = {
            'iteration': self.iteration,
            'total_configs': len(self.configs),
            'total_targets': len(self.current_config.get('targets', {})),
            'targets': []
        }
        
        for module_path, module_config in self.current_config.get('targets', {}).items():
            target_stats = {
                'module': module_path,
                'wrappers': 0
            }
            
            # Count line range wrappings
            if 'func_line_range_wrappings' in module_config:
                target_stats['wrappers'] += len(module_config['func_line_range_wrappings'])
            
            stats['targets'].append(target_stats)
        
        return stats
    
    def save_trace_summary(
        self, 
        summary: Dict[str, Any], 
        trace_path: str,
        trace_name: Optional[str] = None,
        output_dir: str = ".cache/ncompass/sessions",
    ) -> tuple[str, str]:
        """Save trace summary to both JSON and markdown files.
        
        Uses associated naming: summary_<trace_name>.json/.md
        
        Args:
            summary: Trace summary dictionary (from trace-handler-service/summarize)
            trace_path: Path to the trace file this summary is for
            trace_name: Name of the trace (extracted from trace_path if not provided)
            output_dir: Directory to save summaries
            
        Returns:
            Tuple of (json_path, markdown_path)
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Extract trace name from path if not provided
        if not trace_name:
            # Remove .pt.trace.json extension
            trace_name = Path(trace_path).name.replace('.pt.trace.json', '')
        
        # Generate summary filename: summary_<trace_name>
        base_name = f"summary_{trace_name}"
        
        json_path = output_path / f"{base_name}.json"
        md_path = output_path / f"{base_name}.md"
        
        # Prepare JSON data with metadata
        json_data = {
            'trace_name': trace_name,
            'trace_path': trace_path,
            'summary': summary
        }
        
        # Save JSON
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        # Save markdown (extract from summary)
        markdown_content = summary.get('markdown', '')
        with open(md_path, 'w') as f:
            f.write(f"# Trace Summary: {trace_name}\n\n")
            f.write(f"**Trace File:** `{trace_path}`\n\n")
            f.write("---\n\n")
            f.write(markdown_content)
        
        logger.debug(f"[ConfigManager] Saved trace summary to {json_path} and {md_path}")
        
        return str(json_path), str(md_path)
    
    def load_trace_summary(self, filepath: str) -> Dict[str, Any]:
        """Load trace summary from a JSON file.
        
        Args:
            filepath: Path to trace summary JSON file
            
        Returns:
            Trace summary dictionary
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        logger.debug(f"[ConfigManager] Loaded trace summary from {filepath}")
        
        return data
    
    def get_latest_trace_summary(
        self, 
        trace_name_filter: Optional[str] = None,
        output_dir: str = "./.profiling_session"
    ) -> Optional[Dict[str, Any]]:
        """Get the most recent trace summary, optionally filtered by trace name.
        
        Args:
            trace_name_filter: Optional filter to match trace names (e.g., 'vllm_attention')
                              Only summaries for traces containing this string will be considered
            output_dir: Directory where summaries are saved
            
        Returns:
            Latest trace summary dictionary or None if not found
        """
        output_path = Path(output_dir)
        if not output_path.exists():
            return None
        
        # Find all trace summary JSON files
        summary_files = list(output_path.glob("summary_*.json"))
        
        if not summary_files:
            return None
        
        # Apply filter if provided
        if trace_name_filter:
            filtered_files = []
            for summary_file in summary_files:
                # Load to check trace_name
                try:
                    with open(summary_file, 'r') as f:
                        data = json.load(f)
                        trace_name = data.get('trace_name', '')
                        if trace_name_filter in trace_name:
                            filtered_files.append(summary_file)
                except Exception as e:
                    logger.warning(f"[ConfigManager] Could not load {summary_file}: {e}")
                    continue
            
            summary_files = filtered_files
        
        if not summary_files:
            return None
        
        # Sort by modification time, most recent first
        latest_file = max(summary_files, key=lambda p: p.stat().st_mtime)
        
        return self.load_trace_summary(str(latest_file))

