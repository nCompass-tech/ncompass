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
Description: Finders for AST rewriting.
"""

import importlib.abc
import importlib.util
import importlib.machinery
from importlib.machinery import PathFinder
import traceback
import os
from types import ModuleType
from typing import Optional, Dict, Any, Sequence
import sys
from copy import deepcopy

from ncompass.trace.replacers.utils import create_replacer_from_config
from ncompass.trace.core.loader import RewritingLoader
from ncompass.trace.infra.utils import logger
from ncompass.trace.core.utils import (
    merge_marker_configs,
    submit_queue_request,
    filepath_to_canonical_module_name,
)

class _RewritingFinderBase(importlib.abc.MetaPathFinder):
    """Base class for AST rewriting finders."""
    def __init__(self, config: Optional[dict] = None) -> None:
        self.config = config or {}
        # Target fullnames from config (manual) or will be populated by AI analysis
        self.target_fullnames = list(config.get('targets', {}).keys()) if config else []
        self.manual_configs: Dict[str, Dict[str, Any]] = config.get('targets', {}) if config else {}
        self.merged_configs: Dict[str, Any] = dict(self.manual_configs)
        self.base_url = os.getenv('BASE_URL', 'http://localhost:8000')

    def find_spec(
        self,
        fullname: str,
        path: Optional[Sequence[str]],
        target: Optional[ModuleType] = None
    ) -> Optional[importlib.machinery.ModuleSpec]:
        raise NotImplementedError

class RewritingFinder(_RewritingFinderBase):
    """Finder for AST rewriting."""
    
    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config=config)
        
        # Run AI analysis and get AI configs
        use_ai = os.getenv('USE_AI_PROFILING', 'false').lower() in ('true', '1', 'yes')
        if use_ai:
            ai_configs = self._run_ai_analysis()
        else:
            ai_configs = {}
        
        # Merge AI configs with manual configs, resolving conflicts
        if ai_configs:
            self.merged_configs = merge_marker_configs(ai_configs, self.manual_configs)
            
            # Add AI-discovered targets to target_fullnames
            for fullname in self.merged_configs.keys():
                if fullname not in self.target_fullnames:
                    self.target_fullnames.append(fullname)
            
            logger.debug(f"Final merged configs for {len(self.merged_configs)} files")
        else:
            self.merged_configs = deepcopy(self.manual_configs)
        
        # Canonicalize module names based on sys.path before building mappings
        self._canonicalize_module_names()
        
        # Build a mapping from file paths to fullnames for local import resolution
        self.filepath_to_fullname: Dict[str, str] = self._build_filepath_mapping()
    
    def _canonicalize_module_names(self) -> None:
        """Canonicalize module names in configs based on sys.path.
        
        This ensures that modules are identified by their canonical import name
        (e.g., 'vllm.v1.worker.gpu_model_runner') rather than a path-derived name
        (e.g., 'examples.vllm_example.vllm_src.vllm.v1.worker.gpu_model_runner').
        
        The canonical name is derived from the file path relative to sys.path entries.
        """
        # Build a mapping from old names to canonical names
        name_mapping: Dict[str, str] = {}
        
        for fullname, config in list(self.merged_configs.items()):
            file_path = config.get('filePath')
            if not file_path:
                continue
            
            # Compute canonical name from file path
            canonical_name = filepath_to_canonical_module_name(file_path)
            
            if canonical_name and canonical_name != fullname:
                name_mapping[fullname] = canonical_name
                logger.info(
                    f"Canonicalizing module name: {fullname} -> {canonical_name}"
                )
        
        # Apply the remapping
        for old_name, canonical_name in name_mapping.items():
            # Move the config to the canonical name
            config = self.merged_configs.pop(old_name)
            self.merged_configs[canonical_name] = config
            
            # Update target_fullnames
            if old_name in self.target_fullnames:
                self.target_fullnames.remove(old_name)
            if canonical_name not in self.target_fullnames:
                self.target_fullnames.append(canonical_name)
        
        if name_mapping:
            logger.debug(f"Canonicalized {len(name_mapping)} module names")
    
    def _build_filepath_mapping(self) -> Dict[str, str]:
        """Build a mapping from file paths to fullnames for all target modules."""
        filepath_to_fullname = {}
        for fullname in self.target_fullnames:
            # First, check if config has a file_path field
            config_file_path = self.merged_configs.get(fullname, {}).get('filePath')
            
            if config_file_path:
                # Use the file path from config if available
                normalized_path = os.path.normpath(os.path.abspath(config_file_path))
                filepath_to_fullname[normalized_path] = fullname
                logger.debug(f"Mapped file path from config {normalized_path} -> {fullname}")
            
            # Also try to find the actual spec and add that mapping
            try:
                spec = PathFinder.find_spec(fullname)
                if spec and spec.origin and spec.has_location:
                    # Normalize the path to handle different representations
                    normalized_path = os.path.normpath(os.path.abspath(spec.origin))
                    filepath_to_fullname[normalized_path] = fullname
                    logger.debug(f"Mapped file path from spec {normalized_path} -> {fullname}")
            except (ImportError, ModuleNotFoundError, ValueError, AttributeError) as e:
                logger.debug(f"Could not map file path for {fullname}: {e}")
            
        return filepath_to_fullname
    
    def _run_ai_analysis(self) -> Dict[str, Any]:
        """Run AI profiling analysis once for all target files if enabled.
        
        Returns:
            Dictionary of AI-generated configurations (empty if AI is disabled or errors occur)
        """
        
        try:
            logger.debug("[AI Profiler] Starting AI-powered profiling analysis...")
            
            # Get analysis targets from config or discover from manual configs
            analysis_targets = self.config.get('ai_analysis_targets', [])

            # Collect all target files
            file_paths = {}
            for fullname in analysis_targets:
                try:
                    spec = PathFinder.find_spec(fullname)
                    if spec and spec.origin and spec.has_location:
                        file_paths[fullname] = spec.origin
                        logger.debug(f"[AI Profiler] Found: {fullname} -> {spec.origin}")
                    else:
                        logger.debug(f"[AI Profiler] No valid spec for {fullname}")
                except (ImportError, ModuleNotFoundError, ValueError) as e:
                    logger.debug(f"[AI Profiler] Could not find spec for {fullname}: {e}")

            result = submit_queue_request(
                request={
                    'contents_by_module': file_paths
                },
                base_url=self.base_url,
                endpoint='analyze_codebase',
                await_result=True
            )
            # When await_result=True, submit_queue_request returns a dict, not a string
            assert isinstance(result, dict), "Expected dict from submit_queue_request with await_result=True"
            ai_configs = result
            logger.debug(f"[AI Profiler] Generated AI configs for {len(ai_configs)} files")
            return ai_configs
                
        except Exception as e:
            logger.debug(f"[AI Profiler] Error during AI analysis: {e}")
            traceback.print_exc()
            return {}

    def _find_spec_from_other_finders(
        self,
        fullname: str,
        path: Optional[Sequence[str]],
        target: Optional[ModuleType]
    ) -> Optional[importlib.machinery.ModuleSpec]:
        """Find spec using other finders in sys.meta_path."""
        for finder in sys.meta_path:
            if isinstance(finder, RewritingFinder):
                continue
            if hasattr(finder, "find_spec"):
                spec = finder.find_spec(fullname, path, target)
                if spec is not None:
                    return spec
        return None
    
    def _match_fullname_by_filepath(self, spec: importlib.machinery.ModuleSpec) -> Optional[str]:
        """Match a spec's file path to a target fullname."""
        if not spec or not spec.origin or not spec.has_location:
            return None
        
        normalized_origin = os.path.normpath(os.path.abspath(spec.origin))
        return self.filepath_to_fullname.get(normalized_origin)
    
    def _create_rewriting_spec(
        self,
        fullname: str,
        matched_fullname: str,
        spec: importlib.machinery.ModuleSpec
    ) -> Optional[importlib.machinery.ModuleSpec]:
        """Create a rewriting spec with the appropriate loader and replacer."""
        merged_config = self.merged_configs.get(matched_fullname)
        if not merged_config:
            return None
        
        replacer = create_replacer_from_config(matched_fullname, merged_config)
        return importlib.util.spec_from_loader(
            fullname,
            RewritingLoader(matched_fullname, spec.origin, replacer),
            origin=spec.origin
        )

    def find_spec(
        self,
        fullname: str,
        path: Optional[Sequence[str]],
        target: Optional[ModuleType] = None
    ) -> Optional[importlib.machinery.ModuleSpec]:
        # Determine the matched fullname (either direct match or by file path)
        matched_fullname = None
        spec = None
        if fullname in self.target_fullnames:
            matched_fullname = fullname
        else:
            # Try to match by file path for local imports
            spec = self._find_spec_from_other_finders(fullname, path, target)
            if spec:
                matched_fullname = self._match_fullname_by_filepath(spec)
        
        if not matched_fullname:
            return None
        
        # Get the spec if we don't have it yet
        if matched_fullname == fullname:
            spec = self._find_spec_from_other_finders(fullname, path, target)
        
        if (spec is None) or (not spec.origin) or (not spec.has_location):
            return None
        else:
            return self._create_rewriting_spec(fullname, matched_fullname, spec)