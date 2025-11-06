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
import traceback
import os
from types import ModuleType
from typing import Optional, Dict, Any, Sequence
import sys
from copy import deepcopy

from ncompass.trace.replacers.utils import create_replacer_from_config
from ncompass.trace.core.loader import RewritingLoader
from ncompass.trace.infra.utils import logger
from ncompass.trace.core.utils import merge_marker_configs, submit_queue_request

class _RewritingFinderBase(importlib.abc.MetaPathFinder):
    """Base class for AST rewriting finders."""
    def __init__(self, config: Optional[dict] = None) -> None:
        self.config = config or {}
        # Target fullnames from config (manual) or will be populated by AI analysis
        self.target_fullnames = list(config.get('targets', {}).keys()) if config else []
        self.manual_configs: Dict[str, Dict[str, Any]] = config.get('targets', {}) if config else {}
        self.merged_configs: Dict[str, Any] = dict(self.manual_configs)
        self.ai_analysis_done = False
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
        ai_configs = self._run_ai_analysis_if_needed()
        
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
    
    def _run_ai_analysis_if_needed(self) -> Dict[str, Any]:
        """Run AI profiling analysis once for all target files if enabled.
        
        Returns:
            Dictionary of AI-generated configurations (empty if AI is disabled or errors occur)
        """
        if self.ai_analysis_done:
            return {}
        
        self.ai_analysis_done = True
        
        # Check if AI profiling is enabled
        use_ai = os.getenv('USE_AI_PROFILING', 'false').lower() in ('true', '1', 'yes')
        if not use_ai:
            return {}
        
        try:
            logger.debug("[AI Profiler] Starting AI-powered profiling analysis...")
            
            # Get analysis targets from config or discover from manual configs
            analysis_targets = self.config.get('ai_analysis_targets', [])

            # Collect all target files
            file_paths = {}
            for fullname in analysis_targets:
                try:
                    spec = importlib.util.find_spec(fullname)
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

    def find_spec(
        self,
        fullname: str,
        path: Optional[Sequence[str]],
        target: Optional[ModuleType] = None
    ) -> Optional[importlib.machinery.ModuleSpec]:
        if fullname not in self.target_fullnames:
            return None

        for finder in sys.meta_path:
            if isinstance(finder, RewritingFinder):
                continue
            if hasattr(finder, "find_spec"):
                spec = finder.find_spec(fullname, path, target)
                if spec is not None:
                    break
        else:
            return None
        
        if (not spec) or (not spec.origin) or (not spec.has_location):
            return None
        
        # Get merged config if available
        merged_config = self.merged_configs.get(fullname)
        
        # Determine which config to use
        if not merged_config:
            return None

        # Create dynamic replacer from config
        replacer = create_replacer_from_config(fullname, merged_config)
        
        return importlib.util.spec_from_loader(
            fullname,
            RewritingLoader(fullname, spec.origin, replacer),
            origin=spec.origin
        )