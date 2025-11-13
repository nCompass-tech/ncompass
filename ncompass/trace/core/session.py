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
Description: Main ProfilingSession API for iterative profiling workflow.
"""

import os
from pathlib import Path
from typing import Optional, Callable, Dict, Any, Sequence
import glob
import gzip
import shutil
from datetime import datetime

from ncompass.trace.core.rewrite import (
    enable_rewrites, disable_rewrites
)
from ncompass.trace.core.config_manager import ConfigManager
from ncompass.trace.core.pydantic import RewriteConfig
from ncompass.trace.infra.utils import logger
from ncompass.trace.core.utils import (
    submit_queue_request, extract_source_code
)


class ProfilingSession:
    """Session for iterative profiling workflow.
    
    Workflow:
    1. run_profile() - Run with minimal markers to capture full trace
    2. get_trace_summary() - Get AI-powered trace analysis
    3. submit_feedback() - Provide user feedback for targeted profiling
    4. apply_targeted_markers() - Apply markers based on feedback
    5. run_profile() - Re-run with new markers
    6. Repeat steps 3-5 as needed
    """
    
    def __init__(
        self, 
        trace_output_dir: str,
        cache_dir: Optional[str] = None,
        session_name: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        """Initialize profiling session.
        
        Args:
            trace_output_dir: Directory where trace files are saved
            cache_dir: Optional cache directory for AI analysis
            session_name: Optional name for this session (used for trace/config naming)
        """
        self.trace_output_dir = Path(trace_output_dir)
        self.cache_dir = cache_dir or ".cache/ncompass/sessions"
        self.session_name = session_name or "profiling_session"
        self.base_url = base_url or "http://localhost:8000"

        self.config_manager = ConfigManager(cache_dir=self.cache_dir)
        
        self.latest_trace_path: Optional[str] = None
        self.latest_trace_name: Optional[str] = None
        self.latest_trace_summary: Optional[Dict[str, Any]] = None
        self.latest_feedback_context: Optional[Dict[str, Any]] = None  # Store latest feedback for summaries
        
        logger.info(f"[ProfilingSession] Session initialized: {self.session_name}")
        
    def run_profile(
        self,
        user_code: Callable,
        user_code_args: Optional[Sequence[Any]] = None,
        user_code_kwargs: Optional[Dict] = None,
        trace_name_suffix: Optional[str] = None,
        filter_trace: Optional[bool] = False,
        filter_trace_args: Optional[Dict[str, Any]] = None
    ) -> str:
        """Profile the user's code.
        Args:
            user_code: Callable that runs the code to profile
            user_code_args: Optional arguments to pass to the user code
            user_code_kwargs: Optional keyword arguments to pass to the user code
            trace_name_suffix: Optional suffix to add to trace name
            filter_trace: Optional whether to filter the trace
            filter_trace_args: Optional arguments to filter the trace
        Returns:
            Path to generated trace file
        """
        # Generate trace name with timestamp
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        if trace_name_suffix:
            self.latest_trace_name = f"{self.session_name}_{trace_name_suffix}_{timestamp}"
        else:
            self.latest_trace_name = f"{self.session_name}_iter{self.config_manager.iteration}_{timestamp}"
        
        # User is managing profiler manually (e.g., vLLM's profiler)
        logger.info("[ProfilingSession] Starting profile with external profiler (no injection)")

        # Run user code
        try:
            user_code_args = user_code_args or ()
            user_code_kwargs = user_code_kwargs or {}
            user_code(*user_code_args, **user_code_kwargs)
        except Exception as e:
            logger.error(f"[ProfilingSession] Error during profiling: {e}")
            raise
        
        # Find and rename latest trace file
        self.latest_trace_path = self._find_and_rename_latest_trace(self.latest_trace_name)
        if filter_trace:
            logger.info(f"[ProfilingSession] Filtering trace: {self.latest_trace_path}")
            result = submit_queue_request(
                request={
                    'trace_path': self.latest_trace_path,
                    # 'output_path': self.latest_trace_path,
                    'filter_args': filter_trace_args or {}
                },
                base_url=self.base_url,
                endpoint='filter',
                await_result=True
            )
            # When await_result=True, submit_queue_request returns a dict, not a string
            assert isinstance(result, dict), "Expected dict from submit_queue_request with await_result=True"
            self.latest_trace_path = result['filtered_trace_path']
        logger.info(f"[ProfilingSession] Profile complete: {self.latest_trace_path}")
        assert self.latest_trace_path is not None, "latest_trace_path should not be None"
        return self.latest_trace_path

    def get_trace_summary(
        self, 
        trace_path: Optional[str] = None,
        feedback_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Get AI-powered summary of trace.
        
        Args:
            trace_path: Optional path to trace file (uses latest if not provided)
            use_cached: If True, try to load cached summary before generating new one
            feedback_context: Optional feedback context dict with keys:
                            - feedback_text: User's question
                            - target_module: Target module
                            - start_line: Start line
                            - end_line: End line
            
        Returns:
            Dictionary with 'markdown' and 'structured' summaries
        """
        trace_path = trace_path or self.latest_trace_path
        feedback_context = feedback_context or {}
        if not trace_path:
            raise ValueError("No trace file available. Run run_profile() first.")
        
        logger.info(f"[ProfilingSession] Analyzing trace: {trace_path}")
        
        summary_request = {
            "trace_path": trace_path,
            "feedback_text": feedback_context.get('feedback_text'),
            "target_module": feedback_context.get('target_module'),
            "start_line": feedback_context.get('start_line'),
            "end_line": feedback_context.get('end_line'),
        }
        logger.info(f"[ProfilingSession] Generating summary report")
        result = submit_queue_request(
            request=summary_request,
            base_url=self.base_url,
            endpoint='summarize',
            await_result=True
        )
        # When await_result=True, submit_queue_request returns a dict, not a string
        assert isinstance(result, dict), "Expected dict from submit_queue_request with await_result=True"
        self.latest_trace_summary = result
        
        # Auto-save the summary with associated naming
        self.save_trace_summary(self.latest_trace_summary, trace_path)
        
        logger.info("[ProfilingSession] Trace summary generated and saved")
        return self.latest_trace_summary
    
    def submit_feedback(
        self,
        feedback_text: str,
        target_module: str,
        start_line: int,
        end_line: int,
        trace_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Submit user feedback to generate targeted markers.
        
        Args:
            feedback_text: User's question or feedback (e.g., "Why does flash_attention take so long?")
            target_module: Module to analyze (e.g., 'vllm.attention.backends.flash_attn')
            start_line: Start line of target region
            end_line: End line of target region
            trace_path: Optional path to trace file for context
            
        Returns:
            Dictionary with generated marker configuration
        """
        logger.info(f"[ProfilingSession] Processing feedback for {target_module}:{start_line}-{end_line}")
        
        # Store feedback context for later use in summaries
        self.latest_feedback_context = {
            'feedback_text': feedback_text,
            'target_module': target_module,
            'start_line': start_line,
            'end_line': end_line
        }
        
        # Get trace context if available
        trace_context = None
        if trace_path or self.latest_trace_summary:
            if self.latest_trace_summary:
                trace_context = self.latest_trace_summary.get('markdown', '')
            elif trace_path:
                # Load and summarize trace
                summary = self.get_trace_summary(trace_path)
                trace_context = summary.get('markdown', '')
        
        # Use AI to analyze feedback and generate markers
        source_code = extract_source_code(target_module)
        result = submit_queue_request(
            request={
                'feedback_text': feedback_text,
                'target_module': target_module,
                'start_line': start_line,
                'end_line': end_line,
                'trace_context': trace_context,
                'source_code': source_code
            },
            base_url=self.base_url,
            endpoint='analyze_with_feedback',
            await_result=True
        )
        # When await_result=True, submit_queue_request returns a dict, not a string
        assert isinstance(result, dict), "Expected dict from submit_queue_request with await_result=True"
        new_config = result
        # Store in config manager
        final_config = self.config_manager.add_config(new_config, merge=True)
        
        logger.info(f"[ProfilingSession] Feedback processed, {len(new_config.get('targets', {}))} targets added")
        
        return final_config
    
    def apply_targeted_markers(self) -> Dict[str, Any]:
        """Apply markers based on accumulated feedback.
        
        Returns:
            Current configuration
        """
        # Get feedback-generated config
        feedback_config = self.config_manager.get_current_config()
        
        # Assumes external profiler
        feedback_config_obj = RewriteConfig.from_dict(feedback_config)
        logger.info(
            f"[ProfilingSession] Applying targeted markers: {self.config_manager.iteration} iterations "
            f"(external profiler: {feedback_config_obj.full_trace_mode}) "
        )

        enable_rewrites(config=feedback_config_obj)
        return feedback_config
    
    def get_current_config(self) -> Dict[str, Any]:
        """Get current AST rewrite configuration.
        
        Returns:
            Current configuration dictionary
        """
        return self.config_manager.get_current_config()
    
    def get_config_stats(self) -> Dict[str, Any]:
        """Get statistics about current configuration.
        
        Returns:
            Dictionary with configuration statistics
        """
        return self.config_manager.get_stats()

    def get_config_file_path(self, name: Optional[str] = None) -> str:
        """Get the path to the configuration file.
        
        Args:
            name: Optional name for config file. If not provided, uses trace_name with prefix.
        
        Returns:
            Path to config file
        """
        if name:
            return f"{self.cache_dir}/{name}.json"
        elif self.latest_trace_name:
            # Use associated naming: profile_config_<trace_name>.json
            return f"{self.cache_dir}/profile_config_{self.latest_trace_name}.json"
        else:
            return f"{self.cache_dir}/profile_config.json"

    def save_config(self, name: Optional[str] = None) -> None:
        """Save current configuration to file.
        
        Uses associated naming: profile_config_<trace_name>.json if trace_name available.
        
        Args:
            name: Optional explicit name for config file (without .json extension)
        """
        filepath = self.get_config_file_path(name)
        self.config_manager.save_to_file(filepath)
        logger.info(f"[ProfilingSession] Configuration saved to {filepath}")
    
    def load_config(self, name: Optional[str] = None) -> None:
        """Load configuration from file.
        
        Args:
            name: Optional explicit name for config file (without .json extension)
        """
        filepath = self.get_config_file_path(name)
        self.config_manager.load_from_file(filepath)
        logger.info(f"[ProfilingSession] Configuration loaded from {filepath}")
    
    def reset(self) -> None:
        """Reset session to initial state."""
        self.config_manager.reset()
        disable_rewrites()
        self.latest_trace_path = None
        self.latest_trace_summary = None
        logger.info("[ProfilingSession] Session reset")
    
    def save_trace_summary(
        self, 
        summary: Optional[Dict[str, Any]] = None,
        trace_path: Optional[str] = None,
        trace_name: Optional[str] = None
    ) -> tuple[str, str]:
        """Save trace summary to disk (JSON + markdown).
        
        Similar to how profiling configs are persisted, trace summaries
        are saved in both structured (JSON) and human-readable (markdown) formats.
        Uses associated naming: summary_<trace_name>.json/.md
        
        Args:
            summary: Trace summary to save (uses latest_trace_summary if not provided)
            trace_path: Path to trace file (uses latest_trace_path if not provided)
            trace_name: Name of the trace (uses latest_trace_name if not provided)
            
        Returns:
            Tuple of (json_path, markdown_path)
        """
        summary = summary or self.latest_trace_summary
        trace_path = trace_path or self.latest_trace_path
        trace_name = trace_name or self.latest_trace_name
        
        if not summary:
            raise ValueError("No trace summary available to save")
        if not trace_path:
            raise ValueError("No trace path available")
        
        json_path, md_path = self.config_manager.save_trace_summary(
            summary=summary,
            trace_path=trace_path,
            trace_name=trace_name
        )
        
        logger.info(f"[ProfilingSession] Trace summary saved to {json_path} and {md_path}")
        return json_path, md_path
    
    def load_trace_summary(
        self, 
        filepath: Optional[str] = None,
        trace_name_filter: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Load trace summary from disk.
        
        Args:
            filepath: Optional path to specific trace summary file.
                     If not provided, loads the most recent summary.
            trace_name_filter: Optional filter for trace names (e.g., 'vllm_attention').
                              Only used when filepath is not provided.
            
        Returns:
            Trace summary dictionary or None if not found
        """
        if filepath:
            summary_data = self.config_manager.load_trace_summary(filepath)
            logger.info(f"[ProfilingSession] Loaded trace summary from {filepath}")
            return summary_data
        else:
            # Load latest, optionally filtered by trace name
            summary_data = self.config_manager.get_latest_trace_summary(
                trace_name_filter=trace_name_filter
            )
            if summary_data:
                trace_name = summary_data.get('trace_name', 'unknown')
                logger.info(f"[ProfilingSession] Loaded latest trace summary: {trace_name}")
            return summary_data
    
    def _find_latest_trace(self) -> str:
        """Find the most recently created trace file.
        
        Looks for trace files in multiple formats:
        - *.pt.trace.json* (PyTorch trace format)
        - trace.json (Chrome trace format from PyTorch profiler)
        - *.json (any JSON trace files)
        
        Returns:
            Path to latest trace file
        """
        # Try multiple patterns in order of preference
        patterns = [
            "*.pt.trace.json*",  # Preferred PyTorch trace format
            "trace.json",         # Chrome trace format from PyTorch profiler
            "*.json",            # Any JSON trace files
        ]
        
        trace_files = []
        for pattern in patterns:
            trace_pattern = str(self.trace_output_dir / pattern)
            found_files = glob.glob(trace_pattern)
            trace_files.extend(found_files)
        
        if not trace_files:
            raise FileNotFoundError(
                f"No trace files found in {self.trace_output_dir}. "
                f"Expected formats: *.pt.trace.json*, trace.json, or *.json"
            )
        
        # Sort by modification time, most recent first
        latest_trace = max(trace_files, key=os.path.getmtime)
        return latest_trace
    
    def _find_and_rename_latest_trace(self, trace_name: str) -> str:
        """Find the most recently created trace file and rename it.
        
        Args:
            trace_name: New name for the trace (without extension)
            
        Returns:
            Path to renamed trace file
        """
        # Find the latest trace file
        latest_trace = self._find_latest_trace()

        # Decompress the trace file if it is gzipped
        if latest_trace.endswith('.gz'):
            decompressed_path = latest_trace[:-3]  # Remove .gz
            with gzip.open(latest_trace, 'rb') as f_in:
                with open(decompressed_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            os.remove(latest_trace)  # Remove gzipped version
            latest_trace = decompressed_path
        
        # Generate new filename: trace_name.pt.trace.json
        trace_dir = Path(latest_trace).parent
        new_trace_path = trace_dir / f"{trace_name}.pt.trace.json"
        
        # Rename the file
        shutil.move(latest_trace, new_trace_path)
        
        logger.debug(f"[ProfilingSession] Renamed trace: {Path(latest_trace).name} -> {new_trace_path.name}")
        
        return str(new_trace_path)
    
    def filter_trace(
        self,
        trace_path: Optional[str] = None,
        output_path: Optional[str] = None,
        include_cuda_kernels: bool = True,
        include_direct_children: bool = False,
        min_duration_us: float = 0.0
    ) -> str:
        """Filter a trace to only include user annotations (trace markers).
        
        This creates a smaller, more readable trace file that only contains
        the trace markers explicitly added by the user (manual + AI-generated),
        filtering out all other Python functions and stacks.
        
        Args:
            trace_path: Path to trace file to filter (uses latest if not provided)
            output_path: Optional output path for filtered trace
            include_cuda_kernels: If True, include CUDA kernels within marker scopes
            include_direct_children: If True, include immediate child operations
            min_duration_us: Minimum duration threshold in microseconds
            
        Returns:
            Path to filtered trace file
        """
        trace_path = trace_path or self.latest_trace_path
        if not trace_path:
            raise ValueError("No trace file available. Run run_profile() first.")
        
        logger.info(f"[ProfilingSession] Filtering trace to only include user annotations")

        result = submit_queue_request(
            request={
                'trace_path': trace_path,
                'output_path': output_path,
                'filter_args': {
                    'include_cuda_kernels': include_cuda_kernels,
                    'include_direct_children': include_direct_children,
                    'min_duration_us': min_duration_us
                }
            },
            base_url=self.base_url,
            endpoint='filter',
            await_result=True
        )
        if isinstance(result, str):
            self.latest_trace_path = result
        else:
            raise ValueError(f"Expected string from submit_queue_request, got: {type(result)}")
        return self.latest_trace_path


