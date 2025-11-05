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
AST Rewrites Library - Iterative Profiling System

Main exports:
- ProfilingSession: High-level API for iterative profiling
- enable_rewrites: Enable AST rewrites with configuration
- enable_full_trace_mode: Enable minimal profiling for full trace capture
"""

from ncompass.trace.core.session import ProfilingSession
from ncompass.trace.core.rewrite import enable_rewrites, enable_full_trace_mode, disable_rewrites

__all__ = ['ProfilingSession', 'enable_rewrites', 'enable_full_trace_mode', 'disable_rewrites']

