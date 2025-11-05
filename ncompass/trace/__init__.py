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

