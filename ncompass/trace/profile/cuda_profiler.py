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
Description: CUDA Profiler context manager for AST rewriting.
"""

import torch
from typing import Optional, Any
from ncompass.trace.profile.base import ProfileContextBase


class CudaProfilerContext(ProfileContextBase):
    """Context manager for torch.cuda.profiler start/stop."""

    def __init__(self, name: str = "") -> None:
        """Initialize CUDA profiler context with an optional name."""
        self.name = name

    def __enter__(self) -> "CudaProfilerContext":
        """Start the CUDA profiler."""
        torch.cuda.profiler.start()
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_value: Optional[Exception],
        traceback: Optional[Any],
    ) -> None:
        """Stop the CUDA profiler."""
        torch.cuda.profiler.stop()
