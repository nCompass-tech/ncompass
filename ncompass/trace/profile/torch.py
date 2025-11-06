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
Description: Torch profiler context managers for AST rewriting.
"""

from typing import Optional, Any
import torch
from ncompass.trace.profile.base import ProfileContextBase

class TorchRecordContext(ProfileContextBase):
    """Context manager for Torch profiler record_function."""
    def __init__(self, name: str) -> None:
        """Initialize Torch profiler context with a name."""
        self.name = name
        self.context = torch.profiler.record_function(self.name)

    def __enter__(self) -> Any:
        """Enter the profiler context."""
        return self.context.__enter__()
    
    def __exit__(self, exc_type: Optional[type], exc_value: Optional[Exception], traceback: Optional[Any]) -> None:
        """Exit the profiler context."""
        return self.context.__exit__(exc_type, exc_value, traceback)