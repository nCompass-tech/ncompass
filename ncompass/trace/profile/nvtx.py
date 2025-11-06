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
Description: NVTX utils for AST rewriting.
"""

from typing import Optional, Any
import torch
from ncompass.trace.infra.utils import tag
from ncompass.trace.profile.base import ProfileContextBase

class NvtxContext(ProfileContextBase):
    """Context manager for NVTX ranges."""
    def __init__(self, name: str) -> None:
        """Initialize NVTX context with a name."""
        self.name = name
        self.tag_info: list[str] = []

    def __enter__(self) -> None:
        """Push NVTX range."""
        torch.cuda.nvtx.range_push(tag(self.generate_tag_info()))

    def __exit__(self, exc_type: Optional[type], exc_value: Optional[Exception], traceback: Optional[Any]) -> None:
        """Pop NVTX range."""
        torch.cuda.nvtx.range_pop()

    def generate_tag_info(self) -> list[str]:
        """Generate tag information for the NVTX range."""
        self.tag_info.append(f"name={self.name}")
        return self.tag_info