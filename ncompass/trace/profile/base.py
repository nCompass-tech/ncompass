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
Description: Base profiling context manager.
"""

from typing import Optional, Any
from ncompass.types.trait import Trait


class ProfileContextBase(Trait):
    """Context manager for profiling."""
    def __init__(self) -> None:
        """Initialize profiling context."""
        raise NotImplementedError("Subclasses must implement __init__")

    def __enter__(self) -> Any:
        """Enter the profiling context."""
        raise NotImplementedError("Subclasses must implement __enter__")
    
    def __exit__(self, exc_type: Optional[type], exc_value: Optional[Exception], traceback: Optional[Any]) -> None:
        """Exit the profiling context."""
        raise NotImplementedError("Subclasses must implement __exit__")
