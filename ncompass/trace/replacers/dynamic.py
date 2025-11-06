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
Description: Replacer classes for AST rewriting.
"""

from dataclasses import dataclass, field
from ncompass.trace.replacers.base import ReplacerBase


@dataclass
class DynamicReplacer(ReplacerBase):
    """Dynamically created Replacer from AI-generated configs."""
    _fullname: str
    _class_replacements: dict[str, str] = field(default_factory=dict)
    _class_func_replacements: dict[str, dict[str, str]] = field(default_factory=dict)
    _class_func_context_wrappings: dict[str, dict[str, dict]] = field(default_factory=dict)
    _func_line_range_wrappings: list[dict] = field(default_factory=list)
    
    @property
    def fullname(self) -> str:
        return self._fullname
    
    @property
    def class_replacements(self) -> dict[str, str]:
        return self._class_replacements
    
    @property
    def class_func_replacements(self) -> dict[str, dict[str, str]]:
        return self._class_func_replacements
    
    @property
    def class_func_context_wrappings(self) -> dict[str, dict[str, dict]]:
        return self._class_func_context_wrappings
    
    @property
    def func_line_range_wrappings(self) -> list[dict]:
        return self._func_line_range_wrappings