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
Description: Pydantic models for AST rewrite configuration.
"""

from typing import Dict, List, Optional, Any, Literal
from pydantic import BaseModel, Field, field_validator


class ContextValue(BaseModel):
    """Context manager constructor argument specification."""
    name: str = Field(..., description="Argument name for the context manager constructor")
    value: str = Field(..., description="The value to pass to the argument")
    type: Literal['literal', 'variable'] = Field(
        ...,
        description="Type of value: 'literal' for string constants, 'variable' for variable references"
    )


class LineRangeWrapping(BaseModel):
    """Configuration for wrapping a line range with a context manager."""
    function: str = Field(..., description="Function/method name to target")
    start_line: int = Field(..., ge=1, description="Starting line number (inclusive)")
    end_line: int = Field(..., ge=1, description="Ending line number (inclusive)")
    context_class: str = Field(..., description="Full path to context manager class (e.g., 'module.ContextClass')")
    context_values: List[ContextValue] = Field(
        default_factory=list,
        description="List of arguments to pass to the context manager"
    )
    reasoning: Optional[str] = Field(None, description="Explanation for why this wrapping is needed")
    start_line_content: Optional[str] = Field(None, description="Content of the start line for validation")
    end_line_content: Optional[str] = Field(None, description="Content of the end line for validation")

    @field_validator('end_line')
    @classmethod
    def validate_line_range(cls, v: int, info) -> int:
        """Validate that end_line is greater than or equal to start_line."""
        if 'start_line' in info.data and v < info.data['start_line']:
            raise ValueError(f"end_line ({v}) must be >= start_line ({info.data['start_line']})")
        return v


class ModuleConfig(BaseModel):
    """Configuration for a single module/file."""
    class_replacements: Dict[str, str] = Field(
        default_factory=dict,
        description="Map of class names to replacement class paths"
    )
    class_func_replacements: Dict[str, str] = Field(
        default_factory=dict,
        description="Map of method names to replacement function paths"
    )
    class_func_context_wrappings: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Configuration for wrapping class methods with context managers"
    )
    func_line_range_wrappings: List[LineRangeWrapping] = Field(
        default_factory=list,
        description="List of line ranges to wrap with context managers"
    )
    filePath: str = Field(..., description="File path of the module")


class RewriteConfig(BaseModel):
    """Top-level configuration for AST rewrites."""
    targets: Dict[str, ModuleConfig] = Field(
        default_factory=dict,
        description="Map of module fullnames to their configurations"
    )
    ai_analysis_targets: List[str] = Field(
        default_factory=list,
        description="List of module fullnames to be analyzed by AI profiler"
    )
    full_trace_mode: bool = Field(
        default=False,
        description="Enable full trace capture mode for comprehensive AI analysis"
    )

    @field_validator('targets')
    @classmethod
    def validate_targets(cls, v: Dict[str, Any]) -> Dict[str, ModuleConfig]:
        """Ensure all target values are ModuleConfig instances."""
        result = {}
        for key, value in v.items():
            if isinstance(value, ModuleConfig):
                result[key] = value
            elif isinstance(value, dict):
                result[key] = ModuleConfig(**value)
            else:
                raise ValueError(f"Target '{key}' must be a ModuleConfig or dict")
        return result

    def to_dict(self) -> dict:
        """Convert to dictionary format compatible with existing code."""
        return {
            'targets': {
                fullname: config.model_dump()
                for fullname, config in self.targets.items()
            },
            'ai_analysis_targets': self.ai_analysis_targets,
            'full_trace_mode': self.full_trace_mode
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'RewriteConfig':
        """Create RewriteConfig from a dictionary."""
        return cls(**data)

