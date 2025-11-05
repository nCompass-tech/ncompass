"""
Description: Replacer classes for AST rewriting.
"""

from dataclasses import dataclass, field
from ncompass.trace.replacers.base import _Replacer


@dataclass
class DynamicReplacer(_Replacer):
    """Dynamically created Replacer from AI-generated configs."""
    _fullname: str
    _class_replacements: dict[str, str] = field(default_factory=dict)
    _class_func_replacements: dict[str, str] = field(default_factory=dict)
    _class_func_context_wrappings: dict[str, dict[str, dict]] = field(default_factory=dict)
    _func_line_range_wrappings: list[dict] = field(default_factory=list)
    
    @property
    def fullname(self) -> str:
        return self._fullname
    
    @property
    def class_replacements(self) -> dict[str, str]:
        return self._class_replacements
    
    @property
    def class_func_replacements(self) -> dict[str, str]:
        return self._class_func_replacements
    
    @property
    def class_func_context_wrappings(self) -> dict[str, dict[str, dict]]:
        return self._class_func_context_wrappings
    
    @property
    def func_line_range_wrappings(self) -> list[dict]:
        return self._func_line_range_wrappings