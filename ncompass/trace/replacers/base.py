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

import ast


class ReplacerBase(ast.NodeTransformer):
    """Base class for AST replacers."""
    
    @property
    def is_active(self) -> bool:
        """Whether the class is active.
        If True, the class will be added to the rewrite loader.
        """
        return True

    @property
    def fullname(self) -> str:
        """Fullname of the file containing components to be replaced."""
        raise NotImplementedError

    @property
    def class_replacements(self) -> dict[str, str]:
        """Map of class: module.replacement_class_name."""
        raise NotImplementedError
    
    @property
    def class_func_replacements(self) -> dict[str, dict[str, str]]:
        """Map of class: {old method: module.replacement_class_name}."""
        raise NotImplementedError
    
    @property
    def class_func_context_wrappings(self) -> dict[str, dict[str, dict]]:
        """Map of class: {method_name: {
            'wrap_calls': [
                {
                    'context_class': 'module.ContextClass',
                    'call_pattern': 'layer',  # function/method name to wrap
                    'context_values': [
                        {'name': 'name', 'value': 'LlamaDecoderLayer.forward', 'type': 'literal'},
                        {'name': 'idx', 'value': 'idx', 'type': 'variable'}
                    ]
                }
            ]
        }}."""
        raise NotImplementedError

    @property
    def func_line_range_wrappings(self) -> list[dict]:
        """List of line ranges to wrap with context managers.
        
        [
            {
                'function': 'forward',  # function/method name to target
                'start_line': 100,      # inclusive
                'end_line': 105,        # inclusive
                'context_class': 'module.ContextClass',
                'context_values': [
                    {'name': 'name', 'value': 'some_operation', 'type': 'literal'}
                ]
            }
        ]
        
        context_values is a list of argument specifications:
        - 'name': argument name for the context manager constructor
        - 'value': the value to pass
        - 'type': either 'literal' (string constant) or 'variable' (variable reference)
        
        The line range will be validated to ensure it represents complete statements
        that can be wrapped without violating syntax rules.
        """
        raise NotImplementedError

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        """Visit and potentially modify class definitions."""
        raise NotImplementedError
    
    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        """Handle function line range wrapping for both methods and top-level functions."""
        raise NotImplementedError