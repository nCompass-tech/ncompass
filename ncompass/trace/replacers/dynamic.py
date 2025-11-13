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
from typing import List, Optional, cast
from dataclasses import dataclass, field

from ncompass.trace.infra.utils import logger
from ncompass.trace.replacers.base import ReplacerBase
from ncompass.trace.replacers.utils import (
    make_wrapper, CallWrapperTransformer, create_with_statement, build_context_args
)


class DynamicReplacerImpl:
    """Mixin containing implementation methods for DynamicReplacer."""
    
    def _handle_class_replacement(self, node: ast.ClassDef) -> Optional[ast.stmt]:
        """Handle class replacement by swapping the class definition with an alias.
        
        Returns the replacement statement if a replacement is found, None otherwise.
        """
        repl = self.class_replacements.get(node.name)  # type: ignore[attr-defined]
        if not repl:
            return None
        
        mod, _, name = repl.rpartition(".")
        if mod:
            new_stmt = ast.ImportFrom(
                module=mod,
                names=[ast.alias(name=name, asname=node.name)],
                level=0,
            )
        else:
            # replacement is a bare name in scope (e.g. Foo rather than myproj.mymod.Foo)
            new_stmt = ast.Assign(
                targets=[ast.Name(id=node.name, ctx=ast.Store())],
                value=ast.Name(id=repl, ctx=ast.Load()),
            )
        return ast.copy_location(new_stmt, node)
    
    def _handle_method_transplants(self, node: ast.ClassDef) -> None:
        """Handle method transplants by replacing methods with wrappers.
        
        Modifies node.body in place.
        """
        repl_map = self.class_func_replacements.get(node.name, {})  # type: ignore[attr-defined]
        if not repl_map:
            return
        
        new_body: List[ast.stmt] = []
        for stmt in node.body:
            if isinstance(stmt, ast.FunctionDef) and stmt.name in repl_map:
                decorators = {d.id for d in stmt.decorator_list if isinstance(d, ast.Name)}
                if "staticmethod" in decorators:
                    kind = "static"
                elif "classmethod" in decorators:
                    kind = "cls"
                else:
                    kind = "inst"
                method_name: str = stmt.name
                new_body.append(make_wrapper(method_name, repl_map[method_name], kind))
            else:
                new_body.append(stmt)
        node.body = new_body
    
    def _handle_function_context_wrapping(self, node: ast.ClassDef) -> None:
        """Handle function body context wrapping.
        
        Modifies node.body in place by wrapping specified function calls with contexts.
        """
        context_wrappings = self.class_func_context_wrappings.get(node.name, {})  # type: ignore[attr-defined]
        if not context_wrappings:
            return
        
        new_body = []
        for stmt in node.body:
            if isinstance(stmt, ast.FunctionDef) and stmt.name in context_wrappings:
                # Transform the function body to wrap specified calls with contexts
                wrapper_config = context_wrappings[stmt.name]
                stmt = self._wrap_function_calls_with_context(stmt, wrapper_config)
                logger.debug(f"Wrapped function: {stmt.name}")
            new_body.append(stmt)
        node.body = new_body

    def _wrap_function_calls_with_context(self, func_node: ast.FunctionDef, config: dict) -> ast.FunctionDef:
        """Transform function body to wrap specified calls with context managers."""
        wrap_calls = config['wrap_calls']
        
        # Add imports for the context classes at the beginning of the function
        context_imports: List[ast.stmt] = [self._create_context_import(wc["context_class"]) for wc in wrap_calls]
        
        # Transform the function body
        transformer = CallWrapperTransformer(wrap_calls)
        new_body: List[ast.stmt] = context_imports.copy()
        for stmt in func_node.body:
            transformed_stmt = transformer.visit(stmt)
            new_body.append(cast(ast.stmt, transformed_stmt))
        
        func_node.body = new_body
        return func_node

    def _create_context_import(self, context_class: str) -> ast.ImportFrom:
        """Create import statement for context class."""
        module_path, _, class_name = context_class.rpartition('.')
        return ast.ImportFrom(
            module=module_path,
            names=[ast.alias(name=class_name, asname=None)],
            level=0
        )

    def _wrap_function_line_ranges_with_context(self, func_node: ast.FunctionDef, wrap_configs: List[dict]) -> ast.FunctionDef:
        """Transform function body to wrap specified line ranges with context managers.
        
        Processes from innermost to outermost range for proper nesting support.
        """
        sorted_configs = sorted(wrap_configs, key=lambda x: (x['end_line'] - x['start_line'], x['start_line']))
        context_imports: List[ast.stmt] = [self._create_context_import(wc["context_class"]) for wc in wrap_configs]
        
        stmt_metadata = self._build_statement_metadata(func_node.body)
        
        for wrap_config in sorted_configs:
            stmts_to_wrap, wrap_indices = self._find_statements_in_range(
                stmt_metadata, wrap_config['start_line'], wrap_config['end_line']
            )
            
            if not stmts_to_wrap:
                logger.warning(f"No statements found in line range {wrap_config['start_line']}-{wrap_config['end_line']}")
                continue
            
            with_stmt = create_with_statement(
                build_context_args(wrap_config),
                stmts_to_wrap,
                wrap_config
            )
            
            if hasattr(stmts_to_wrap[0], 'lineno'):
                with_stmt.lineno = stmts_to_wrap[0].lineno
                with_stmt.col_offset = getattr(stmts_to_wrap[0], 'col_offset', 0)
            
            stmt_metadata = self._replace_statements_with_wrapper(
                stmt_metadata, wrap_indices, with_stmt
            )
        
        func_node.body = context_imports + [meta['stmt'] for meta in stmt_metadata]
        return func_node

    def _build_statement_metadata(self, statements: List[ast.stmt]) -> List[dict]:
        """Build metadata for statements tracking original line numbers."""
        return [
            {
                'stmt': stmt,
                'original_lineno': getattr(stmt, 'lineno', None),
                'original_end_lineno': getattr(stmt, 'end_lineno', getattr(stmt, 'lineno', None))
            }
            for stmt in statements
        ]

    def _find_statements_in_range(
        self, stmt_metadata: List[dict], start_line: int, end_line: int
    ) -> tuple[List[ast.stmt], List[int]]:
        """Find statements that fall within the specified line range."""
        stmts_to_wrap = []
        wrap_indices = []
        
        for idx, meta in enumerate(stmt_metadata):
            stmt_line = meta['original_lineno']
            stmt_end_line = meta['original_end_lineno']
            
            if stmt_line is None:
                continue
            
            if (start_line <= stmt_line <= end_line or 
                (stmt_end_line and stmt_line < start_line and stmt_end_line >= start_line)):
                stmts_to_wrap.append(meta['stmt'])
                wrap_indices.append(idx)
        
        return stmts_to_wrap, wrap_indices

    def _replace_statements_with_wrapper(
        self, stmt_metadata: List[dict], wrap_indices: List[int], with_stmt: ast.With
    ) -> List[dict]:
        """Replace wrapped statements with a single with statement in metadata."""
        if not wrap_indices:
            return stmt_metadata
        
        first_idx = wrap_indices[0]
        last_idx = wrap_indices[-1]
        
        new_meta = {
            'stmt': with_stmt,
            'original_lineno': stmt_metadata[first_idx]['original_lineno'],
            'original_end_lineno': stmt_metadata[last_idx]['original_end_lineno']
        }
        
        return (
            stmt_metadata[:first_idx] +
            [new_meta] +
            stmt_metadata[last_idx + 1:]
        )


@dataclass(slots=True, frozen=False, init=False)  # type: ignore[call-overload]
class DynamicReplacer(ReplacerBase, DynamicReplacerImpl):
    """Dynamically created Replacer from AI-generated configs."""
    _fullname: str
    _class_replacements: dict[str, str] = field(default_factory=dict)
    _class_func_replacements: dict[str, dict[str, str]] = field(default_factory=dict)
    _class_func_context_wrappings: dict[str, dict[str, dict]] = field(default_factory=dict)
    _func_line_range_wrappings: list[dict] = field(default_factory=list)
    

    def __init__(
        self,
        _fullname: str,
        _class_replacements: Optional[dict[str, str]] = None,
        _class_func_replacements: Optional[dict[str, dict[str, str]]] = None,
        _class_func_context_wrappings: Optional[dict[str, dict[str, dict]]] = None,
        _func_line_range_wrappings: Optional[list[dict]] = None,
    ) -> None:
        super(ReplacerBase, self).__init__()  # NodeTransformer init
        self._fullname = _fullname
        self._class_replacements = _class_replacements or {}
        self._class_func_replacements = _class_func_replacements or {}
        self._class_func_context_wrappings = _class_func_context_wrappings or {}
        self._func_line_range_wrappings = _func_line_range_wrappings or []

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
    
    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        """Visit and potentially modify class definitions."""
        logger.debug(f"[VISIT_CLASSDEF] Scanning {node.name}")
        replacement_stmt = self._handle_class_replacement(node)
        if replacement_stmt:
            return replacement_stmt
        
        # *) Method transplants
        self._handle_method_transplants(node)
        
        # *) Function body context wrapping
        self._handle_function_context_wrapping(node)
        
        return self.generic_visit(node)
    
    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        """Handle function line range wrapping for both methods and top-level functions."""
        # Find all line range configs that target this function
        matching_configs = [
            config for config in self.func_line_range_wrappings
            if config.get('function') == node.name
        ]
        
        if matching_configs:
            logger.debug(f"[LINE_RANGE_WRAPPING] Processing function: {node.name} with {len(matching_configs)} configs")
            node = self._wrap_function_line_ranges_with_context(node, matching_configs)
        
        return self.generic_visit(node)
