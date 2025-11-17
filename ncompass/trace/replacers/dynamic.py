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
        
        **Nested Wrapper Handling:**
        When processing nested wrappers (e.g., inner wrapper at lines 12-15, outer wrapper at lines 10-50):
        1. First iteration (inner): Wraps statements → creates a `With` statement
        2. Second iteration (outer): Rebuilds metadata, which recursively processes the `With` statement
           created in step 1, finding both the `With` statement itself AND its contents.
        3. Special logic in `_find_statements_in_range` detects when all contents of a `With` statement
           are in the wrap range, and wraps the `With` statement itself (not its contents) to preserve
           the nested structure and avoid duplicate wrappers.
        
        Without this special handling, we would either:
        - Wrap both the `With` statement and its contents → duplicate wrappers
        - Wrap only the contents → breaks nesting structure
        """
        sorted_configs = sorted(wrap_configs, key=lambda x: (x['end_line'] - x['start_line'], x['start_line']))
        context_imports: List[ast.stmt] = [self._create_context_import(wc["context_class"]) for wc in wrap_configs]
        
        for wrap_config in sorted_configs:
            # Rebuild metadata each time since we modify the AST in-place
            stmt_metadata = []
            for idx, stmt in enumerate(func_node.body):
                metadata = self._build_statement_metadata([stmt], top_level_index=idx)
                stmt_metadata.extend(metadata)
            
            stmts_to_wrap, wrap_indices, common_parent = self._find_statements_in_range(
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
            
            if common_parent is not None:
                # Statements are nested inside a compound statement
                # Modify the parent's body directly (in-place modification)
                self._wrap_statements_in_parent(
                    stmt_metadata, wrap_indices, with_stmt, common_parent
                )
            else:
                # Top-level statements - modify func_node.body directly
                self._wrap_top_level_statements(
                    func_node.body, stmt_metadata, wrap_indices, with_stmt
                )
        
        # Add context imports at the beginning
        func_node.body = context_imports + func_node.body
        return func_node

    def _build_statement_metadata(self, statements: List[ast.stmt], parent: Optional[ast.stmt] = None, parent_body_index: Optional[int] = None, top_level_index: Optional[int] = None) -> List[dict]:
        """Build metadata for statements, recursively flattening compound statements to atomic statements.
        
        **Important for Nested Wrappers:**
        This function recursively processes compound statements (If, For, While, With, Try, etc.),
        which means it will also process `With` statements created by previous wrapper iterations.
        For example, if an inner wrapper created a `With` statement, this function will:
        1. Add the `With` statement itself to metadata (marked with 'is_with_statement': True)
        2. Recursively process its body, adding all statements inside it to metadata
        
        This dual representation (both the `With` statement and its contents) requires special
        handling in `_find_statements_in_range` to avoid wrapping both the `With` statement
        and its contents when processing outer wrappers.
        
        Args:
            statements: List of statements to process
            parent: The compound statement that contains these statements (if any)
            parent_body_index: Index of parent in its own parent's body (for tracking hierarchy)
            top_level_index: Index of statement in top-level function body (if parent is None)
        
        Returns:
            List of metadata dicts for statements, each containing:
            - 'stmt': The statement AST node (can be atomic or compound)
            - 'original_lineno': Starting line number
            - 'original_end_lineno': Ending line number
            - 'parent': The compound statement containing this statement (if any)
            - 'parent_body_index': Index within parent's body where this statement appears
            - 'index_in_parent': Index within parent's body (same as parent_body_index)
            - 'top_level_index': Index in top-level function body (if parent is None)
            - 'is_with_statement': True if this is a With statement (for nested wrapper handling)
            - 'is_compound_statement': True if this is a compound statement (If, For, While, With, Try, etc.)
        """
        result = []
        
        for idx, stmt in enumerate(statements):
            stmt_line = getattr(stmt, 'lineno', None)
            stmt_end_line = getattr(stmt, 'end_lineno', getattr(stmt, 'lineno', None))
            
            # Check if this is a compound statement that needs flattening
            if isinstance(stmt, (ast.For, ast.While, ast.If, ast.With, ast.Try, ast.AsyncFor, ast.AsyncWith)):
                # For compound statements, add the statement itself to metadata first
                # This allows wrappers to wrap entire compound statements (like If, With, etc.)
                is_with_statement = isinstance(stmt, ast.With)
                result.append({
                    'stmt': stmt,
                    'original_lineno': stmt_line,
                    'original_end_lineno': stmt_end_line,
                    'parent': parent,
                    'parent_body_index': parent_body_index if parent is not None else None,
                    'index_in_parent': idx if parent is not None else None,
                    'top_level_index': top_level_index if parent is None else None,
                    'is_with_statement': is_with_statement,
                    'is_compound_statement': True  # Mark as compound statement
                })
                
                # Recursively flatten the body of the compound statement
                nested_metadata = self._build_statement_metadata(
                    stmt.body, 
                    parent=stmt, 
                    parent_body_index=idx,
                    top_level_index=top_level_index if parent is None else None
                )
                result.extend(nested_metadata)
                
                # Also check for else/elif/except/finally clauses
                if isinstance(stmt, ast.If):
                    # Process orelse (else/elif clauses)
                    if stmt.orelse:
                        # Recursively process orelse statements (handles elif chains)
                        nested_metadata = self._build_statement_metadata(
                            stmt.orelse,
                            parent=stmt,
                            parent_body_index=idx,
                            top_level_index=top_level_index if parent is None else None
                        )
                        result.extend(nested_metadata)
                elif isinstance(stmt, ast.Try):
                    # Handle except and finally clauses
                    for handler in getattr(stmt, 'handlers', []):
                        nested_metadata = self._build_statement_metadata(
                            handler.body,
                            parent=stmt,
                            parent_body_index=idx,
                            top_level_index=top_level_index if parent is None else None
                        )
                        result.extend(nested_metadata)
                    if hasattr(stmt, 'orelse') and stmt.orelse:
                        nested_metadata = self._build_statement_metadata(
                            stmt.orelse,
                            parent=stmt,
                            parent_body_index=idx,
                            top_level_index=top_level_index if parent is None else None
                        )
                        result.extend(nested_metadata)
                    if hasattr(stmt, 'finalbody') and stmt.finalbody:
                        nested_metadata = self._build_statement_metadata(
                            stmt.finalbody,
                            parent=stmt,
                            parent_body_index=idx,
                            top_level_index=top_level_index if parent is None else None
                        )
                        result.extend(nested_metadata)
            else:
                # Atomic statement - add to result with parent info
                result.append({
                    'stmt': stmt,
                    'original_lineno': stmt_line,
                    'original_end_lineno': stmt_end_line,
                    'parent': parent,
                    'parent_body_index': parent_body_index if parent is not None else None,
                    'index_in_parent': idx if parent is not None else None,
                    'top_level_index': top_level_index if parent is None else None,
                    'is_with_statement': False,
                    'is_compound_statement': False
                })
        
        return result

    def _find_initial_statements_in_range(
        self, stmt_metadata: List[dict], start_line: int, end_line: int
    ) -> tuple[List[ast.stmt], List[int], List[tuple[str, Optional[ast.stmt]]]]:
        """Find all statements that fall within the specified line range.
        
        Args:
            stmt_metadata: List of statement metadata dictionaries
            start_line: Start line of the range (inclusive)
            end_line: End line of the range (inclusive)
        
        Returns:
            Tuple of (statements_to_wrap, wrap_indices, wrapped_lines_info)
            - statements_to_wrap: List of statement AST nodes in range
            - wrap_indices: List of indices in stmt_metadata corresponding to statements_to_wrap
            - wrapped_lines_info: List of (line_range_str, parent) tuples for logging
        """
        stmts_to_wrap = []
        wrap_indices = []
        wrapped_lines = []
        
        for idx, meta in enumerate(stmt_metadata):
            stmt_line = meta['original_lineno']
            stmt_end_line = meta['original_end_lineno']
            
            if stmt_line is None:
                continue
            
            # Match if statement starts within range, or starts before but ends within/after range start
            if (start_line <= stmt_line <= end_line or 
                (stmt_end_line and stmt_line < start_line and stmt_end_line >= start_line)):
                stmts_to_wrap.append(meta['stmt'])
                wrap_indices.append(idx)
                # Track line range for this statement
                line_range = f"{stmt_line}"
                if stmt_end_line and stmt_end_line != stmt_line:
                    line_range += f"-{stmt_end_line}"
                wrapped_lines.append((line_range, meta.get('parent')))
        
        return stmts_to_wrap, wrap_indices, wrapped_lines

    def _should_wrap_compound_statement_entirely(
        self, compound_stmt: ast.stmt, wrap_indices: List[int], stmt_metadata: List[dict]
    ) -> bool:
        """Check if a compound statement should be wrapped entirely.
        
        A compound statement should be wrapped entirely if ALL of its direct children
        (statements in its body) are also in the wrap range.
        
        Args:
            compound_stmt: The compound statement AST node to check
            wrap_indices: List of indices in stmt_metadata that are in the wrap range
            stmt_metadata: Full metadata list
        
        Returns:
            True if all direct children are in the wrap range, False otherwise
        """
        # Type narrowing: compound statements have a body attribute
        if not isinstance(compound_stmt, (ast.For, ast.While, ast.If, ast.With, ast.Try, ast.AsyncFor, ast.AsyncWith)):
            return False
        
        compound_body_statements = set(id(stmt) for stmt in compound_stmt.body)
        wrapped_body_statements = set()
        wrapped_body_indices = set()
        
        for other_idx in wrap_indices:
            other_meta = stmt_metadata[other_idx]
            # Skip the compound statement itself
            if (other_meta.get('is_compound_statement') and 
                other_meta['stmt'] is compound_stmt):
                continue
            # Check if this statement is a direct child
            if other_meta.get('parent') is compound_stmt:
                stmt_id = id(other_meta['stmt'])
                if stmt_id in compound_body_statements:
                    wrapped_body_statements.add(stmt_id)
                    parent_idx = other_meta.get('index_in_parent')
                    if parent_idx is not None:
                        wrapped_body_indices.add(parent_idx)
        
        # Check both by ID and by index to be thorough
        return (wrapped_body_statements == compound_body_statements and 
                wrapped_body_indices == set(range(len(compound_stmt.body))))

    def _prefer_wrapping_compound_statements(
        self, 
        wrap_indices: List[int], 
        stmt_metadata: List[dict],
        compound_statements_in_range: dict[ast.stmt, int],
        is_with_statement: bool,
        start_line: int,
        end_line: int
    ) -> tuple[List[int], List[ast.stmt]]:
        """Prefer wrapping compound statements entirely over wrapping their contents.
        
        This implements a two-pass algorithm:
        1. First pass: Identify compound statements where all children are in range
        2. Second pass: Build new list, wrapping compound statements instead of their contents
        
        Args:
            wrap_indices: Current list of indices to wrap
            stmt_metadata: Full metadata list
            compound_statements_in_range: Dict mapping compound statements to their indices
            is_with_statement: If True, only process With statements; if False, exclude With statements
            start_line: Start line of the wrap range
            end_line: End line of the wrap range
        
        Returns:
            Tuple of (new_wrap_indices, new_stmts_to_wrap)
        """
        processed_compound_stmts = set()
        statements_to_skip = set()
        
        # First pass: identify compound statements that should be wrapped entirely
        for compound_stmt, compound_idx in compound_statements_in_range.items():
            if self._should_wrap_compound_statement_entirely(compound_stmt, wrap_indices, stmt_metadata):
                processed_compound_stmts.add(compound_stmt)
                # Mark all body statements for skipping
                for other_idx in wrap_indices:
                    other_meta = stmt_metadata[other_idx]
                    if other_meta.get('parent') is compound_stmt:
                        statements_to_skip.add(other_idx)
        
        # Second pass: build new list, skipping statements inside wrapped compound statements
        new_wrap_indices = []
        new_stmts_to_wrap = []
        
        for idx in wrap_indices:
            if idx in statements_to_skip:
                continue
            
            meta = stmt_metadata[idx]
            parent = meta.get('parent')
            
            # Skip if inside a compound statement we're wrapping entirely
            if parent in processed_compound_stmts:
                continue
            
            # Add the compound statement itself if we're wrapping it entirely
            is_compound = meta.get('is_compound_statement', False)
            is_with = meta.get('is_with_statement', False)
            
            if is_compound and (is_with_statement == is_with) and meta['stmt'] in processed_compound_stmts:
                compound_idx = compound_statements_in_range[meta['stmt']]
                if compound_idx not in new_wrap_indices:
                    new_wrap_indices.append(compound_idx)
                    new_stmts_to_wrap.append(meta['stmt'])
                continue
            
            # IMPORTANT: Skip compound statements that overlap but aren't being wrapped entirely
            # BUT only if they're matched via overlap (not by their own line range)
            # If their own line range is in the wrap range, include them
            if is_compound and (is_with_statement == is_with) and meta['stmt'] not in processed_compound_stmts:
                stmt_line = meta['original_lineno']
                
                # Check if this compound statement's own line range is directly in the wrap range
                # If yes, include it (it's a direct match)
                # If no (only matched via overlap), skip it
                is_direct_match = start_line <= stmt_line <= end_line
                
                if not is_direct_match:
                    # This compound statement overlaps the range but doesn't have all children in range
                    # and its own line range is not directly in the wrap range
                    # Skip it - we'll wrap its children individually instead
                    continue
            
            # Include this statement
            new_wrap_indices.append(idx)
            new_stmts_to_wrap.append(meta['stmt'])
        
        return new_wrap_indices, new_stmts_to_wrap

    def _find_common_parent(
        self, wrap_indices: List[int], stmt_metadata: List[dict]
    ) -> Optional[ast.stmt]:
        """Find the smallest common parent (innermost compound statement) containing all matches.
        
        Args:
            wrap_indices: List of indices in stmt_metadata
            stmt_metadata: Full metadata list
        
        Returns:
            The common parent AST node, or None if statements are top-level
        """
        if not wrap_indices:
            return None
        
        # Get all unique parents
        parents = set()
        for idx in wrap_indices:
            parent = stmt_metadata[idx].get('parent')
            if parent is not None:
                parents.add(id(parent))  # Use id() to compare AST nodes
        
        # If all statements share the same parent, that's our common parent
        if len(parents) == 1:
            return stmt_metadata[wrap_indices[0]].get('parent')
        
        # If statements have different parents, they're top-level (no common parent)
        return None

    def _find_statements_in_range(
        self, stmt_metadata: List[dict], start_line: int, end_line: int
    ) -> tuple[List[ast.stmt], List[int], Optional[ast.stmt]]:
        """Find statements that fall within the specified line range.
        
        **Special Handling for Nested Wrappers:**
        When processing outer wrappers, `stmt_metadata` contains both:
        - `With` statements created by previous (inner) wrapper iterations
        - All statements inside those `With` statements (due to recursive processing)
        - Compound statements (If, For, While, Try, etc.) and their contents
        
        This function includes special logic to detect when a compound statement (including `With`)
        AND all its direct children are in the wrap range. In such cases, it prefers wrapping
        the compound statement itself rather than its contents. This:
        - Preserves the nested structure: outer_with(inner_with(...)) or outer_with(for_loop(...))
        - Prevents duplicate wrappers
        - Ensures proper nesting when processing wrappers from innermost to outermost
        
        **Example 1 (With statements):**
        - Inner wrapper (lines 12-15) creates: `With(statement_12, statement_15)`
        - Outer wrapper (lines 10-50) finds both the `With` statement and its contents
        - Special logic detects all contents are in range → wraps the `With` statement itself
        - Result: `outer_with(inner_with(...), other_statements)`
        
        **Example 2 (Other compound statements):**
        - Inner wrapper (lines 15-18) targets a For loop
        - For loop (lines 15-18) contains statements at lines 16-17
        - Special logic detects all For loop contents are in range → wraps the For loop itself
        - Result: `with_context(for_loop(...))` instead of wrapping individual statements inside
        
        Returns:
            Tuple of (statements_to_wrap, wrap_indices, common_parent)
            - statements_to_wrap: List of statement AST nodes to wrap (may include compound statements)
            - wrap_indices: List of indices in stmt_metadata corresponding to statements_to_wrap
            - common_parent: The smallest compound statement containing all matches (None if top-level)
        """
        # Step 1: Find initial statements in range
        stmts_to_wrap, wrap_indices, wrapped_lines = self._find_initial_statements_in_range(
            stmt_metadata, start_line, end_line
        )
        
        if not wrap_indices:
            return [], [], None
        
        # Step 2: Handle With statements first (they're created by our injection process)
        with_statements_in_range = {}
        for idx in wrap_indices:
            meta = stmt_metadata[idx]
            if meta.get('is_with_statement'):
                with_stmt = meta['stmt']
                with_statements_in_range[with_stmt] = idx
        
        if with_statements_in_range:
            wrap_indices, stmts_to_wrap = self._prefer_wrapping_compound_statements(
                wrap_indices, stmt_metadata, with_statements_in_range, is_with_statement=True,
                start_line=start_line, end_line=end_line
            )
        
        # Step 3: Handle other compound statements (For, If, While, Try, etc.)
        compound_statements_in_range = {}
        for idx in wrap_indices:
            meta = stmt_metadata[idx]
            if meta.get('is_compound_statement') and not meta.get('is_with_statement'):
                compound_stmt = meta['stmt']
                compound_statements_in_range[compound_stmt] = idx
        
        if compound_statements_in_range:
            wrap_indices, stmts_to_wrap = self._prefer_wrapping_compound_statements(
                wrap_indices, stmt_metadata, compound_statements_in_range, is_with_statement=False,
                start_line=start_line, end_line=end_line
            )
        
        # Step 4: Find common parent
        common_parent = self._find_common_parent(wrap_indices, stmt_metadata)
        
        # Log what was found
        if wrapped_lines:
            lines_str = ", ".join([f"line {lr}" for lr, _ in wrapped_lines])
            if common_parent:
                if isinstance(common_parent, ast.For):
                    parent_info = "for loop"
                elif isinstance(common_parent, ast.While):
                    parent_info = "while loop"
                elif isinstance(common_parent, ast.If):
                    parent_info = "if statement"
                elif isinstance(common_parent, ast.With):
                    parent_info = "with statement"
                elif isinstance(common_parent, ast.Try):
                    parent_info = "try statement"
                else:
                    parent_info = f"{type(common_parent).__name__}"
                logger.debug(
                    f"[LINE_RANGE_WRAPPING] Found {len(wrapped_lines)} statement(s) in range "
                    f"{start_line}-{end_line}: {lines_str} (nested inside {parent_info})"
                )
            else:
                logger.debug(
                    f"[LINE_RANGE_WRAPPING] Found {len(wrapped_lines)} statement(s) in range "
                    f"{start_line}-{end_line}: {lines_str} (top-level)"
                )
        
        return stmts_to_wrap, wrap_indices, common_parent

    def _wrap_statements_in_parent(
        self, stmt_metadata: List[dict], wrap_indices: List[int], 
        with_stmt: ast.With, parent: ast.stmt
    ) -> None:
        """Modify a compound statement's body to wrap matching statements.
        
        This modifies the parent's body in-place by replacing the matching statements
        with the wrapped version.
        """
        # Type narrowing: parent must be a compound statement with a body
        if not isinstance(parent, (ast.For, ast.While, ast.If, ast.With, ast.Try, ast.AsyncFor, ast.AsyncWith)):
            return
        
        # Get indices of matching statements within the parent's body
        parent_indices = []
        wrapped_line_info = []
        for idx in wrap_indices:
            meta = stmt_metadata[idx]
            if meta.get('parent') is parent:
                parent_idx = meta.get('index_in_parent')
                parent_indices.append(parent_idx)
                stmt_line = meta['original_lineno']
                stmt_end_line = meta['original_end_lineno']
                line_range = f"{stmt_line}"
                if stmt_end_line and stmt_end_line != stmt_line:
                    line_range += f"-{stmt_end_line}"
                wrapped_line_info.append((parent_idx, line_range))
        
        if not parent_indices:
            return
        
        # Sort indices to maintain order
        parent_indices.sort()
        wrapped_line_info.sort(key=lambda x: x[0])
        
        # Replace with wrapped version
        first_idx = parent_indices[0]
        last_idx = parent_indices[-1]
        
        # Log what's being wrapped
        lines_str = ", ".join([f"line {lr}" for _, lr in wrapped_line_info])
        parent_type = type(parent).__name__
        logger.debug(
            f"[LINE_RANGE_WRAPPING] Wrapping {len(wrapped_line_info)} statement(s) "
            f"({lines_str}) inside {parent_type} at body indices {first_idx}-{last_idx}"
        )
        
        parent.body = (
            parent.body[:first_idx] +
            [with_stmt] +
            parent.body[last_idx + 1:]
        )
    
    def _wrap_top_level_statements(
        self, func_body: List[ast.stmt], stmt_metadata: List[dict],
        wrap_indices: List[int], with_stmt: ast.With
    ) -> None:
        """Modify function body to wrap top-level statements.
        
        This modifies func_body in-place by replacing matching statements with wrapped version.
        """
        if not wrap_indices:
            return
        
        # Get top-level indices from metadata
        top_level_indices = []
        wrapped_line_info = []
        for idx in wrap_indices:
            meta = stmt_metadata[idx]
            top_level_idx = meta.get('top_level_index')
            if top_level_idx is not None:
                top_level_indices.append(top_level_idx)
                stmt_line = meta['original_lineno']
                stmt_end_line = meta['original_end_lineno']
                line_range = f"{stmt_line}"
                if stmt_end_line and stmt_end_line != stmt_line:
                    line_range += f"-{stmt_end_line}"
                wrapped_line_info.append((top_level_idx, line_range))
        
        if not top_level_indices:
            return
        
        # Remove duplicates and sort
        top_level_indices = sorted(set(top_level_indices))
        wrapped_line_info.sort(key=lambda x: x[0])
        
        # Replace with wrapped version
        first_idx = top_level_indices[0]
        last_idx = top_level_indices[-1]
        
        # Log what's being wrapped
        lines_str = ", ".join([f"line {lr}" for _, lr in wrapped_line_info])
        logger.debug(
            f"[LINE_RANGE_WRAPPING] Wrapping {len(wrapped_line_info)} top-level statement(s) "
            f"({lines_str}) at function body indices {first_idx}-{last_idx}"
        )
        
        func_body[:] = (
            func_body[:first_idx] +
            [with_stmt] +
            func_body[last_idx + 1:]
        )
    
    def _replace_statements_with_wrapper(
        self, stmt_metadata: List[dict], wrap_indices: List[int], with_stmt: ast.With
    ) -> List[dict]:
        """Replace wrapped statements with a single with statement in metadata.
        
        This is used for top-level statements that don't have a parent.
        """
        if not wrap_indices:
            return stmt_metadata
        
        first_idx = wrap_indices[0]
        last_idx = wrap_indices[-1]
        
        new_meta = {
            'stmt': with_stmt,
            'original_lineno': stmt_metadata[first_idx]['original_lineno'],
            'original_end_lineno': stmt_metadata[last_idx]['original_end_lineno'],
            'parent': None,
            'parent_body_index': None,
            'index_in_parent': None
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
