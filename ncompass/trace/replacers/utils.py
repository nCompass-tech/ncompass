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
from typing import List, Optional, Union, TYPE_CHECKING, cast
from ncompass.trace.infra.utils import logger
if TYPE_CHECKING:
    from ncompass.trace.replacers.dynamic import DynamicReplacer

class CallWrapperTransformer(ast.NodeTransformer):
    """Transformer to wrap specific function calls with context managers."""
    
    def __init__(self, wrap_calls: List[dict]) -> None:
        self.wrap_calls = wrap_calls
    
    def visit_Assign(self, node: ast.Assign) -> Union[ast.Assign, ast.With]:
        """Visit assignment statements and wrap matching calls."""
        if isinstance(node.value, ast.Call):
            wrapped = self._maybe_wrap_call_in_assignment(node)
            if wrapped:
                return wrapped
        result = self.generic_visit(node)
        # generic_visit returns the node (possibly transformed), which should be Assign or With
        return cast(Union[ast.Assign, ast.With], result)
    
    def _maybe_wrap_call_in_assignment(self, assign_node: ast.Assign) -> Optional[ast.With]:
        """Check if assignment call should be wrapped and return wrapped version."""
        call = assign_node.value
        # We know call is a Call from the caller's isinstance check
        assert isinstance(call, ast.Call), "Expected Call node"
        
        for wrap_config in self.wrap_calls:
            if self._matches_call_pattern(call, wrap_config['call_pattern']):
                context_args = build_context_args(wrap_config)
                return create_with_statement(context_args, [assign_node], wrap_config)
        
        return None
    
    def _matches_call_pattern(self, call: ast.Call, pattern: str) -> bool:
        """Check if a call matches the specified pattern."""
        if isinstance(call.func, ast.Name):
            return call.func.id == pattern
        elif isinstance(call.func, ast.Attribute):
            return call.func.attr == pattern
        return False
    

def make_wrapper(old_name: str, target_path: str, kind: str) -> ast.FunctionDef:
    """Create a wrapper function that delegates to a target function."""
    if kind not in {"inst", "cls", "static"}:
        raise ValueError(f"Invalid kind: {kind}")
    
    # Parse the target path
    # For "module.submodule.Class.method" -> mod="module.submodule", cls="Class", meth="method"  
    # For "module.submodule.function" -> mod="module.submodule", cls="", meth="function"
    parts = target_path.split(".")
    
    if len(parts) >= 3:
        # Check if this looks like module.Class.method or just module.function
        # We assume it's module.Class.method if there are 3+ parts
        mod = ".".join(parts[:-2])
        cls = parts[-2] 
        meth = parts[-1]
    else:
        raise ValueError(f"Ambiguous target path, unclear whether it's a class or a function: {target_path}")
    
    # Build: from <mod> import <cls> as _NC_T  OR  from <mod> import <meth> as _NC_F
    if cls:
        import_stmt = ast.ImportFrom(module=mod, names=[ast.alias(name=cls)], level=0)
        call_attr = ast.Attribute(value=ast.Name(id=cls, ctx=ast.Load()), attr=meth, ctx=ast.Load())
    else:
        import_stmt = ast.ImportFrom(module=mod, names=[ast.alias(name=meth)], level=0)
        call_attr = ast.Name(id=meth, ctx=ast.Load())

    # Signature
    call_args: List[ast.expr]
    if kind == "static":
        args = ast.arguments(
            posonlyargs=[], args=[], vararg=ast.arg("args"),
            kwonlyargs=[], kw_defaults=[], kwarg=ast.arg("kwargs"), defaults=[]
        )
        call_args = [ast.Starred(ast.Name("args", ast.Load()), ast.Load())]
    else:
        first = "cls" if kind == "cls" else "self"
        args = ast.arguments(
            posonlyargs=[], args=[ast.arg(first)], vararg=ast.arg("args"),
            kwonlyargs=[], kw_defaults=[], kwarg=ast.arg("kwargs"), defaults=[]
        )
        call_args = [ast.Name(first, ast.Load()), ast.Starred(ast.Name("args", ast.Load()), ast.Load())]
    call = ast.Call(
        func=call_attr, args=call_args,
        keywords=[ast.keyword(arg=None, value=ast.Name("kwargs", ast.Load()))]
    )

    body = [import_stmt, ast.Return(value=call)]
    fn = ast.FunctionDef(name=old_name, args=args, body=body, decorator_list=[], returns=None, type_comment=None)
    return ast.fix_missing_locations(fn)


def create_replacer_from_config(fullname: str, config: dict) -> "DynamicReplacer":
    """Create a DynamicReplacer instance from a configuration dict.
    
    Args:
        fullname: The fully qualified module name to rewrite
        config: Configuration dict with keys like:
            - class_replacements
            - class_func_replacements
            - class_func_context_wrappings
            - func_line_range_wrappings
    
    Returns:
        DynamicReplacer instance configured with the provided config
    """
    from ncompass.trace.replacers.dynamic import DynamicReplacer
    logger.debug(f"Creating DynamicReplacer for {fullname}: {config}")
    return DynamicReplacer(
        _fullname=fullname,
        _class_replacements=config.get('class_replacements', {}),
        _class_func_replacements=config.get('class_func_replacements', {}),
        _class_func_context_wrappings=config.get('class_func_context_wrappings', {}),
        _func_line_range_wrappings=config.get('func_line_range_wrappings', [])
    )

def create_with_statement(
    context_args: List[ast.expr], body: List[ast.stmt], wrap_config: dict
) -> ast.With:
    """Create a with statement wrapping the given body."""
    context_class = wrap_config['context_class'].split('.')[-1]  # Get class name only
    
    context_call = ast.Call(
        func=ast.Name(id=context_class, ctx=ast.Load()),
        args=context_args,
        keywords=[]
    )
    
    with_item = ast.withitem(
        context_expr=context_call,
        optional_vars=None
    )
    
    return ast.With(
        items=[with_item],
        body=body
    )

def build_context_args(wrap_config: dict) -> List[ast.expr]:
    """Build arguments for the context manager constructor.
    
    context_values is expected to be a list of dicts with structure:
    [
        {'name': 'arg_name', 'value': 'some_value', 'type': 'literal'},
        {'name': 'idx', 'value': 'layer_idx', 'type': 'variable'}
    ]
    
    type can be:
    - 'literal': Create ast.Constant with the value as a string
    - 'variable': Create ast.Name to reference a variable
    """
    args = []
    context_values = wrap_config.get('context_values', [])
    
    for ctx_arg in context_values:
        arg_type = ctx_arg.get('type', 'literal')
        value = ctx_arg.get('value', '')
        
        if arg_type == 'literal':
            args.append(ast.Constant(value=value))
        elif arg_type == 'variable':
            args.append(ast.Name(id=value, ctx=ast.Load()))
        else:
            raise ValueError(f"Unknown context argument type: {arg_type}")
    
    return args