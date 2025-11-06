"""
Tests for ncompass.trace.replacers.utils module.
"""

import unittest
import ast
from unittest.mock import patch, MagicMock

from ncompass.trace.replacers.utils import (
    _CallWrapperTransformer,
    make_wrapper,
    create_replacer_from_config,
    create_with_statement,
    build_context_args
)


class TestCallWrapperTransformer(unittest.TestCase):
    """Test cases for _CallWrapperTransformer class."""
    
    def test_init(self):
        """Test _CallWrapperTransformer initialization."""
        wrap_calls = [{'call_pattern': 'foo'}]
        transformer = _CallWrapperTransformer(wrap_calls)
        
        self.assertEqual(transformer.wrap_calls, wrap_calls)
    
    def test_visit_assign_no_call(self):
        """Test visit_Assign with non-call assignment."""
        transformer = _CallWrapperTransformer([])
        
        # Create assignment: x = 5
        assign_node = ast.Assign(
            targets=[ast.Name(id="x", ctx=ast.Store())],
            value=ast.Constant(value=5)
        )
        
        result = transformer.visit_Assign(assign_node)
        
        # Should return the original node (via generic_visit)
        self.assertIsInstance(result, ast.Assign)
    
    def test_visit_assign_call_no_match(self):
        """Test visit_Assign with call that doesn't match pattern."""
        wrap_calls = [{'call_pattern': 'target_func'}]
        transformer = _CallWrapperTransformer(wrap_calls)
        
        # Create assignment: x = other_func()
        assign_node = ast.Assign(
            targets=[ast.Name(id="x", ctx=ast.Store())],
            value=ast.Call(
                func=ast.Name(id="other_func", ctx=ast.Load()),
                args=[],
                keywords=[]
            )
        )
        
        result = transformer.visit_Assign(assign_node)
        
        # Should return the original node
        self.assertIsInstance(result, ast.Assign)
    
    def test_visit_assign_call_matches_name(self):
        """Test visit_Assign with call matching Name pattern."""
        wrap_calls = [{
            'call_pattern': 'target_func',
            'context_class': 'test.Context',
            'context_values': []
        }]
        transformer = _CallWrapperTransformer(wrap_calls)
        
        # Create assignment: x = target_func()
        assign_node = ast.Assign(
            targets=[ast.Name(id="x", ctx=ast.Store())],
            value=ast.Call(
                func=ast.Name(id="target_func", ctx=ast.Load()),
                args=[],
                keywords=[]
            )
        )
        
        result = transformer.visit_Assign(assign_node)
        
        # Should return a With statement
        self.assertIsInstance(result, ast.With)
        # The body should contain the original assignment
        self.assertEqual(len(result.body), 1)
        self.assertIsInstance(result.body[0], ast.Assign)
    
    def test_visit_assign_call_matches_attribute(self):
        """Test visit_Assign with call matching Attribute pattern."""
        wrap_calls = [{
            'call_pattern': 'method',
            'context_class': 'test.Context',
            'context_values': []
        }]
        transformer = _CallWrapperTransformer(wrap_calls)
        
        # Create assignment: x = obj.method()
        assign_node = ast.Assign(
            targets=[ast.Name(id="x", ctx=ast.Store())],
            value=ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="obj", ctx=ast.Load()),
                    attr="method",
                    ctx=ast.Load()
                ),
                args=[],
                keywords=[]
            )
        )
        
        result = transformer.visit_Assign(assign_node)
        
        # Should return a With statement
        self.assertIsInstance(result, ast.With)
    
    def test_matches_call_pattern_name(self):
        """Test _matches_call_pattern with Name node."""
        transformer = _CallWrapperTransformer([])
        
        call = ast.Call(
            func=ast.Name(id="func_name", ctx=ast.Load()),
            args=[],
            keywords=[]
        )
        
        self.assertTrue(transformer._matches_call_pattern(call, "func_name"))
        self.assertFalse(transformer._matches_call_pattern(call, "other_name"))
    
    def test_matches_call_pattern_attribute(self):
        """Test _matches_call_pattern with Attribute node."""
        transformer = _CallWrapperTransformer([])
        
        call = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="obj", ctx=ast.Load()),
                attr="method",
                ctx=ast.Load()
            ),
            args=[],
            keywords=[]
        )
        
        self.assertTrue(transformer._matches_call_pattern(call, "method"))
        self.assertFalse(transformer._matches_call_pattern(call, "other_method"))
    
    def test_matches_call_pattern_other_type(self):
        """Test _matches_call_pattern with other node types."""
        transformer = _CallWrapperTransformer([])
        
        # Create a call with a subscript (not Name or Attribute)
        call = ast.Call(
            func=ast.Subscript(
                value=ast.Name(id="obj", ctx=ast.Load()),
                slice=ast.Constant(value=0),
                ctx=ast.Load()
            ),
            args=[],
            keywords=[]
        )
        
        self.assertFalse(transformer._matches_call_pattern(call, "anything"))


class TestMakeWrapper(unittest.TestCase):
    """Test cases for make_wrapper function."""
    
    def test_make_wrapper_inst_method(self):
        """Test make_wrapper for instance method."""
        wrapper = make_wrapper("old_method", "my.module.MyClass.new_method", "inst")
        
        self.assertIsInstance(wrapper, ast.FunctionDef)
        self.assertEqual(wrapper.name, "old_method")
        
        # Check arguments: self, *args, **kwargs
        self.assertEqual(len(wrapper.args.args), 1)
        self.assertEqual(wrapper.args.args[0].arg, "self")
        self.assertEqual(wrapper.args.vararg.arg, "args")
        self.assertEqual(wrapper.args.kwarg.arg, "kwargs")
        
        # Check body has import and return
        self.assertEqual(len(wrapper.body), 2)
        self.assertIsInstance(wrapper.body[0], ast.ImportFrom)
        self.assertIsInstance(wrapper.body[1], ast.Return)
    
    def test_make_wrapper_cls_method(self):
        """Test make_wrapper for class method."""
        wrapper = make_wrapper("cls_method", "my.module.MyClass.new_cls_method", "cls")
        
        # Check arguments: cls, *args, **kwargs
        self.assertEqual(wrapper.args.args[0].arg, "cls")
    
    def test_make_wrapper_static_method(self):
        """Test make_wrapper for static method."""
        wrapper = make_wrapper("static_method", "my.module.MyClass.new_static", "static")
        
        # Check arguments: *args, **kwargs (no self/cls)
        self.assertEqual(len(wrapper.args.args), 0)
        self.assertEqual(wrapper.args.vararg.arg, "args")
        self.assertEqual(wrapper.args.kwarg.arg, "kwargs")
    
    def test_make_wrapper_invalid_kind(self):
        """Test make_wrapper with invalid kind."""
        with self.assertRaises(ValueError) as cm:
            make_wrapper("method", "module.Class.method", "invalid")
        
        self.assertIn("Invalid kind: invalid", str(cm.exception))
    
    def test_make_wrapper_ambiguous_path(self):
        """Test make_wrapper with ambiguous target path."""
        with self.assertRaises(ValueError) as cm:
            make_wrapper("method", "module.function", "inst")
        
        self.assertIn("Ambiguous target path", str(cm.exception))
    
    def test_make_wrapper_import_statement(self):
        """Test make_wrapper creates correct import statement."""
        wrapper = make_wrapper("old_method", "my.module.MyClass.new_method", "inst")
        
        import_stmt = wrapper.body[0]
        self.assertEqual(import_stmt.module, "my.module")
        self.assertEqual(import_stmt.names[0].name, "MyClass")
    
    def test_make_wrapper_call_structure(self):
        """Test make_wrapper creates correct call structure."""
        wrapper = make_wrapper("old_method", "my.module.MyClass.new_method", "inst")
        
        return_stmt = wrapper.body[1]
        call = return_stmt.value
        
        # Check it's calling MyClass.new_method
        self.assertIsInstance(call.func, ast.Attribute)
        self.assertEqual(call.func.attr, "new_method")
        self.assertEqual(call.func.value.id, "MyClass")
    
    def test_make_wrapper_static_no_class_arg(self):
        """Test make_wrapper for static method doesn't include self/cls."""
        wrapper = make_wrapper("static_method", "my.module.MyClass.new_static", "static")
        
        return_stmt = wrapper.body[1]
        call = return_stmt.value
        
        # First arg should be *args (Starred), not a name
        self.assertEqual(len(call.args), 1)
        self.assertIsInstance(call.args[0], ast.Starred)


class TestCreateReplacerFromConfig(unittest.TestCase):
    """Test cases for create_replacer_from_config function."""
    
    def test_create_replacer_from_config_basic(self):
        """Test create_replacer_from_config with basic config."""
        config = {
            'class_replacements': {'OldClass': 'NewClass'},
            'class_func_replacements': {},
            'class_func_context_wrappings': {},
            'func_line_range_wrappings': []
        }
        
        replacer = create_replacer_from_config('test.module', config)
        
        self.assertIsNotNone(replacer)
        # Verify it's a DynamicReplacer instance
        from ncompass.trace.replacers.dynamic import DynamicReplacer
        self.assertIsInstance(replacer, DynamicReplacer)
    
    def test_create_replacer_from_config_empty(self):
        """Test create_replacer_from_config with empty config."""
        config = {}
        
        replacer = create_replacer_from_config('test.module', config)
        
        self.assertIsNotNone(replacer)
    
    def test_create_replacer_from_config_partial(self):
        """Test create_replacer_from_config with partial config."""
        config = {
            'class_replacements': {'A': 'B'}
        }
        
        replacer = create_replacer_from_config('test.module', config)
        
        self.assertIsNotNone(replacer)
    
    @patch('ncompass.trace.infra.utils.logger.debug')
    def test_create_replacer_from_config_logs(self, mock_logger):
        """Test create_replacer_from_config logs creation."""
        config = {'class_replacements': {}}
        
        create_replacer_from_config('test.module', config)
        
        mock_logger.assert_called_once()


class TestCreateWithStatement(unittest.TestCase):
    """Test cases for create_with_statement function."""
    
    def test_create_with_statement_basic(self):
        """Test create_with_statement with basic arguments."""
        context_args = [ast.Constant(value="test_name")]
        body = [ast.Pass()]
        wrap_config = {'context_class': 'profiler.NVTXContext'}
        
        with_stmt = create_with_statement(context_args, body, wrap_config)
        
        self.assertIsInstance(with_stmt, ast.With)
        self.assertEqual(len(with_stmt.items), 1)
        self.assertEqual(with_stmt.body, body)
    
    def test_create_with_statement_context_class_extraction(self):
        """Test create_with_statement extracts class name from full path."""
        context_args = []
        body = [ast.Pass()]
        wrap_config = {'context_class': 'ncompass.trace.profile.NVTXContext'}
        
        with_stmt = create_with_statement(context_args, body, wrap_config)
        
        # Should extract 'NVTXContext' from the full path
        context_call = with_stmt.items[0].context_expr
        self.assertEqual(context_call.func.id, "NVTXContext")
    
    def test_create_with_statement_multiple_args(self):
        """Test create_with_statement with multiple context args."""
        context_args = [
            ast.Constant(value="arg1"),
            ast.Constant(value="arg2"),
            ast.Name(id="var", ctx=ast.Load())
        ]
        body = [ast.Pass()]
        wrap_config = {'context_class': 'test.Context'}
        
        with_stmt = create_with_statement(context_args, body, wrap_config)
        
        context_call = with_stmt.items[0].context_expr
        self.assertEqual(len(context_call.args), 3)
    
    def test_create_with_statement_no_optional_vars(self):
        """Test create_with_statement doesn't set optional_vars."""
        context_args = []
        body = [ast.Pass()]
        wrap_config = {'context_class': 'test.Context'}
        
        with_stmt = create_with_statement(context_args, body, wrap_config)
        
        self.assertIsNone(with_stmt.items[0].optional_vars)


class TestBuildContextArgs(unittest.TestCase):
    """Test cases for build_context_args function."""
    
    def test_build_context_args_empty(self):
        """Test build_context_args with empty context_values."""
        wrap_config = {'context_values': []}
        
        args = build_context_args(wrap_config)
        
        self.assertEqual(args, [])
    
    def test_build_context_args_literal_type(self):
        """Test build_context_args with literal type."""
        wrap_config = {
            'context_values': [
                {'name': 'arg1', 'value': 'test_value', 'type': 'literal'}
            ]
        }
        
        args = build_context_args(wrap_config)
        
        self.assertEqual(len(args), 1)
        self.assertIsInstance(args[0], ast.Constant)
        self.assertEqual(args[0].value, 'test_value')
    
    def test_build_context_args_variable_type(self):
        """Test build_context_args with variable type."""
        wrap_config = {
            'context_values': [
                {'name': 'arg1', 'value': 'var_name', 'type': 'variable'}
            ]
        }
        
        args = build_context_args(wrap_config)
        
        self.assertEqual(len(args), 1)
        self.assertIsInstance(args[0], ast.Name)
        self.assertEqual(args[0].id, 'var_name')
    
    def test_build_context_args_mixed_types(self):
        """Test build_context_args with mixed types."""
        wrap_config = {
            'context_values': [
                {'name': 'literal_arg', 'value': 'string_value', 'type': 'literal'},
                {'name': 'var_arg', 'value': 'some_variable', 'type': 'variable'},
                {'name': 'another_literal', 'value': 'another_string', 'type': 'literal'}
            ]
        }
        
        args = build_context_args(wrap_config)
        
        self.assertEqual(len(args), 3)
        self.assertIsInstance(args[0], ast.Constant)
        self.assertIsInstance(args[1], ast.Name)
        self.assertIsInstance(args[2], ast.Constant)
    
    def test_build_context_args_default_type(self):
        """Test build_context_args with missing type defaults to literal."""
        wrap_config = {
            'context_values': [
                {'name': 'arg1', 'value': 'default_value'}  # No 'type' specified
            ]
        }
        
        args = build_context_args(wrap_config)
        
        self.assertEqual(len(args), 1)
        self.assertIsInstance(args[0], ast.Constant)
        self.assertEqual(args[0].value, 'default_value')
    
    def test_build_context_args_invalid_type(self):
        """Test build_context_args with invalid type raises ValueError."""
        wrap_config = {
            'context_values': [
                {'name': 'arg1', 'value': 'some_value', 'type': 'invalid_type'}
            ]
        }
        
        with self.assertRaises(ValueError) as cm:
            build_context_args(wrap_config)
        
        self.assertIn("Unknown context argument type", str(cm.exception))
    
    def test_build_context_args_empty_value(self):
        """Test build_context_args with empty value."""
        wrap_config = {
            'context_values': [
                {'name': 'arg1', 'value': '', 'type': 'literal'}
            ]
        }
        
        args = build_context_args(wrap_config)
        
        self.assertEqual(len(args), 1)
        self.assertIsInstance(args[0], ast.Constant)
        self.assertEqual(args[0].value, '')
    
    def test_build_context_args_no_context_values_key(self):
        """Test build_context_args when context_values key is missing."""
        wrap_config = {}
        
        args = build_context_args(wrap_config)
        
        self.assertEqual(args, [])


if __name__ == '__main__':
    unittest.main()

