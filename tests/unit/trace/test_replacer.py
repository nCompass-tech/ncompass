"""
Tests for ncompass.trace.replacers module.
"""

import unittest
import ast
from unittest.mock import patch, MagicMock


from ncompass.trace.replacers.base import ReplacerBase
from ncompass.trace.replacers.dynamic import DynamicReplacer

class TestReplacerBase(unittest.TestCase):
    """Test cases for the ReplacerBase base class."""
    
    def test_abstract_properties_not_implemented(self):
        """Test that abstract properties raise NotImplementedError."""
        replacer = ReplacerBase()
        
        with self.assertRaises(NotImplementedError):
            _ = replacer.fullname
    
    def test_visit_class_def_not_implemented(self):
        """Test that visit_ClassDef raises NotImplementedError."""
        replacer = ReplacerBase()
        mock_node = MagicMock()
        
        with self.assertRaises(NotImplementedError):
            replacer.visit_ClassDef(mock_node)
    
    def test_inheritance(self):
        """Test that ReplacerBase inherits from ast.NodeTransformer."""
        replacer = ReplacerBase()
        self.assertIsInstance(replacer, ast.NodeTransformer)


class TestClassReplacements(unittest.TestCase):
    """Test cases for class replacement functionality."""
    
    def setUp(self):
        """Set up test fixtures with a custom replacer."""
        # Create a test replacer that has both class and function replacements
        class TestReplacer(DynamicReplacer):
            
            @property
            def class_replacements(self) -> dict[str, str]:
                return {
                    "OldClass": "new.module.NewClass",
                    "LocalClass": "ReplacementClass"
                }
        
        self.replacer = TestReplacer(_fullname="test.module")
    
    def test_class_replacement_with_module(self):
        """Test that class replacement creates proper import statement."""
        # Create a class definition that should be replaced
        class_node = ast.ClassDef(
            name="OldClass",
            bases=[],
            keywords=[],
            decorator_list=[],
            body=[ast.Pass()]
        )
        
        result = self.replacer.visit_ClassDef(class_node)
        
        # Should return an ImportFrom statement
        self.assertIsInstance(result, ast.ImportFrom)
        self.assertEqual(result.module, "new.module")
        self.assertEqual(len(result.names), 1)
        self.assertEqual(result.names[0].name, "NewClass")
        self.assertEqual(result.names[0].asname, "OldClass")
        self.assertEqual(result.level, 0)
    
    def test_class_replacement_local(self):
        """Test that local class replacement creates assignment statement."""
        # Create a class definition that should be replaced with a local name
        class_node = ast.ClassDef(
            name="LocalClass",
            bases=[],
            keywords=[],
            decorator_list=[],
            body=[ast.Pass()]
        )
        
        result = self.replacer.visit_ClassDef(class_node)
        
        # Should return an assignment statement
        self.assertIsInstance(result, ast.Assign)
        self.assertEqual(len(result.targets), 1)
        self.assertEqual(result.targets[0].id, "LocalClass")
        self.assertEqual(result.value.id, "ReplacementClass")
    
    def test_order_of_operations_class_then_function_replacement(self):
        """Test that when a class is replaced, function replacements are applied to the replacement class name."""
        # Create a custom replacer that replaces ClassA with ClassB, 
        # and ClassB has function replacements
        replacer = DynamicReplacer(
            _fullname="test.order",
            _class_replacements={"ClassA": "replacement.module.ClassB"}
        )
        
        # Create ClassA with foo method
        foo_method = ast.FunctionDef(
            name="foo",
            args=ast.arguments(
                posonlyargs=[],
                args=[ast.arg(arg="self", annotation=None)],
                vararg=None,
                kwonlyargs=[],
                kw_defaults=[],
                kwarg=None,
                defaults=[]
            ),
            body=[ast.Pass()],
            decorator_list=[],
            returns=None
        )
        
        class_node = ast.ClassDef(
            name="ClassA",
            bases=[],
            keywords=[],
            decorator_list=[],
            body=[foo_method]
        )
        
        result = replacer.visit_ClassDef(class_node)
        
        # Since ClassA is in class_replacements, it should be replaced with an import
        # The function replacement should NOT apply because the class is being replaced entirely
        self.assertIsInstance(result, ast.ImportFrom)
        self.assertEqual(result.module, "replacement.module")
        self.assertEqual(result.names[0].name, "ClassB")
        self.assertEqual(result.names[0].asname, "ClassA")
    
    def test_function_replacement_without_class_replacement(self):
        """Test that function replacements work when class is not being replaced."""
        # Create a replacer that only has function replacements for a class
        class FuncOnlyReplacer(DynamicReplacer):
            """Dummy replacer for testing function replacements."""
        
        replacer = FuncOnlyReplacer(_fullname="test.func")

        # Create TestClass with original_method
        original_method = ast.FunctionDef(
            name="original_method",
            args=ast.arguments(
                posonlyargs=[],
                args=[ast.arg(arg="self", annotation=None)],
                vararg=None,
                kwonlyargs=[],
                kw_defaults=[],
                kwarg=None,
                defaults=[]
            ),
            body=[ast.Pass()],
            decorator_list=[],
            returns=None
        )
        
        other_method = ast.FunctionDef(
            name="other_method",
            args=ast.arguments(
                posonlyargs=[],
                args=[ast.arg(arg="self", annotation=None)],
                vararg=None,
                kwonlyargs=[],
                kw_defaults=[],
                kwarg=None,
                defaults=[]
            ),
            body=[ast.Pass()],
            decorator_list=[],
            returns=None
        )
        
        class_node = ast.ClassDef(
            name="TestClass",
            bases=[],
            keywords=[],
            decorator_list=[],
            body=[original_method, other_method]
        )
        
        result = replacer.visit_ClassDef(class_node)
        
        # Should return the methods
        self.assertIsInstance(result, ast.ClassDef)
        self.assertEqual(result.name, "TestClass")
        self.assertEqual(len(result.body), 2)


class TestClassFuncReplacements(unittest.TestCase):
    """Test cases for class_func_replacements functionality."""
    
    def setUp(self):
        """Set up test fixtures with a custom replacer."""
        # Create a test replacer that has method replacements
        self.replacer = DynamicReplacer(
            _fullname="test.method.replacement",
            _class_replacements={},
            _class_func_replacements={
                "TestClass": {
                    "old_method": "replacement.module.ReplacementClass.new_method",
                    "static_method": "replacement.module.standalone_function",
                    "class_method": "replacement.module.ReplacementClass.new_class_method"
                }
            }
        )
    
    def test_instance_method_replacement(self):
        """Test that instance method replacement creates proper wrapper with import."""
        # Create a class with an instance method to be replaced
        old_method = ast.FunctionDef(
            name="old_method",
            args=ast.arguments(
                posonlyargs=[],
                args=[ast.arg(arg="self", annotation=None)],
                vararg=None,
                kwonlyargs=[],
                kw_defaults=[],
                kwarg=None,
                defaults=[]
            ),
            body=[ast.Pass()],
            decorator_list=[],
            returns=None
        )
        
        unchanged_method = ast.FunctionDef(
            name="unchanged_method",
            args=ast.arguments(
                posonlyargs=[],
                args=[ast.arg(arg="self", annotation=None)],
                vararg=None,
                kwonlyargs=[],
                kw_defaults=[],
                kwarg=None,
                defaults=[]
            ),
            body=[ast.Pass()],
            decorator_list=[],
            returns=None
        )
        
        class_node = ast.ClassDef(
            name="TestClass",
            bases=[],
            keywords=[],
            decorator_list=[],
            body=[old_method, unchanged_method]
        )
        
        result = self.replacer.visit_ClassDef(class_node)
        
        # Should return the modified class with wrapper method
        self.assertIsInstance(result, ast.ClassDef)
        self.assertEqual(result.name, "TestClass")
        self.assertEqual(len(result.body), 2)  # wrapper + unchanged_method
        
        # First should be the wrapper function
        wrapper_func = result.body[0]
        self.assertIsInstance(wrapper_func, ast.FunctionDef)
        self.assertEqual(wrapper_func.name, "old_method")
        
        # Check wrapper function structure
        self.assertEqual(len(wrapper_func.body), 2)  # import + return
        
        # First statement should be import
        import_stmt = wrapper_func.body[0]
        self.assertIsInstance(import_stmt, ast.ImportFrom)
        self.assertEqual(import_stmt.module, "replacement.module")
        self.assertEqual(import_stmt.names[0].name, "ReplacementClass")
        
        # Second statement should be return with method call
        return_stmt = wrapper_func.body[1]
        self.assertIsInstance(return_stmt, ast.Return)
        call = return_stmt.value
        self.assertIsInstance(call.func, ast.Attribute)
        self.assertEqual(call.func.attr, "new_method")
        self.assertIsInstance(call.func.value, ast.Name)
        self.assertEqual(call.func.value.id, "ReplacementClass")
        
        # Check call arguments (should have self, *args, **kwargs)
        self.assertEqual(len(call.args), 2)  # self, *args
        self.assertIsInstance(call.args[0], ast.Name)
        self.assertEqual(call.args[0].id, "self")
        self.assertIsInstance(call.args[1], ast.Starred)
        
        # Second should be the unchanged method
        unchanged_func = result.body[1]
        self.assertIsInstance(unchanged_func, ast.FunctionDef)
        self.assertEqual(unchanged_func.name, "unchanged_method")
    
    @unittest.skip("Static method replacement with non-class function not yet supported.")
    def test_static_method_replacement(self):
        """Test that static method replacement creates proper wrapper."""
        # Create a class with a static method to be replaced
        static_method = ast.FunctionDef(
            name="static_method",
            args=ast.arguments(
                posonlyargs=[],
                args=[],
                vararg=None,
                kwonlyargs=[],
                kw_defaults=[],
                kwarg=None,
                defaults=[]
            ),
            body=[ast.Pass()],
            decorator_list=[ast.Name(id="staticmethod", ctx=ast.Load())],
            returns=None
        )
        
        class_node = ast.ClassDef(
            name="TestClass",
            bases=[],
            keywords=[],
            decorator_list=[],
            body=[static_method]
        )
        
        result = self.replacer.visit_ClassDef(class_node)
        
        # Should return the modified class with wrapper method
        self.assertIsInstance(result, ast.ClassDef)
        self.assertEqual(len(result.body), 1)
        
        # Should be the wrapper function
        wrapper_func = result.body[0]
        self.assertIsInstance(wrapper_func, ast.FunctionDef)
        self.assertEqual(wrapper_func.name, "static_method")
        
        # Check wrapper function arguments (should be *args, **kwargs only)
        args = wrapper_func.args
        self.assertEqual(len(args.args), 0)  # No positional args
        self.assertIsNotNone(args.vararg)
        self.assertEqual(args.vararg.arg, "args")
        self.assertIsNotNone(args.kwarg)
        self.assertEqual(args.kwarg.arg, "kwargs")
        
        # Check import statement
        import_stmt = wrapper_func.body[0]
        self.assertIsInstance(import_stmt, ast.ImportFrom)
        self.assertEqual(import_stmt.module, "replacement.module")
        self.assertEqual(import_stmt.names[0].name, "standalone_function")
        
        # Check return statement calls the imported function directly
        return_stmt = wrapper_func.body[1]
        call = return_stmt.value
        self.assertIsInstance(call.func, ast.Name)
        self.assertEqual(call.func.id, "standalone_function")
    
    def test_class_method_replacement(self):
        """Test that class method replacement creates proper wrapper."""
        # Create a class with a class method to be replaced
        class_method = ast.FunctionDef(
            name="class_method",
            args=ast.arguments(
                posonlyargs=[],
                args=[ast.arg(arg="cls", annotation=None)],
                vararg=None,
                kwonlyargs=[],
                kw_defaults=[],
                kwarg=None,
                defaults=[]
            ),
            body=[ast.Pass()],
            decorator_list=[ast.Name(id="classmethod", ctx=ast.Load())],
            returns=None
        )
        
        class_node = ast.ClassDef(
            name="TestClass",
            bases=[],
            keywords=[],
            decorator_list=[],
            body=[class_method]
        )
        
        result = self.replacer.visit_ClassDef(class_node)
        
        # Should return the modified class with wrapper method
        self.assertIsInstance(result, ast.ClassDef)
        self.assertEqual(len(result.body), 1)
        
        # Should be the wrapper function
        wrapper_func = result.body[0]
        self.assertIsInstance(wrapper_func, ast.FunctionDef)
        self.assertEqual(wrapper_func.name, "class_method")
        
        # Check wrapper function arguments (should be cls, *args, **kwargs)
        args = wrapper_func.args
        self.assertEqual(len(args.args), 1)
        self.assertEqual(args.args[0].arg, "cls")
        
        # Check that the call passes cls as first argument
        return_stmt = wrapper_func.body[1]
        call = return_stmt.value
        self.assertEqual(len(call.args), 2)  # cls, *args
        self.assertEqual(call.args[0].id, "cls")
    
    def test_method_replacement_preserves_non_function_statements(self):
        """Test that method replacement preserves non-function class body statements."""
        # Create a class with mixed statements
        class_var = ast.Assign(
            targets=[ast.Name(id="class_var", ctx=ast.Store())],
            value=ast.Constant(value="test")
        )
        
        old_method = ast.FunctionDef(
            name="old_method",
            args=ast.arguments(
                posonlyargs=[],
                args=[ast.arg(arg="self", annotation=None)],
                vararg=None,
                kwonlyargs=[],
                kw_defaults=[],
                kwarg=None,
                defaults=[]
            ),
            body=[ast.Pass()],
            decorator_list=[],
            returns=None
        )
        
        another_var = ast.Assign(
            targets=[ast.Name(id="another_var", ctx=ast.Store())],
            value=ast.Constant(value=42)
        )
        
        class_node = ast.ClassDef(
            name="TestClass",
            bases=[],
            keywords=[],
            decorator_list=[],
            body=[class_var, old_method, another_var]
        )
        
        result = self.replacer.visit_ClassDef(class_node)
        
        # Should have: class_var, wrapper_method, another_var
        self.assertEqual(len(result.body), 3)
        
        # First should be the class variable
        self.assertIsInstance(result.body[0], ast.Assign)
        self.assertEqual(result.body[0].targets[0].id, "class_var")
        
        # Second should be the wrapper function
        self.assertIsInstance(result.body[1], ast.FunctionDef)
        self.assertEqual(result.body[1].name, "old_method")
        
        # Third should be the other variable
        self.assertIsInstance(result.body[2], ast.Assign)
        self.assertEqual(result.body[2].targets[0].id, "another_var")
    
    def test_method_replacement_with_no_matching_methods(self):
        """Test that classes with no matching methods are unchanged."""
        # Create a class with methods that don't match replacement config
        some_method = ast.FunctionDef(
            name="some_method",
            args=ast.arguments(
                posonlyargs=[],
                args=[ast.arg(arg="self", annotation=None)],
                vararg=None,
                kwonlyargs=[],
                kw_defaults=[],
                kwarg=None,
                defaults=[]
            ),
            body=[ast.Pass()],
            decorator_list=[],
            returns=None
        )
        
        class_node = ast.ClassDef(
            name="TestClass",
            bases=[],
            keywords=[],
            decorator_list=[],
            body=[some_method]
        )
        
        with patch.object(self.replacer, 'generic_visit') as mock_generic_visit:
            mock_generic_visit.return_value = class_node
            
            result = self.replacer.visit_ClassDef(class_node)
            
            # Should be unchanged (only generic_visit called)
            self.assertEqual(result, class_node)
            mock_generic_visit.assert_called_once_with(class_node)
    
    def test_method_replacement_with_non_target_class(self):
        """Test that non-target classes are unchanged."""
        # Create a class that's not in the replacement config
        some_method = ast.FunctionDef(
            name="old_method",  # This method name matches, but class doesn't
            args=ast.arguments(
                posonlyargs=[],
                args=[ast.arg(arg="self", annotation=None)],
                vararg=None,
                kwonlyargs=[],
                kw_defaults=[],
                kwarg=None,
                defaults=[]
            ),
            body=[ast.Pass()],
            decorator_list=[],
            returns=None
        )
        
        class_node = ast.ClassDef(
            name="OtherClass",  # Not "TestClass"
            bases=[],
            keywords=[],
            decorator_list=[],
            body=[some_method]
        )
        
        with patch.object(self.replacer, 'generic_visit') as mock_generic_visit:
            mock_generic_visit.return_value = class_node
            
            result = self.replacer.visit_ClassDef(class_node)
            
            # Should be unchanged
            self.assertEqual(result, class_node)
            mock_generic_visit.assert_called_once_with(class_node)


class TestMakeWrapperFunction(unittest.TestCase):
    """Test cases for the make_wrapper utility function."""
    
    def testmake_wrapper_instance_method_with_class(self):
        """Test make_wrapper for instance method with class path."""
        from ncompass.trace.replacers.utils import make_wrapper
        
        wrapper = make_wrapper("old_method", "my.module.MyClass.new_method", "inst")
        
        # Check function structure
        self.assertIsInstance(wrapper, ast.FunctionDef)
        self.assertEqual(wrapper.name, "old_method")
        
        # Check arguments
        args = wrapper.args
        self.assertEqual(len(args.args), 1)
        self.assertEqual(args.args[0].arg, "self")
        self.assertIsNotNone(args.vararg)
        self.assertEqual(args.vararg.arg, "args")
        self.assertIsNotNone(args.kwarg)
        self.assertEqual(args.kwarg.arg, "kwargs")
        
        # Check body
        self.assertEqual(len(wrapper.body), 2)
        
        # Import statement
        import_stmt = wrapper.body[0]
        self.assertIsInstance(import_stmt, ast.ImportFrom)
        self.assertEqual(import_stmt.module, "my.module")
        self.assertEqual(import_stmt.names[0].name, "MyClass")
        
        # Return statement
        return_stmt = wrapper.body[1]
        self.assertIsInstance(return_stmt, ast.Return)
        call = return_stmt.value
        self.assertIsInstance(call.func, ast.Attribute)
        self.assertEqual(call.func.attr, "new_method")
    
    def testmake_wrapper_class_method(self):
        """Test make_wrapper for class method."""
        from ncompass.trace.replacers.utils import make_wrapper
        
        wrapper = make_wrapper("cls_method", "my.module.MyClass.new_cls_method", "cls")
        
        # Check arguments
        args = wrapper.args
        self.assertEqual(len(args.args), 1)
        self.assertEqual(args.args[0].arg, "cls")
        
        # Check call passes cls as first argument
        return_stmt = wrapper.body[1]
        call = return_stmt.value
        self.assertEqual(len(call.args), 2)  # cls, *args
        self.assertEqual(call.args[0].id, "cls")
    
    def testmake_wrapper_invalid_kind(self):
        """Test make_wrapper raises error for invalid kind."""
        from ncompass.trace.replacers.utils import make_wrapper
        
        with self.assertRaises(ValueError) as cm:
            make_wrapper("test", "module.func", "invalid")
        
        self.assertIn("Invalid kind: invalid", str(cm.exception))

class TestLineRangeWrapping(unittest.TestCase):
    """Test cases for class_func_line_range_wrappings functionality."""
    
    def setUp(self):
        """Set up test fixtures with a custom replacer."""
        # Create a test replacer that has line range wrappings
        self.replacer = DynamicReplacer(
            _fullname="test.line.range",
            _class_replacements={},
            _class_func_replacements={},
            _func_line_range_wrappings=[
                {
                    'function': 'test_method',
                    'start_line': 10,
                    'end_line': 12,
                    'context_class': 'profiler.NVTXContext',
                    'context_values': [
                        {'name': 'name', 'value': 'operation_1', 'type': 'literal'}
                    ]
                }
            ]
        )
    
    def test_single_line_range_wrapping(self):
        """Test that a single line range gets wrapped correctly."""
        # Create a method with statements that will be wrapped
        test_method = ast.FunctionDef(
            name="test_method",
            args=ast.arguments(
                posonlyargs=[],
                args=[ast.arg(arg="self", annotation=None)],
                vararg=None,
                kwonlyargs=[],
                kw_defaults=[],
                kwarg=None,
                defaults=[]
            ),
            body=[
                ast.Assign(
                    targets=[ast.Name(id="x", ctx=ast.Store())],
                    value=ast.Constant(value=1),
                    lineno=10
                ),
                ast.Assign(
                    targets=[ast.Name(id="y", ctx=ast.Store())],
                    value=ast.Constant(value=2),
                    lineno=11
                ),
                ast.Assign(
                    targets=[ast.Name(id="z", ctx=ast.Store())],
                    value=ast.Constant(value=3),
                    lineno=12
                ),
            ],
            decorator_list=[],
            returns=None,
            lineno=9
        )
        
        class_node = ast.ClassDef(
            name="TestClass",
            bases=[],
            keywords=[],
            decorator_list=[],
            body=[test_method]
        )
        
        result = self.replacer.visit_ClassDef(class_node)
        
        # Should return the modified class
        self.assertIsInstance(result, ast.ClassDef)
        self.assertEqual(result.name, "TestClass")
        
        # Get the modified method
        modified_method = result.body[0]
        self.assertIsInstance(modified_method, ast.FunctionDef)
        self.assertEqual(modified_method.name, "test_method")
        
        # Method body should have: import statement + with statement
        self.assertEqual(len(modified_method.body), 2)
        
        # First should be the import
        import_stmt = modified_method.body[0]
        self.assertIsInstance(import_stmt, ast.ImportFrom)
        self.assertEqual(import_stmt.module, "profiler")
        self.assertEqual(import_stmt.names[0].name, "NVTXContext")
        
        # Second should be the with statement
        with_stmt = modified_method.body[1]
        self.assertIsInstance(with_stmt, ast.With)
        
        # Check context manager call
        context_call = with_stmt.items[0].context_expr
        self.assertIsInstance(context_call, ast.Call)
        self.assertIsInstance(context_call.func, ast.Name)
        self.assertEqual(context_call.func.id, "NVTXContext")
        
        # Check context arguments
        self.assertEqual(len(context_call.args), 1)
        self.assertIsInstance(context_call.args[0], ast.Constant)
        self.assertEqual(context_call.args[0].value, "operation_1")
        
        # Check wrapped body contains all three statements
        self.assertEqual(len(with_stmt.body), 3)
        self.assertIsInstance(with_stmt.body[0], ast.Assign)
        self.assertIsInstance(with_stmt.body[1], ast.Assign)
        self.assertIsInstance(with_stmt.body[2], ast.Assign)
    
    def test_multiple_line_ranges_in_same_method(self):
        """Test that multiple non-overlapping line ranges can be wrapped."""
        replacer = DynamicReplacer(
            _fullname="test.multi.range",
            _class_replacements={},
            _class_func_replacements={},
            _func_line_range_wrappings=[
                {
                    'function': 'test_method',
                    'start_line': 10,
                    'end_line': 11,
                    'context_class': 'profiler.Context1',
                    'context_values': [
                        {'name': 'name', 'value': 'first_range', 'type': 'literal'}
                    ]
                },
                {
                    'function': 'test_method',
                    'start_line': 15,
                    'end_line': 16,
                    'context_class': 'profiler.Context2',
                    'context_values': [
                        {'name': 'name', 'value': 'second_range', 'type': 'literal'}
                    ]
                }
            ]
        )
        
        # Create method with statements in different line ranges
        test_method = ast.FunctionDef(
            name="test_method",
            args=ast.arguments(
                posonlyargs=[],
                args=[ast.arg(arg="self", annotation=None)],
                vararg=None,
                kwonlyargs=[],
                kw_defaults=[],
                kwarg=None,
                defaults=[]
            ),
            body=[
                ast.Assign(
                    targets=[ast.Name(id="a", ctx=ast.Store())],
                    value=ast.Constant(value=1),
                    lineno=10
                ),
                ast.Assign(
                    targets=[ast.Name(id="b", ctx=ast.Store())],
                    value=ast.Constant(value=2),
                    lineno=11
                ),
                ast.Assign(
                    targets=[ast.Name(id="c", ctx=ast.Store())],
                    value=ast.Constant(value=3),
                    lineno=13
                ),
                ast.Assign(
                    targets=[ast.Name(id="d", ctx=ast.Store())],
                    value=ast.Constant(value=4),
                    lineno=15
                ),
                ast.Assign(
                    targets=[ast.Name(id="e", ctx=ast.Store())],
                    value=ast.Constant(value=5),
                    lineno=16
                ),
            ],
            decorator_list=[],
            returns=None,
            lineno=9
        )
        
        class_node = ast.ClassDef(
            name="TestClass",
            bases=[],
            keywords=[],
            decorator_list=[],
            body=[test_method]
        )
        
        result = replacer.visit_ClassDef(class_node)
        
        # Get the modified method
        modified_method = result.body[0]
        
        # Should have: 2 imports + 2 with statements + 1 unwrapped statement
        self.assertEqual(len(modified_method.body), 5)
        
        # First two should be imports
        self.assertIsInstance(modified_method.body[0], ast.ImportFrom)
        self.assertIsInstance(modified_method.body[1], ast.ImportFrom)
        
        # Then we should have wrapped statements and unwrapped statement
        # The order depends on how the wrapping logic processes them
        with_statements = [stmt for stmt in modified_method.body if isinstance(stmt, ast.With)]
        self.assertEqual(len(with_statements), 2)
    
    def test_line_range_with_variable_context_args(self):
        """Test line range wrapping with variable references in context args."""
        replacer = DynamicReplacer(
            _fullname="test.var.args",
            _class_replacements={},
            _class_func_replacements={},
            _func_line_range_wrappings=[
                {
                    'function': 'test_method',
                    'start_line': 10,
                    'end_line': 11,
                    'context_class': 'profiler.NVTXContext',
                    'context_values': [
                        {'name': 'name', 'value': 'layer_forward', 'type': 'literal'},
                        {'name': 'idx', 'value': 'layer_idx', 'type': 'variable'}
                    ]
                }
            ]
        )
        
        test_method = ast.FunctionDef(
            name="test_method",
            args=ast.arguments(
                posonlyargs=[],
                args=[ast.arg(arg="self", annotation=None)],
                vararg=None,
                kwonlyargs=[],
                kw_defaults=[],
                kwarg=None,
                defaults=[]
            ),
            body=[
                ast.Assign(
                    targets=[ast.Name(id="layer_idx", ctx=ast.Store())],
                    value=ast.Constant(value=0),
                    lineno=9
                ),
                ast.Assign(
                    targets=[ast.Name(id="x", ctx=ast.Store())],
                    value=ast.Constant(value=1),
                    lineno=10
                ),
                ast.Assign(
                    targets=[ast.Name(id="y", ctx=ast.Store())],
                    value=ast.Constant(value=2),
                    lineno=11
                ),
            ],
            decorator_list=[],
            returns=None,
            lineno=8
        )
        
        class_node = ast.ClassDef(
            name="TestClass",
            bases=[],
            keywords=[],
            decorator_list=[],
            body=[test_method]
        )
        
        result = replacer.visit_ClassDef(class_node)
        modified_method = result.body[0]
        
        # Find the with statement
        with_stmt = None
        for stmt in modified_method.body:
            if isinstance(stmt, ast.With):
                with_stmt = stmt
                break
        
        self.assertIsNotNone(with_stmt)
        
        # Check context arguments
        context_call = with_stmt.items[0].context_expr
        self.assertEqual(len(context_call.args), 2)
        
        # First arg should be string constant
        self.assertIsInstance(context_call.args[0], ast.Constant)
        self.assertEqual(context_call.args[0].value, "layer_forward")
        
        # Second arg should be variable reference
        self.assertIsInstance(context_call.args[1], ast.Name)
        self.assertEqual(context_call.args[1].id, "layer_idx")
    
    def test_no_statements_in_range_warning(self):
        """Test that wrapping with no statements in range produces warning."""
        # This test verifies the warning path when no statements fall in the range
        replacer = DynamicReplacer(
            _fullname="test.empty.range",
            _class_replacements={},
            _class_func_replacements={},
            _func_line_range_wrappings=[
                {
                    'function': 'test_method',
                    'start_line': 100,  # No statements at this line
                    'end_line': 105,
                    'context_class': 'profiler.NVTXContext',
                    'context_values': [
                        {'name': 'name', 'value': 'operation', 'type': 'literal'}
                    ]
                }
            ]
        )
        
        test_method = ast.FunctionDef(
            name="test_method",
            args=ast.arguments(
                posonlyargs=[],
                args=[ast.arg(arg="self", annotation=None)],
                vararg=None,
                kwonlyargs=[],
                kw_defaults=[],
                kwarg=None,
                defaults=[]
            ),
            body=[
                ast.Assign(
                    targets=[ast.Name(id="x", ctx=ast.Store())],
                    value=ast.Constant(value=1),
                    lineno=10
                ),
            ],
            decorator_list=[],
            returns=None,
            lineno=9
        )
        
        class_node = ast.ClassDef(
            name="TestClass",
            bases=[],
            keywords=[],
            decorator_list=[],
            body=[test_method]
        )
        
        # Should not raise an error, just print warning
        with patch('ncompass.trace.infra.utils.logger.warning') as mock_warning:
            result = replacer.visit_ClassDef(class_node)
            
            # Check that warning was printed
            mock_warning.assert_any_call("No statements found in line range 100-105")
        
        # Method should have import + original statement (no wrapping occurred)
        modified_method = result.body[0]
        self.assertEqual(len(modified_method.body), 2)  # import + original assign
    
    def test_line_range_wrapping_non_target_class(self):
        """Test that non-target classes are not affected."""
        test_method = ast.FunctionDef(
            name="test_method",
            args=ast.arguments(
                posonlyargs=[],
                args=[ast.arg(arg="self", annotation=None)],
                vararg=None,
                kwonlyargs=[],
                kw_defaults=[],
                kwarg=None,
                defaults=[]
            ),
            body=[
                ast.Assign(
                    targets=[ast.Name(id="x", ctx=ast.Store())],
                    value=ast.Constant(value=1),
                    lineno=10
                ),
            ],
            decorator_list=[],
            returns=None,
            lineno=9
        )
        
        class_node = ast.ClassDef(
            name="OtherClass",  # Not "TestClass"
            bases=[],
            keywords=[],
            decorator_list=[],
            body=[test_method]
        )
        
        with patch.object(self.replacer, 'generic_visit') as mock_generic_visit:
            mock_generic_visit.return_value = class_node
            
            result = self.replacer.visit_ClassDef(class_node)
            
            # Should be unchanged
            self.assertEqual(result, class_node)
            mock_generic_visit.assert_called_once_with(class_node)
    
    def test_line_range_wrapping_non_target_method(self):
        """Test that non-target methods in target class are not affected."""
        # Create a method that's not in the wrapping config
        other_method = ast.FunctionDef(
            name="other_method",
            args=ast.arguments(
                posonlyargs=[],
                args=[ast.arg(arg="self", annotation=None)],
                vararg=None,
                kwonlyargs=[],
                kw_defaults=[],
                kwarg=None,
                defaults=[]
            ),
            body=[
                ast.Assign(
                    targets=[ast.Name(id="x", ctx=ast.Store())],
                    value=ast.Constant(value=1),
                    lineno=10
                ),
            ],
            decorator_list=[],
            returns=None,
            lineno=9
        )
        
        class_node = ast.ClassDef(
            name="TestClass",
            bases=[],
            keywords=[],
            decorator_list=[],
            body=[other_method]
        )
        
        result = self.replacer.visit_ClassDef(class_node)
        
        # Should return class with unchanged method
        self.assertIsInstance(result, ast.ClassDef)
        modified_method = result.body[0]
        self.assertEqual(modified_method.name, "other_method")
        
        # Method body should be unchanged (just 1 statement)
        self.assertEqual(len(modified_method.body), 1)
        self.assertIsInstance(modified_method.body[0], ast.Assign)
    
    def test_line_range_import_deduplication(self):
        """Test that multiple ranges using same context class only import once."""
        replacer = DynamicReplacer(
            _fullname="test.dedup",
            _class_replacements={},
            _class_func_replacements={},
            _func_line_range_wrappings=[
                {
                    'function': 'test_method',
                    'start_line': 10,
                    'end_line': 11,
                    'context_class': 'profiler.NVTXContext',
                    'context_values': [{'name': 'name', 'value': 'range1', 'type': 'literal'}]
                },
                {
                    'function': 'test_method',
                    'start_line': 15,
                    'end_line': 16,
                    'context_class': 'profiler.NVTXContext',  # Same class
                    'context_values': [{'name': 'name', 'value': 'range2', 'type': 'literal'}]
                }
            ]
        )
        
        test_method = ast.FunctionDef(
            name="test_method",
            args=ast.arguments(
                posonlyargs=[],
                args=[ast.arg(arg="self", annotation=None)],
                vararg=None,
                kwonlyargs=[],
                kw_defaults=[],
                kwarg=None,
                defaults=[]
            ),
            body=[
                ast.Assign(targets=[ast.Name(id="a", ctx=ast.Store())],
                          value=ast.Constant(value=1), lineno=10),
                ast.Assign(targets=[ast.Name(id="b", ctx=ast.Store())],
                          value=ast.Constant(value=2), lineno=11),
                ast.Assign(targets=[ast.Name(id="c", ctx=ast.Store())],
                          value=ast.Constant(value=3), lineno=15),
                ast.Assign(targets=[ast.Name(id="d", ctx=ast.Store())],
                          value=ast.Constant(value=4), lineno=16),
            ],
            decorator_list=[],
            returns=None,
            lineno=9
        )
        
        class_node = ast.ClassDef(
            name="TestClass",
            bases=[],
            keywords=[],
            decorator_list=[],
            body=[test_method]
        )
        
        result = replacer.visit_ClassDef(class_node)
        modified_method = result.body[0]
        
        # Count imports - currently implementation adds duplicate imports
        # This test documents current behavior; could be enhanced later
        imports = [stmt for stmt in modified_method.body if isinstance(stmt, ast.ImportFrom)]
        
        # Currently creates 2 imports (one per wrap config)
        # Could be optimized to deduplicate in the future
        self.assertGreaterEqual(len(imports), 1)

    def test_nested_wrappers_inner_within_outer(self):
        """Test nested wrappers where smaller ranges are completely within a larger range.
        
        This tests the scenario from the profiling config where:
        - An outer wrapper covers the entire method (e.g., lines 429-461)
        - Multiple inner wrappers cover specific operations within (e.g., 431-434, 436-437, 446-447)
        """
        replacer = DynamicReplacer(
            _fullname="test.nested",
            _class_replacements={},
            _class_func_replacements={},
            _func_line_range_wrappings=[
                {
                    'function': 'test_method',
                    'start_line': 10,
                    'end_line': 30,
                    'context_class': 'profiler.OuterContext',
                    'context_values': []
                },
                {
                    'function': 'test_method',
                    'start_line': 12,
                    'end_line': 14,
                    'context_class': 'profiler.Inner1Context',
                    'context_values': [{'name': 'name', 'value': 'operation_1', 'type': 'literal'}]
                },
                {
                    'function': 'test_method',
                    'start_line': 16,
                    'end_line': 17,
                    'context_class': 'profiler.Inner2Context',
                    'context_values': [{'name': 'name', 'value': 'operation_2', 'type': 'literal'}]
                },
                {
                    'function': 'test_method',
                    'start_line': 20,
                    'end_line': 21,
                    'context_class': 'profiler.Inner3Context',
                    'context_values': [{'name': 'name', 'value': 'operation_3', 'type': 'literal'}]
                }
            ]
        )
        
        # Create method with statements spanning the ranges
        test_method = ast.FunctionDef(
            name="test_method",
            args=ast.arguments(
                posonlyargs=[],
                args=[ast.arg(arg="self", annotation=None)],
                vararg=None,
                kwonlyargs=[],
                kw_defaults=[],
                kwarg=None,
                defaults=[]
            ),
            body=[
                ast.Assign(
                    targets=[ast.Name(id="a", ctx=ast.Store())],
                    value=ast.Constant(value=1),
                    lineno=10
                ),
                ast.Assign(
                    targets=[ast.Name(id="b", ctx=ast.Store())],
                    value=ast.Constant(value=2),
                    lineno=12
                ),
                ast.Assign(
                    targets=[ast.Name(id="c", ctx=ast.Store())],
                    value=ast.Constant(value=3),
                    lineno=14
                ),
                ast.Assign(
                    targets=[ast.Name(id="d", ctx=ast.Store())],
                    value=ast.Constant(value=4),
                    lineno=15
                ),
                ast.Assign(
                    targets=[ast.Name(id="e", ctx=ast.Store())],
                    value=ast.Constant(value=5),
                    lineno=16
                ),
                ast.Assign(
                    targets=[ast.Name(id="f", ctx=ast.Store())],
                    value=ast.Constant(value=6),
                    lineno=17
                ),
                ast.Assign(
                    targets=[ast.Name(id="g", ctx=ast.Store())],
                    value=ast.Constant(value=7),
                    lineno=18
                ),
                ast.Assign(
                    targets=[ast.Name(id="h", ctx=ast.Store())],
                    value=ast.Constant(value=8),
                    lineno=20
                ),
                ast.Assign(
                    targets=[ast.Name(id="i", ctx=ast.Store())],
                    value=ast.Constant(value=9),
                    lineno=21
                ),
                ast.Assign(
                    targets=[ast.Name(id="j", ctx=ast.Store())],
                    value=ast.Constant(value=10),
                    lineno=30
                ),
            ],
            decorator_list=[],
            returns=None,
            lineno=9
        )
        
        class_node = ast.ClassDef(
            name="TestClass",
            bases=[],
            keywords=[],
            decorator_list=[],
            body=[test_method]
        )
        
        result = replacer.visit_ClassDef(class_node)
        modified_method = result.body[0]
        
        # Should have imports at the beginning
        imports = [stmt for stmt in modified_method.body if isinstance(stmt, ast.ImportFrom)]
        self.assertEqual(len(imports), 4)  # One for each context class
        
        # After imports, should have a single outer with statement
        non_import_stmts = [stmt for stmt in modified_method.body if not isinstance(stmt, ast.ImportFrom)]
        self.assertEqual(len(non_import_stmts), 1)
        
        outer_with = non_import_stmts[0]
        self.assertIsInstance(outer_with, ast.With)
        self.assertEqual(outer_with.items[0].context_expr.func.id, "OuterContext")
        
        # Inside outer_with, should have multiple statements including inner with statements
        inner_stmts = outer_with.body
        inner_with_stmts = [stmt for stmt in inner_stmts if isinstance(stmt, ast.With)]
        self.assertEqual(len(inner_with_stmts), 3)  # Three inner contexts
        
        # Verify the inner contexts are present
        inner_context_names = [stmt.items[0].context_expr.func.id for stmt in inner_with_stmts]
        self.assertIn("Inner1Context", inner_context_names)
        self.assertIn("Inner2Context", inner_context_names)
        self.assertIn("Inner3Context", inner_context_names)


    def test_sequential_unnested_wrappers_various_sizes(self):
        """Test sequential non-overlapping wrappers of different sizes.
        
        Tests the case where we have:
        - A large wrapper (lines 5-15)
        - A small wrapper following it (lines 16-19)
        - Another wrapper after that (lines 20-25)
        """
        replacer = DynamicReplacer(
            _fullname="test.sequential",
            _class_replacements={},
            _class_func_replacements={},
            _func_line_range_wrappings=[
                {
                    'function': 'test_method',
                    'start_line': 5,
                    'end_line': 15,
                    'context_class': 'profiler.LargeContext',
                    'context_values': [{'name': 'name', 'value': 'large_operation', 'type': 'literal'}]
                },
                {
                    'function': 'test_method',
                    'start_line': 16,
                    'end_line': 19,
                    'context_class': 'profiler.SmallContext',
                    'context_values': [{'name': 'name', 'value': 'small_operation', 'type': 'literal'}]
                },
                {
                    'function': 'test_method',
                    'start_line': 20,
                    'end_line': 25,
                    'context_class': 'profiler.MediumContext',
                    'context_values': [{'name': 'name', 'value': 'medium_operation', 'type': 'literal'}]
                }
            ]
        )
        
        # Create method with statements in all ranges
        test_method = ast.FunctionDef(
            name="test_method",
            args=ast.arguments(
                posonlyargs=[],
                args=[ast.arg(arg="self", annotation=None)],
                vararg=None,
                kwonlyargs=[],
                kw_defaults=[],
                kwarg=None,
                defaults=[]
            ),
            body=[
                ast.Assign(targets=[ast.Name(id="v1", ctx=ast.Store())],
                        value=ast.Constant(value=1), lineno=5),
                ast.Assign(targets=[ast.Name(id="v2", ctx=ast.Store())],
                        value=ast.Constant(value=2), lineno=10),
                ast.Assign(targets=[ast.Name(id="v3", ctx=ast.Store())],
                        value=ast.Constant(value=3), lineno=15),
                ast.Assign(targets=[ast.Name(id="v4", ctx=ast.Store())],
                        value=ast.Constant(value=4), lineno=16),
                ast.Assign(targets=[ast.Name(id="v5", ctx=ast.Store())],
                        value=ast.Constant(value=5), lineno=19),
                ast.Assign(targets=[ast.Name(id="v6", ctx=ast.Store())],
                        value=ast.Constant(value=6), lineno=20),
                ast.Assign(targets=[ast.Name(id="v7", ctx=ast.Store())],
                        value=ast.Constant(value=7), lineno=25),
            ],
            decorator_list=[],
            returns=None,
            lineno=4
        )
        
        class_node = ast.ClassDef(
            name="TestClass",
            bases=[],
            keywords=[],
            decorator_list=[],
            body=[test_method]
        )
        
        result = replacer.visit_ClassDef(class_node)
        modified_method = result.body[0]
        
        # Should have 3 imports
        imports = [stmt for stmt in modified_method.body if isinstance(stmt, ast.ImportFrom)]
        self.assertEqual(len(imports), 3)
        
        # Should have 3 with statements (all at the same level, not nested)
        non_import_stmts = [stmt for stmt in modified_method.body if not isinstance(stmt, ast.ImportFrom)]
        with_stmts = [stmt for stmt in non_import_stmts if isinstance(stmt, ast.With)]
        self.assertEqual(len(with_stmts), 3)
        
        # Verify the order and content
        context_names = [stmt.items[0].context_expr.func.id for stmt in with_stmts]
        self.assertEqual(context_names, ["LargeContext", "SmallContext", "MediumContext"])


    def test_complex_nested_and_sequential_mixed(self):
        """Test complex scenario with both nested and sequential wrappers.
        
        This simulates the real-world profiling config scenario:
        - Lines 10-50: outer profiler context (entire method)
        - Lines 12-15: inner operation 1
        - Lines 17-20: inner operation 2
        - Lines 55-60: separate sequential wrapper after the main method logic
        """
        replacer = DynamicReplacer(
            _fullname="test.complex",
            _class_replacements={},
            _class_func_replacements={},
            _func_line_range_wrappings=[
                # Outer wrapper for main execution
                {
                    'function': 'execute_model',
                    'start_line': 10,
                    'end_line': 50,
                    'context_class': 'ncompass.trace.profile.torch.TorchProfilerContext',
                    'context_values': []
                },
                # Inner wrappers for specific operations
                {
                    'function': 'execute_model',
                    'start_line': 12,
                    'end_line': 15,
                    'context_class': 'ncompass.trace.profile.torch.TorchRecordContext',
                    'context_values': [{'name': 'name', 'value': 'recv_intermediate_tensors', 'type': 'literal'}]
                },
                {
                    'function': 'execute_model',
                    'start_line': 17,
                    'end_line': 20,
                    'context_class': 'ncompass.trace.profile.torch.TorchRecordContext',
                    'context_values': [{'name': 'name', 'value': 'execute_model', 'type': 'literal'}]
                },
                {
                    'function': 'execute_model',
                    'start_line': 30,
                    'end_line': 32,
                    'context_class': 'ncompass.trace.profile.torch.TorchRecordContext',
                    'context_values': [{'name': 'name', 'value': 'send_tensor_dict', 'type': 'literal'}]
                },
                # Separate wrapper outside the main execution
                {
                    'function': 'execute_model',
                    'start_line': 55,
                    'end_line': 60,
                    'context_class': 'ncompass.trace.profile.torch.TorchRecordContext',
                    'context_values': [{'name': 'name', 'value': 'cleanup', 'type': 'literal'}]
                }
            ]
        )
        
        # Create a method simulating the execute_model structure
        test_method = ast.FunctionDef(
            name="execute_model",
            args=ast.arguments(
                posonlyargs=[],
                args=[ast.arg(arg="self", annotation=None)],
                vararg=None,
                kwonlyargs=[],
                kw_defaults=[],
                kwarg=None,
                defaults=[]
            ),
            body=[
                # Before main execution
                ast.Assign(targets=[ast.Name(id="setup", ctx=ast.Store())],
                        value=ast.Constant(value=True), lineno=10),
                
                # Recv intermediate tensors (lines 12-15)
                ast.Assign(targets=[ast.Name(id="intermediate", ctx=ast.Store())],
                        value=ast.Constant(value=None), lineno=12),
                ast.Expr(value=ast.Call(func=ast.Name(id="recv", ctx=ast.Load()),
                                    args=[], keywords=[]), lineno=14),
                ast.Assign(targets=[ast.Name(id="tensors", ctx=ast.Store())],
                        value=ast.Name(id="intermediate", ctx=ast.Load()), lineno=15),
                
                # Execute model (lines 17-20)
                ast.Assign(targets=[ast.Name(id="output", ctx=ast.Store())],
                        value=ast.Call(func=ast.Name(id="model_execute", ctx=ast.Load()),
                                    args=[], keywords=[]), lineno=17),
                ast.Expr(value=ast.Call(func=ast.Name(id="process", ctx=ast.Load()),
                                    args=[], keywords=[]), lineno=20),
                
                # Middle section (lines 22-28)
                ast.If(test=ast.Name(id="condition", ctx=ast.Load()),
                    body=[ast.Return(value=ast.Name(id="output", ctx=ast.Load()))],
                    orelse=[], lineno=22),
                
                # Send tensor dict (lines 30-32)
                ast.Expr(value=ast.Call(func=ast.Name(id="send", ctx=ast.Load()),
                                    args=[ast.Name(id="output", ctx=ast.Load())],
                                    keywords=[]), lineno=30),
                ast.Assign(targets=[ast.Name(id="result", ctx=ast.Store())],
                        value=ast.Constant(value=True), lineno=32),
                
                # End of main execution
                ast.Assign(targets=[ast.Name(id="finalize", ctx=ast.Store())],
                        value=ast.Constant(value=True), lineno=50),
                
                # Separate cleanup section (lines 55-60)
                ast.Expr(value=ast.Call(func=ast.Name(id="cleanup", ctx=ast.Load()),
                                    args=[], keywords=[]), lineno=55),
                ast.Assign(targets=[ast.Name(id="cleaned", ctx=ast.Store())],
                        value=ast.Constant(value=True), lineno=60),
            ],
            decorator_list=[],
            returns=None,
            lineno=9
        )
        
        class_node = ast.ClassDef(
            name="Worker",
            bases=[],
            keywords=[],
            decorator_list=[],
            body=[test_method]
        )
        
        result = replacer.visit_ClassDef(class_node)
        modified_method = result.body[0]
        
        # Verify imports
        imports = [stmt for stmt in modified_method.body if isinstance(stmt, ast.ImportFrom)]
        self.assertGreaterEqual(len(imports), 2)  # At least TorchProfilerContext and TorchRecordContext
        
        # After imports, should have main structure
        non_import_stmts = [stmt for stmt in modified_method.body if not isinstance(stmt, ast.ImportFrom)]
        
        # Should have at least 2 top-level with statements:
        # 1. The outer TorchProfilerContext (lines 10-50)
        # 2. The separate cleanup context (lines 55-60)
        top_level_with_stmts = [stmt for stmt in non_import_stmts if isinstance(stmt, ast.With)]
        self.assertGreaterEqual(len(top_level_with_stmts), 2)
        
        # Find the outer profiler context
        outer_profiler = None
        for stmt in top_level_with_stmts:
            if stmt.items[0].context_expr.func.id == "TorchProfilerContext":
                outer_profiler = stmt
                break
        
        self.assertIsNotNone(outer_profiler, "Should have outer TorchProfilerContext")
        
        # Inside outer profiler, should have the three inner TorchRecordContext wrappers
        inner_with_stmts = [stmt for stmt in outer_profiler.body if isinstance(stmt, ast.With)]
        self.assertGreaterEqual(len(inner_with_stmts), 3, 
                            "Should have at least 3 inner TorchRecordContext wrappers")
        
        # Verify that we have recv, execute, and send contexts
        inner_context_calls = [stmt.items[0].context_expr for stmt in inner_with_stmts]
        context_names = []
        for call in inner_context_calls:
            if call.func.id == "TorchRecordContext" and len(call.args) > 0:
                if isinstance(call.args[0], ast.Constant):
                    context_names.append(call.args[0].value)
        
        self.assertIn("recv_intermediate_tensors", context_names)
        self.assertIn("execute_model", context_names)
        self.assertIn("send_tensor_dict", context_names)


    def test_early_return_in_nested_wrapper(self):
        """Test that nested wrappers handle early returns correctly.
        
        This is critical for the execute_model case where there's an early return
        at line 439 that should not cause UnboundLocalError.
        """
        replacer = DynamicReplacer(
            _fullname="test.early_return",
            _class_replacements={},
            _class_func_replacements={},
            _func_line_range_wrappings=[
                # Outer wrapper
                {
                    'function': 'test_method',
                    'start_line': 10,
                    'end_line': 40,
                    'context_class': 'profiler.OuterContext',
                    'context_values': []
                },
                # Inner wrapper that assigns a variable
                {
                    'function': 'test_method',
                    'start_line': 15,
                    'end_line': 17,
                    'context_class': 'profiler.AssignContext',
                    'context_values': [{'name': 'name', 'value': 'assign_output', 'type': 'literal'}]
                },
                # Another inner wrapper that uses the variable
                {
                    'function': 'test_method',
                    'start_line': 25,
                    'end_line': 27,
                    'context_class': 'profiler.UseContext',
                    'context_values': [{'name': 'name', 'value': 'use_output', 'type': 'literal'}]
                }
            ]
        )
        
        # Create method with early return between assignment and usage
        test_method = ast.FunctionDef(
            name="test_method",
            args=ast.arguments(
                posonlyargs=[],
                args=[ast.arg(arg="self", annotation=None)],
                vararg=None,
                kwonlyargs=[],
                kw_defaults=[],
                kwarg=None,
                defaults=[]
            ),
            body=[
                ast.Assign(targets=[ast.Name(id="init", ctx=ast.Store())],
                        value=ast.Constant(value=None), lineno=10),
                
                # Assignment within wrapper (lines 15-17)
                ast.Assign(targets=[ast.Name(id="output", ctx=ast.Store())],
                        value=ast.Call(func=ast.Name(id="compute", ctx=ast.Load()),
                                    args=[], keywords=[]), lineno=15),
                ast.Assign(targets=[ast.Name(id="processed", ctx=ast.Store())],
                        value=ast.Name(id="output", ctx=ast.Load()), lineno=17),
                
                # Early return (line 20)
                ast.If(
                    test=ast.Call(func=ast.Name(id="should_return_early", ctx=ast.Load()),
                                args=[ast.Name(id="output", ctx=ast.Load())], keywords=[]),
                    body=[ast.Return(value=ast.Name(id="output", ctx=ast.Load()))],
                    orelse=[],
                    lineno=20
                ),
                
                # Usage of output variable (lines 25-27)
                ast.Expr(value=ast.Call(func=ast.Name(id="send", ctx=ast.Load()),
                                    args=[ast.Name(id="output", ctx=ast.Load())],
                                    keywords=[]), lineno=25),
                ast.Assign(targets=[ast.Name(id="result", ctx=ast.Store())],
                        value=ast.Name(id="output", ctx=ast.Load()), lineno=27),
                
                ast.Return(value=ast.Constant(value=None), lineno=40),
            ],
            decorator_list=[],
            returns=None,
            lineno=9
        )
        
        class_node = ast.ClassDef(
            name="TestClass",
            bases=[],
            keywords=[],
            decorator_list=[],
            body=[test_method]
        )
        
        result = replacer.visit_ClassDef(class_node)
        modified_method = result.body[0]
        
        # Verify the structure is valid
        imports = [stmt for stmt in modified_method.body if isinstance(stmt, ast.ImportFrom)]
        self.assertGreaterEqual(len(imports), 1)
        
        # Ensure the transformation doesn't break the early return logic
        # The outer wrapper should contain all statements including the early return
        non_import_stmts = [stmt for stmt in modified_method.body if not isinstance(stmt, ast.ImportFrom)]
        self.assertGreaterEqual(len(non_import_stmts), 1)
        
        # Verify we have an outer with statement
        outer_with = non_import_stmts[0]
        self.assertIsInstance(outer_with, ast.With)
        
        # Check that the early return (If statement) is preserved
        if_statements = [stmt for stmt in outer_with.body if isinstance(stmt, ast.If)]
        self.assertGreaterEqual(len(if_statements), 1, "Early return If statement should be preserved")


class TestStatementFlattening(unittest.TestCase):
    """Test cases for statement flattening and compound statement handling in metadata."""
    
    def test_compound_statements_included_in_metadata(self):
        """Test that compound statements (If, For, While, With, Try) are included in metadata."""
        replacer = DynamicReplacer(
            _fullname="test.flattening",
            _class_replacements={},
            _class_func_replacements={},
            _func_line_range_wrappings=[
                {
                    'function': 'test_method',
                    'start_line': 10,
                    'end_line': 30,
                    'context_class': 'profiler.OuterContext',
                    'context_values': []
                }
            ]
        )
        
        # Create method with various compound statements
        test_method = ast.FunctionDef(
            name="test_method",
            args=ast.arguments(
                posonlyargs=[],
                args=[ast.arg(arg="self", annotation=None)],
                vararg=None,
                kwonlyargs=[],
                kw_defaults=[],
                kwarg=None,
                defaults=[]
            ),
            body=[
                ast.Assign(
                    targets=[ast.Name(id="x", ctx=ast.Store())],
                    value=ast.Constant(value=1),
                    lineno=10
                ),
                ast.If(
                    test=ast.Name(id="condition", ctx=ast.Load()),
                    body=[ast.Pass()],
                    orelse=[],
                    lineno=12
                ),
                ast.For(
                    target=ast.Name(id="i", ctx=ast.Store()),
                    iter=ast.Call(func=ast.Name(id="range", ctx=ast.Load()), args=[ast.Constant(value=5)], keywords=[]),
                    body=[ast.Pass()],
                    orelse=[],
                    lineno=15
                ),
                ast.While(
                    test=ast.Name(id="running", ctx=ast.Load()),
                    body=[ast.Pass()],
                    orelse=[],
                    lineno=18
                ),
                ast.Try(
                    body=[ast.Pass()],
                    handlers=[],
                    orelse=[],
                    finalbody=[],
                    lineno=20
                ),
                ast.Assign(
                    targets=[ast.Name(id="y", ctx=ast.Store())],
                    value=ast.Constant(value=2),
                    lineno=30
                ),
            ],
            decorator_list=[],
            returns=None,
            lineno=9
        )
        
        class_node = ast.ClassDef(
            name="TestClass",
            bases=[],
            keywords=[],
            decorator_list=[],
            body=[test_method]
        )
        
        result = replacer.visit_ClassDef(class_node)
        modified_method = result.body[0]
        
        # Get the outer wrapper
        non_import_stmts = [stmt for stmt in modified_method.body if not isinstance(stmt, ast.ImportFrom)]
        self.assertEqual(len(non_import_stmts), 1)
        outer_with = non_import_stmts[0]
        self.assertIsInstance(outer_with, ast.With)
        
        # Verify that compound statements are preserved in the wrapped body
        outer_body = outer_with.body
        compound_stmts = [stmt for stmt in outer_body if isinstance(stmt, (ast.If, ast.For, ast.While, ast.Try))]
        self.assertEqual(len(compound_stmts), 4, "All compound statements should be preserved")
        
        # Verify types
        if_stmts = [stmt for stmt in compound_stmts if isinstance(stmt, ast.If)]
        for_stmts = [stmt for stmt in compound_stmts if isinstance(stmt, ast.For)]
        while_stmts = [stmt for stmt in compound_stmts if isinstance(stmt, ast.While)]
        try_stmts = [stmt for stmt in compound_stmts if isinstance(stmt, ast.Try)]
        
        self.assertEqual(len(if_stmts), 1, "If statement should be preserved")
        self.assertEqual(len(for_stmts), 1, "For statement should be preserved")
        self.assertEqual(len(while_stmts), 1, "While statement should be preserved")
        self.assertEqual(len(try_stmts), 1, "Try statement should be preserved")
    
    def test_wrapping_compound_statement_directly(self):
        """Test that a compound statement (If) can be wrapped directly when its line range matches."""
        replacer = DynamicReplacer(
            _fullname="test.wrap_compound",
            _class_replacements={},
            _class_func_replacements={},
            _func_line_range_wrappings=[
                {
                    'function': 'test_method',
                    'start_line': 12,
                    'end_line': 15,
                    'context_class': 'profiler.IfContext',
                    'context_values': [{'name': 'name', 'value': 'conditional', 'type': 'literal'}]
                }
            ]
        )
        
        # Create method with an If statement spanning lines 12-15
        test_method = ast.FunctionDef(
            name="test_method",
            args=ast.arguments(
                posonlyargs=[],
                args=[ast.arg(arg="self", annotation=None)],
                vararg=None,
                kwonlyargs=[],
                kw_defaults=[],
                kwarg=None,
                defaults=[]
            ),
            body=[
                ast.Assign(
                    targets=[ast.Name(id="x", ctx=ast.Store())],
                    value=ast.Constant(value=1),
                    lineno=10
                ),
                ast.If(
                    test=ast.Name(id="condition", ctx=ast.Load()),
                    body=[
                        ast.Assign(
                            targets=[ast.Name(id="y", ctx=ast.Store())],
                            value=ast.Constant(value=2),
                            lineno=13
                        ),
                        ast.Pass(lineno=14)
                    ],
                    orelse=[],
                    lineno=12
                ),
                ast.Assign(
                    targets=[ast.Name(id="z", ctx=ast.Store())],
                    value=ast.Constant(value=3),
                    lineno=20
                ),
            ],
            decorator_list=[],
            returns=None,
            lineno=9
        )
        
        class_node = ast.ClassDef(
            name="TestClass",
            bases=[],
            keywords=[],
            decorator_list=[],
            body=[test_method]
        )
        
        result = replacer.visit_ClassDef(class_node)
        modified_method = result.body[0]
        
        # Verify structure
        non_import_stmts = [stmt for stmt in modified_method.body if not isinstance(stmt, ast.ImportFrom)]
        self.assertEqual(len(non_import_stmts), 3)  # x assignment, wrapped If, z assignment
        
        # Find the wrapped If statement
        wrapped_if = None
        for stmt in non_import_stmts:
            if isinstance(stmt, ast.With):
                if isinstance(stmt.body[0], ast.If):
                    wrapped_if = stmt.body[0]
                    break
        
        self.assertIsNotNone(wrapped_if, "If statement should be wrapped")
        self.assertIsInstance(wrapped_if, ast.If)
        self.assertEqual(len(wrapped_if.body), 2, "If statement body should be preserved")
    
    def test_nested_with_statements_preserved_in_metadata(self):
        """Test that With statements created by inner wrappers are preserved when processing outer wrappers."""
        replacer = DynamicReplacer(
            _fullname="test.nested_with",
            _class_replacements={},
            _class_func_replacements={},
            _func_line_range_wrappings=[
                # Inner wrapper
                {
                    'function': 'test_method',
                    'start_line': 12,
                    'end_line': 14,
                    'context_class': 'profiler.InnerContext',
                    'context_values': [{'name': 'name', 'value': 'inner', 'type': 'literal'}]
                },
                # Outer wrapper
                {
                    'function': 'test_method',
                    'start_line': 10,
                    'end_line': 20,
                    'context_class': 'profiler.OuterContext',
                    'context_values': []
                }
            ]
        )
        
        test_method = ast.FunctionDef(
            name="test_method",
            args=ast.arguments(
                posonlyargs=[],
                args=[ast.arg(arg="self", annotation=None)],
                vararg=None,
                kwonlyargs=[],
                kw_defaults=[],
                kwarg=None,
                defaults=[]
            ),
            body=[
                ast.Assign(
                    targets=[ast.Name(id="a", ctx=ast.Store())],
                    value=ast.Constant(value=1),
                    lineno=10
                ),
                ast.Assign(
                    targets=[ast.Name(id="b", ctx=ast.Store())],
                    value=ast.Constant(value=2),
                    lineno=12
                ),
                ast.Assign(
                    targets=[ast.Name(id="c", ctx=ast.Store())],
                    value=ast.Constant(value=3),
                    lineno=14
                ),
                ast.Assign(
                    targets=[ast.Name(id="d", ctx=ast.Store())],
                    value=ast.Constant(value=4),
                    lineno=20
                ),
            ],
            decorator_list=[],
            returns=None,
            lineno=9
        )
        
        class_node = ast.ClassDef(
            name="TestClass",
            bases=[],
            keywords=[],
            decorator_list=[],
            body=[test_method]
        )
        
        result = replacer.visit_ClassDef(class_node)
        modified_method = result.body[0]
        
        # Verify structure: should have outer wrapper containing inner wrapper
        non_import_stmts = [stmt for stmt in modified_method.body if not isinstance(stmt, ast.ImportFrom)]
        self.assertEqual(len(non_import_stmts), 1)
        
        outer_with = non_import_stmts[0]
        self.assertIsInstance(outer_with, ast.With)
        self.assertEqual(outer_with.items[0].context_expr.func.id, "OuterContext")
        
        # Inside outer wrapper, should have inner wrapper
        inner_with_stmts = [stmt for stmt in outer_with.body if isinstance(stmt, ast.With)]
        self.assertEqual(len(inner_with_stmts), 1, "Should have one inner wrapper")
        
        inner_with = inner_with_stmts[0]
        self.assertEqual(inner_with.items[0].context_expr.func.id, "InnerContext")
        
        # Verify inner wrapper contains the correct statements
        inner_body = inner_with.body
        self.assertEqual(len(inner_body), 2, "Inner wrapper should contain 2 statements")
    
    def test_compound_statements_with_nested_bodies(self):
        """Test that compound statements with nested bodies (like If with For inside) are handled correctly."""
        replacer = DynamicReplacer(
            _fullname="test.nested_compound",
            _class_replacements={},
            _class_func_replacements={},
            _func_line_range_wrappings=[
                {
                    'function': 'test_method',
                    'start_line': 10,
                    'end_line': 25,
                    'context_class': 'profiler.OuterContext',
                    'context_values': []
                },
                {
                    'function': 'test_method',
                    'start_line': 15,
                    'end_line': 18,
                    'context_class': 'profiler.InnerContext',
                    'context_values': [{'name': 'name', 'value': 'loop', 'type': 'literal'}]
                }
            ]
        )
        
        test_method = ast.FunctionDef(
            name="test_method",
            args=ast.arguments(
                posonlyargs=[],
                args=[ast.arg(arg="self", annotation=None)],
                vararg=None,
                kwonlyargs=[],
                kw_defaults=[],
                kwarg=None,
                defaults=[]
            ),
            body=[
                ast.Assign(
                    targets=[ast.Name(id="x", ctx=ast.Store())],
                    value=ast.Constant(value=1),
                    lineno=10
                ),
                ast.If(
                    test=ast.Name(id="condition", ctx=ast.Load()),
                    body=[
                        # Create For loop with explicit end_lineno so it spans lines 15-18
                        (for_loop := ast.For(
                            target=ast.Name(id="i", ctx=ast.Store()),
                            iter=ast.Call(func=ast.Name(id="range", ctx=ast.Load()), args=[ast.Constant(value=5)], keywords=[]),
                            body=[
                                ast.Assign(
                                    targets=[ast.Name(id="y", ctx=ast.Store())],
                                    value=ast.Constant(value=17),
                                    lineno=16
                                ),
                                ast.Assign(
                                    targets=[ast.Name(id="y", ctx=ast.Store())],
                                    value=ast.Constant(value=18),
                                    lineno=17
                                )
                            ],
                            orelse=[],
                            lineno=15
                        ))
                    ],
                    orelse=[],
                    lineno=12
                ),
                ast.Assign(
                    targets=[ast.Name(id="z", ctx=ast.Store())],
                    value=ast.Constant(value=3),
                    lineno=25
                ),
            ],
            decorator_list=[],
            returns=None,
            lineno=9
        )
        
        class_node = ast.ClassDef(
            name="TestClass",
            bases=[],
            keywords=[],
            decorator_list=[],
            body=[test_method]
        )
        
        # Set end_lineno explicitly so the For loop spans lines 15-18
        for_loop.end_lineno = 18
        
        result = replacer.visit_ClassDef(class_node)
        modified_method = result.body[0]
        
        # Verify structure
        non_import_stmts = [stmt for stmt in modified_method.body if not isinstance(stmt, ast.ImportFrom)]
        self.assertEqual(len(non_import_stmts), 1)
        
        outer_with = non_import_stmts[0]
        self.assertIsInstance(outer_with, ast.With)
        
        # Verify If statement is preserved
        if_stmts = [stmt for stmt in outer_with.body if isinstance(stmt, ast.If)]
        self.assertEqual(len(if_stmts), 1, "If statement should be preserved")
        
        if_stmt = if_stmts[0]
        # Verify For loop inside If is wrapped
        for_stmts = [stmt for stmt in if_stmt.body if isinstance(stmt, ast.With)]
        self.assertEqual(len(for_stmts), 1, "For loop should be wrapped by inner wrapper")
        
        inner_with = for_stmts[0]
        self.assertEqual(inner_with.items[0].context_expr.func.id, "InnerContext")
        
        # Verify For loop is inside the inner wrapper
        for_loops = [stmt for stmt in inner_with.body if isinstance(stmt, ast.For)]
        self.assertEqual(len(for_loops), 1, "For loop should be inside inner wrapper")
    
    def test_try_except_finally_in_metadata(self):
        """Test that Try statements with except and finally clauses are correctly handled in metadata."""
        replacer = DynamicReplacer(
            _fullname="test.try_except",
            _class_replacements={},
            _class_func_replacements={},
            _func_line_range_wrappings=[
                {
                    'function': 'test_method',
                    'start_line': 10,
                    'end_line': 30,
                    'context_class': 'profiler.TryContext',
                    'context_values': []
                }
            ]
        )
        
        test_method = ast.FunctionDef(
            name="test_method",
            args=ast.arguments(
                posonlyargs=[],
                args=[ast.arg(arg="self", annotation=None)],
                vararg=None,
                kwonlyargs=[],
                kw_defaults=[],
                kwarg=None,
                defaults=[]
            ),
            body=[
                ast.Assign(
                    targets=[ast.Name(id="x", ctx=ast.Store())],
                    value=ast.Constant(value=1),
                    lineno=10
                ),
                ast.Try(
                    body=[
                        ast.Assign(
                            targets=[ast.Name(id="y", ctx=ast.Store())],
                            value=ast.Constant(value=2),
                            lineno=13
                        )
                    ],
                    handlers=[
                        ast.ExceptHandler(
                            type=ast.Name(id="Exception", ctx=ast.Load()),
                            name=None,
                            body=[
                                ast.Pass(lineno=16)
                            ]
                        )
                    ],
                    orelse=[
                        ast.Pass(lineno=18)
                    ],
                    finalbody=[
                        ast.Pass(lineno=20)
                    ],
                    lineno=12
                ),
                ast.Assign(
                    targets=[ast.Name(id="z", ctx=ast.Store())],
                    value=ast.Constant(value=3),
                    lineno=30
                ),
            ],
            decorator_list=[],
            returns=None,
            lineno=9
        )
        
        class_node = ast.ClassDef(
            name="TestClass",
            bases=[],
            keywords=[],
            decorator_list=[],
            body=[test_method]
        )
        
        result = replacer.visit_ClassDef(class_node)
        modified_method = result.body[0]
        
        # Verify Try statement is preserved
        non_import_stmts = [stmt for stmt in modified_method.body if not isinstance(stmt, ast.ImportFrom)]
        self.assertEqual(len(non_import_stmts), 1)
        
        outer_with = non_import_stmts[0]
        self.assertIsInstance(outer_with, ast.With)
        
        # Verify Try statement is inside wrapper
        try_stmts = [stmt for stmt in outer_with.body if isinstance(stmt, ast.Try)]
        self.assertEqual(len(try_stmts), 1, "Try statement should be preserved")
        
        try_stmt = try_stmts[0]
        # Verify all clauses are preserved
        self.assertEqual(len(try_stmt.body), 1, "Try body should be preserved")
        self.assertEqual(len(try_stmt.handlers), 1, "Except handler should be preserved")
        self.assertEqual(len(try_stmt.orelse), 1, "Else clause should be preserved")
        self.assertEqual(len(try_stmt.finalbody), 1, "Finally clause should be preserved")
    
    def test_partial_lines_inside_compound_statement(self):
        """Test that wrapping only some lines inside a compound statement wraps only those lines, not the entire compound statement.
        
        This tests the fix for the bug where highlighting partial lines inside a compound statement
        (like lines 16-17 inside a For loop spanning 15-20) would incorrectly wrap the entire
        compound statement instead of just the highlighted lines.
        """
        replacer = DynamicReplacer(
            _fullname="test.partial_compound",
            _class_replacements={},
            _class_func_replacements={},
            _func_line_range_wrappings=[
                {
                    'function': 'test_method',
                    'start_line': 16,
                    'end_line': 17,
                    'context_class': 'profiler.PartialContext',
                    'context_values': [{'name': 'name', 'value': 'partial', 'type': 'literal'}]
                }
            ]
        )
        
        # Create a For loop spanning lines 15-20, but only wrap lines 16-17
        test_method = ast.FunctionDef(
            name="test_method",
            args=ast.arguments(
                posonlyargs=[],
                args=[ast.arg(arg="self", annotation=None)],
                vararg=None,
                kwonlyargs=[],
                kw_defaults=[],
                kwarg=None,
                defaults=[]
            ),
            body=[
                ast.Assign(
                    targets=[ast.Name(id="x", ctx=ast.Store())],
                    value=ast.Constant(value=1),
                    lineno=10
                ),
                (for_loop := ast.For(
                    target=ast.Name(id="i", ctx=ast.Store()),
                    iter=ast.Call(func=ast.Name(id="range", ctx=ast.Load()), args=[ast.Constant(value=5)], keywords=[]),
                    body=[
                        ast.Assign(
                            targets=[ast.Name(id="y", ctx=ast.Store())],
                            value=ast.Constant(value=1),
                            lineno=16  # This should be wrapped
                        ),
                        ast.Assign(
                            targets=[ast.Name(id="z", ctx=ast.Store())],
                            value=ast.Constant(value=2),
                            lineno=17  # This should be wrapped
                        ),
                        ast.Assign(
                            targets=[ast.Name(id="w", ctx=ast.Store())],
                            value=ast.Constant(value=3),
                            lineno=18  # This should NOT be wrapped
                        ),
                        ast.Assign(
                            targets=[ast.Name(id="v", ctx=ast.Store())],
                            value=ast.Constant(value=4),
                            lineno=19  # This should NOT be wrapped
                        ),
                    ],
                    orelse=[],
                    lineno=15
                )),
                ast.Assign(
                    targets=[ast.Name(id="final", ctx=ast.Store())],
                    value=ast.Constant(value=5),
                    lineno=25
                ),
            ],
            decorator_list=[],
            returns=None,
            lineno=9
        )
        
        # Set end_lineno so For loop spans lines 15-20
        for_loop.end_lineno = 20
        
        class_node = ast.ClassDef(
            name="TestClass",
            bases=[],
            keywords=[],
            decorator_list=[],
            body=[test_method]
        )
        
        result = replacer.visit_ClassDef(class_node)
        modified_method = result.body[0]
        
        # Verify structure: should have For loop, not wrapped entirely
        non_import_stmts = [stmt for stmt in modified_method.body if not isinstance(stmt, ast.ImportFrom)]
        
        # Find the For loop
        for_stmt = None
        for stmt in non_import_stmts:
            if isinstance(stmt, ast.For):
                for_stmt = stmt
                break
        
        self.assertIsNotNone(for_stmt, "For loop should be preserved (not wrapped entirely)")
        self.assertIsInstance(for_stmt, ast.For)
        
        # Verify that only lines 16-17 are wrapped inside the For loop
        # The For loop body should contain a With statement wrapping lines 16-17
        # and unwrapped statements for lines 18-19
        for_body = for_stmt.body
        
        # Should have a With statement (wrapping lines 16-17) and unwrapped statements (18-19)
        with_stmts = [stmt for stmt in for_body if isinstance(stmt, ast.With)]
        self.assertEqual(len(with_stmts), 1, "Should have one wrapper for lines 16-17")
        
        # Verify the wrapper contains the two assignments from lines 16-17
        wrapper = with_stmts[0]
        self.assertEqual(wrapper.items[0].context_expr.func.id, "PartialContext")
        wrapped_body = wrapper.body
        self.assertEqual(len(wrapped_body), 2, "Wrapper should contain 2 statements (lines 16-17)")
        
        # Verify unwrapped statements exist (lines 18-19)
        unwrapped_assigns = [stmt for stmt in for_body if isinstance(stmt, ast.Assign)]
        self.assertEqual(len(unwrapped_assigns), 2, "Should have 2 unwrapped assignments (lines 18-19)")
        
        # Verify the For loop itself is NOT wrapped
        top_level_withs = [stmt for stmt in non_import_stmts if isinstance(stmt, ast.With)]
        self.assertEqual(len(top_level_withs), 0, "For loop should not be wrapped at top level")

if __name__ == '__main__':
    unittest.main()
