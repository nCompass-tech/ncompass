"""
Tests for ncompass.trace.core.loader module.
"""

import unittest
import ast
import tempfile
import os
from unittest.mock import patch, mock_open, MagicMock

from ncompass.trace.core.loader import _RewritingLoader, RewritingLoader


class MockReplacer(ast.NodeTransformer):
    """Mock replacer class for testing."""
    
    def visit_FunctionDef(self, node):
        # Simple transformation: add prefix to function names
        if node.name == 'test_function':
            node.name = 'replaced_test_function'
        return self.generic_visit(node)


class TestRewritingLoaderBase(unittest.TestCase):
    """Test cases for the _RewritingLoader base class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.fullname = "test.module"
        self.path = "/path/to/test/module.py"
        self.replacer = MockReplacer()
        self.loader = _RewritingLoader(self.fullname, self.path, self.replacer)
    
    def test_init(self):
        """Test _RewritingLoader initialization."""
        self.assertEqual(self.loader.fullname, self.fullname)
        self.assertEqual(self.loader.path, self.path)
        self.assertEqual(self.loader.replacer, self.replacer)
    
    def test_get_filename(self):
        """Test get_filename method."""
        result = self.loader.get_filename(self.fullname)
        self.assertEqual(result, self.path)
        
        # Test with different fullname
        result = self.loader.get_filename("other.module")
        self.assertEqual(result, self.path)
    
    @patch('builtins.open', new_callable=mock_open, read_data=b"test content")
    def test_get_data(self, mock_file):
        """Test get_data method."""
        result = self.loader.get_data(self.path)
        self.assertEqual(result, b"test content")
        mock_file.assert_called_once_with(self.path, "rb")
    
    def test_source_to_code_not_implemented(self):
        """Test that source_to_code raises NotImplementedError in base class."""
        with self.assertRaises(NotImplementedError):
            self.loader.source_to_code(b"test", self.path)


class TestRewritingLoader(unittest.TestCase):
    """Test cases for the RewritingLoader class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.fullname = "test.module"
        self.path = "/path/to/test/module.py"
        self.replacer = MockReplacer()
        self.loader = RewritingLoader(self.fullname, self.path, self.replacer)
    
    def test_init_inheritance(self):
        """Test RewritingLoader inherits from _RewritingLoader."""
        self.assertIsInstance(self.loader, _RewritingLoader)
        self.assertEqual(self.loader.fullname, self.fullname)
        self.assertEqual(self.loader.path, self.path)
        self.assertEqual(self.loader.replacer, self.replacer)
    
    def test_source_to_code_basic(self):
        """Test source_to_code with simple Python code."""
        source_code = b"""
def hello():
    return "Hello, World!"

x = 42
"""
        
        result = self.loader.source_to_code(source_code, self.path)
        
        # Check that result is a code object
        self.assertIsInstance(result, type(compile("", "", "exec")))
        
        # Execute the code to verify it works
        namespace = {}
        exec(result, namespace)
        self.assertIn('hello', namespace)
        self.assertIn('x', namespace)
        self.assertEqual(namespace['x'], 42)
        self.assertEqual(namespace['hello'](), "Hello, World!")
    
    def test_source_to_code_with_renaming(self):
        """Test source_to_code applies renaming transformation."""
        source_code = b"""
def test_function():
    return "original"

def other_function():
    return "unchanged"
"""
        
        result = self.loader.source_to_code(source_code, self.path)
        
        # Execute the code
        namespace = {}
        exec(result, namespace)
        
        # Check that renaming was applied
        self.assertIn('replaced_test_function', namespace)
        self.assertIn('other_function', namespace)
        self.assertEqual(namespace['replaced_test_function'](), "original")
        self.assertEqual(namespace['other_function'](), "unchanged")
    
    @patch('ast.parse')
    @patch('ast.fix_missing_locations')
    def test_source_to_code_ast_operations(self, mock_fix_locations, mock_parse):
        """Test that source_to_code performs correct AST operations."""
        # Mock AST trees
        mock_parsed_tree = MagicMock()
        mock_visited_tree = MagicMock()
        mock_parse.return_value = mock_parsed_tree
        
        # Mock replacer instance (not a class)
        mock_replacer = MagicMock()
        mock_replacer.visit.return_value = mock_visited_tree
        
        # Pass the replacer instance directly
        loader = RewritingLoader(self.fullname, self.path, mock_replacer)
        
        with patch('builtins.compile') as mock_compile:
            mock_compile.return_value = "compiled_code"
            
            result = loader.source_to_code(b"test code", self.path)
            
            # Verify AST operations were called correctly
            mock_parse.assert_called_once_with(b"test code", filename=self.path)
            mock_replacer.visit.assert_called_once_with(mock_parsed_tree)
            mock_fix_locations.assert_called_once_with(mock_visited_tree)
            mock_compile.assert_called_once_with(
                mock_visited_tree, self.path, "exec", dont_inherit=True, optimize=-1
            )
            
            self.assertEqual(result, "compiled_code")
    
    def test_source_to_code_syntax_error(self):
        """Test source_to_code handles syntax errors appropriately."""
        invalid_source = b"def invalid_syntax(:"
        
        with self.assertRaises(SyntaxError):
            self.loader.source_to_code(invalid_source, self.path)
    
    def test_source_to_code_empty_source(self):
        """Test source_to_code with empty source."""
        empty_source = b""
        
        result = self.loader.source_to_code(empty_source, self.path)
        
        # Should still produce a valid code object
        self.assertIsInstance(result, type(compile("", "", "exec")))
        
        # Execute to verify it works
        namespace = {}
        exec(result, namespace)
        # Should have no additional names besides builtins
        expected_builtins = {'__builtins__'}
        self.assertEqual(set(namespace.keys()), expected_builtins)


if __name__ == '__main__':
    unittest.main()
