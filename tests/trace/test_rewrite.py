"""
Tests for ncompass.trace.core.rewrite module.
"""

import unittest
import sys
from unittest.mock import patch

from ncompass.trace.core.rewrite import enable_rewrites
from ncompass.trace.core.finder import RewritingFinder
from ncompass.trace.core.pydantic import RewriteConfig, ModuleConfig
from ncompass.trace.core.utils import clear_cached_modules, update_module_references


class TestEnableRewrites(unittest.TestCase):
    """Test cases for the enable_rewrites function."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Store original meta_path to restore after tests
        self.original_meta_path = sys.meta_path.copy()
        # Store original sys.modules state
        self.original_modules = sys.modules.copy()
    
    def tearDown(self):
        """Clean up after tests."""
        # Restore original meta_path
        sys.meta_path[:] = self.original_meta_path
        # Clean up any test modules we created
        modules_to_remove = [
            name for name in sys.modules.keys()
            if name not in self.original_modules
        ]
        for name in modules_to_remove:
            if name in sys.modules:
                del sys.modules[name]
    
    def test_enable_rewrites_adds_finder_to_meta_path(self):
        """Test that enable_rewrites adds RewritingFinder to sys.meta_path."""
        # Remove any existing RewritingFinder instances
        sys.meta_path = [f for f in sys.meta_path if not isinstance(f, RewritingFinder)]
        
        # Call enable_rewrites
        enable_rewrites()
        
        # Check that a RewritingFinder was added to meta_path
        rewriting_finders = [f for f in sys.meta_path if isinstance(f, RewritingFinder)]
        self.assertEqual(len(rewriting_finders), 1)
        self.assertIsInstance(sys.meta_path[0], RewritingFinder)
    
    def test_enable_rewrites_does_not_add_duplicate_finder(self):
        """Test that enable_rewrites doesn't add duplicate RewritingFinder instances."""
        # Remove any existing RewritingFinder instances
        sys.meta_path = [f for f in sys.meta_path if not isinstance(f, RewritingFinder)]
        
        # Call enable_rewrites twice
        enable_rewrites()
        initial_count = len([f for f in sys.meta_path if isinstance(f, RewritingFinder)])
        
        enable_rewrites()
        final_count = len([f for f in sys.meta_path if isinstance(f, RewritingFinder)])
        
        # Should still only have one RewritingFinder
        self.assertEqual(initial_count, 1)
        self.assertEqual(final_count, 1)
    
    def test_enable_rewrites_inserts_at_beginning(self):
        """Test that RewritingFinder is inserted at the beginning of meta_path."""
        # Remove any existing RewritingFinder instances
        sys.meta_path = [f for f in sys.meta_path if not isinstance(f, RewritingFinder)]
        original_length = len(sys.meta_path)
        
        # Call enable_rewrites
        enable_rewrites()
        
        # Check that the finder was inserted at index 0
        self.assertIsInstance(sys.meta_path[0], RewritingFinder)
        self.assertEqual(len(sys.meta_path), original_length + 1)
    
    @patch.dict('os.environ', {'USE_AI_PROFILING': 'false'})
    def test_enable_rewrites_with_existing_finder_replaces(self):
        """Test enable_rewrites replaces existing RewritingFinder when not incremental."""
        # Add a RewritingFinder manually
        existing_finder = RewritingFinder()
        sys.meta_path.insert(0, existing_finder)
        original_length = len(sys.meta_path)
        
        # Call enable_rewrites (non-incremental mode)
        enable_rewrites()
        
        # Should still have same total count but with a NEW finder instance
        self.assertEqual(len(sys.meta_path), original_length)
        rewriting_finders = [f for f in sys.meta_path if isinstance(f, RewritingFinder)]
        self.assertEqual(len(rewriting_finders), 1)
        # Should be a different instance (replaced, not reused)
        self.assertIsNot(rewriting_finders[0], existing_finder)


class TestClearCachedModules(unittest.TestCase):
    """Test cases for clear_cached_modules function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.original_modules = sys.modules.copy()
    
    def tearDown(self):
        """Clean up after tests."""
        # Clean up any test modules we created
        modules_to_remove = [
            name for name in sys.modules.keys()
            if name not in self.original_modules
        ]
        for name in modules_to_remove:
            if name in sys.modules:
                del sys.modules[name]
    
    def test_clear_cached_modules_removes_from_sys_modules(self):
        """Test that clear_cached_modules removes modules from sys.modules."""
        # Create a test module
        test_module = type(sys)('test_module')
        sys.modules['test_module'] = test_module
        
        # Clear it
        targets = {'test_module': ModuleConfig()}
        old_modules = clear_cached_modules(targets)
        
        # Module should be removed from sys.modules
        self.assertNotIn('test_module', sys.modules)
        # Old module reference should be returned
        self.assertIn('test_module', old_modules)
        self.assertIs(old_modules['test_module'], test_module)
    
    def test_clear_cached_modules_handles_nonexistent_module(self):
        """Test that clear_cached_modules handles modules not in sys.modules."""
        targets = {'nonexistent_module': ModuleConfig()}
        old_modules = clear_cached_modules(targets)
        
        # Should return empty dict for nonexistent module
        self.assertEqual(old_modules, {})
        self.assertNotIn('nonexistent_module', sys.modules)
    
    def test_clear_cached_modules_clears_submodules(self):
        """Test that clear_cached_modules clears submodules."""
        # Create test modules
        parent_module = type(sys)('parent')
        child_module = type(sys)('parent.child')
        sys.modules['parent'] = parent_module
        sys.modules['parent.child'] = child_module
        
        # Clear parent
        targets = {'parent': ModuleConfig()}
        old_modules = clear_cached_modules(targets)
        
        # Both should be removed
        self.assertNotIn('parent', sys.modules)
        self.assertNotIn('parent.child', sys.modules)
        # Only parent should be in old_modules
        self.assertIn('parent', old_modules)
        self.assertNotIn('parent.child', old_modules)


class TestUpdateModuleReferences(unittest.TestCase):
    """Test cases for update_module_references function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.original_modules = sys.modules.copy()
        # Create a temporary test module
        self.create_test_module()
        # Store module-level references for testing
        self.test_module_refs = {}
    
    def tearDown(self):
        """Clean up after tests."""
        # Clean up test modules
        modules_to_remove = [
            name for name in sys.modules.keys()
            if name not in self.original_modules
        ]
        for name in modules_to_remove:
            if name in sys.modules:
                del sys.modules[name]
        # Clean up module-level references
        for key in list(self.test_module_refs.keys()):
            del self.test_module_refs[key]
    
    def create_test_module(self):
        """Create a test module with a function."""
        test_module_code = '''
def test_function():
    return "original_value"

TEST_CONSTANT = 42
'''
        # Create module in memory
        test_module = type(sys)('test_ref_module')
        exec(test_module_code, test_module.__dict__)
        sys.modules['test_ref_module'] = test_module
        return test_module
    
    def test_update_direct_module_reference(self):
        """Test that direct module references in module dicts are updated."""
        # Get the old module
        old_module = sys.modules['test_ref_module']
        
        # Store reference in a module-level dict (simulating module import)
        self.test_module_refs['test_ref_module'] = old_module
        
        # Create a new version of the module
        new_module = type(sys)('test_ref_module')
        new_module.test_function = lambda: "new_value"
        new_module.TEST_CONSTANT = 100
        sys.modules['test_ref_module'] = new_module
        
        # Update references
        old_modules = {'test_ref_module': old_module}
        update_module_references(old_modules)
        
        # The reference in the dict should now point to the new module
        self.assertIs(self.test_module_refs['test_ref_module'], new_module)
        self.assertEqual(self.test_module_refs['test_ref_module'].TEST_CONSTANT, 100)
    
    def test_update_from_imported_symbol(self):
        """Test that from-imported symbols in module dicts are updated."""
        # Get the old module and function
        old_module = sys.modules['test_ref_module']
        old_function = old_module.test_function
        
        # Store function reference in a module-level dict (simulating from-import)
        self.test_module_refs['test_function'] = old_function
        
        # Create a new version with different function
        new_module = type(sys)('test_ref_module')
        def new_test_function():
            return "new_value"
        new_module.test_function = new_test_function
        new_module.TEST_CONSTANT = 100
        sys.modules['test_ref_module'] = new_module
        
        # Update references
        old_modules = {'test_ref_module': old_module}
        update_module_references(old_modules)
        
        # The function reference in the dict should be updated
        self.assertIs(self.test_module_refs['test_function'], new_test_function)
        self.assertEqual(self.test_module_refs['test_function'](), "new_value")


class TestModuleReloadingIntegration(unittest.TestCase):
    """Integration tests for module clearing and reloading."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.original_meta_path = sys.meta_path.copy()
        self.original_modules = sys.modules.copy()
        # Remove any existing RewritingFinder
        sys.meta_path = [f for f in sys.meta_path if not isinstance(f, RewritingFinder)]
    
    def tearDown(self):
        """Clean up after tests."""
        sys.meta_path[:] = self.original_meta_path
        # Clean up test modules
        modules_to_remove = [
            name for name in sys.modules.keys()
            if name not in self.original_modules
        ]
        for name in modules_to_remove:
            if name in sys.modules:
                del sys.modules[name]
    
    @patch.dict('os.environ', {'USE_AI_PROFILING': 'false'})
    @patch('importlib.import_module')
    def test_enable_rewrites_clears_and_reimports_modules(self, mock_import_module):
        """Test that enable_rewrites clears and re-imports target modules."""
        # Create a test module
        test_module = type(sys)('test_integration_module')
        test_module.value = "original"
        sys.modules['test_integration_module'] = test_module
        
        # Mock the re-import to return a new module AND put it in sys.modules
        new_module = type(sys)('test_integration_module')
        new_module.value = "reloaded"
        
        def mock_import_side_effect(module_name):
            sys.modules[module_name] = new_module
            return new_module
        
        mock_import_module.side_effect = mock_import_side_effect
        
        # Enable rewrites with this module as target
        config = RewriteConfig(
            targets={
                'test_integration_module': ModuleConfig()
            }
        )
        enable_rewrites(config)
        
        # Should have called import_module to re-import
        mock_import_module.assert_called_once_with('test_integration_module')
        # Module should be in sys.modules (the new one)
        self.assertIn('test_integration_module', sys.modules)
        self.assertEqual(sys.modules['test_integration_module'].value, "reloaded")
    
    @patch.dict('os.environ', {'USE_AI_PROFILING': 'false'})
    @patch('importlib.import_module')
    def test_enable_rewrites_updates_pre_imported_module_reference(self, mock_import_module):
        """Test that enable_rewrites updates references to pre-imported modules."""
        # Create and import test module BEFORE enabling rewrites
        old_module = type(sys)('test_preimport_module')
        old_module.func = lambda: "original"
        sys.modules['test_preimport_module'] = old_module
        
        # Store reference in instance variable (accessible to gc.get_referrers)
        self.test_refs = {}
        self.test_refs['test_preimport_module'] = old_module
        self.test_refs['func'] = old_module.func
        
        # Mock the re-import to return a new module AND put it in sys.modules
        new_module = type(sys)('test_preimport_module')
        new_module.func = lambda: "reloaded"
        
        def mock_import_side_effect(module_name):
            sys.modules[module_name] = new_module
            return new_module
        
        mock_import_module.side_effect = mock_import_side_effect
        
        # Enable rewrites
        config = RewriteConfig(
            targets={
                'test_preimport_module': ModuleConfig()
            }
        )
        enable_rewrites(config)
        
        # Should have called import_module
        mock_import_module.assert_called_once_with('test_preimport_module')
        # References should be updated
        self.assertIs(self.test_refs['test_preimport_module'], new_module)
        self.assertEqual(self.test_refs['func'](), "reloaded")


class TestLocalImports(unittest.TestCase):
    """Test cases for local imports (e.g., from model import ... instead of from package.model import ...).
    
    This tests the scenario where:
    - A module is imported locally BEFORE enable_rewrites is called
    - The config uses the fully qualified module name (e.g., 'ncompass.examples.basic_example.model')
    - The local import reference (e.g., 'model' in the importing module's namespace) should be updated
    """
    
    def setUp(self):
        """Set up test fixtures."""
        self.original_meta_path = sys.meta_path.copy()
        self.original_modules = sys.modules.copy()
        # Remove any existing RewritingFinder
        sys.meta_path = [f for f in sys.meta_path if not isinstance(f, RewritingFinder)]
    
    def tearDown(self):
        """Clean up after tests."""
        sys.meta_path[:] = self.original_meta_path
        # Clean up test modules
        modules_to_remove = [
            name for name in sys.modules.keys()
            if name not in self.original_modules
        ]
        for name in modules_to_remove:
            if name in sys.modules:
                del sys.modules[name]
    
    @patch.dict('os.environ', {'USE_AI_PROFILING': 'false'})
    @patch('importlib.import_module')
    def test_reimport_modules_local_import_before_enable_rewrites(self, mock_import_module):
        """Test _reimport_modules when module was imported locally before enable_rewrites.
        
        This is the full integration test matching the tmp.py scenario:
        1. Module imported locally BEFORE enable_rewrites (like tmp.py line 5: 'from model import ...')
        2. Python stores it in sys.modules as 'model', NOT 'ncompass.examples.basic_example.model'
        3. enable_rewrites called with fully qualified name in config
        4. clear_cached_modules doesn't find the module (it's under 'model', not fully qualified name)
        5. old_modules is empty, so update_module_references does nothing
        6. Local import references are NOT updated - THIS IS THE BUG
        
        KEY ISSUE: When a module is imported locally, it's stored under the local name in sys.modules.
        clear_cached_modules needs to find modules by checking alternative names (like the local name).
        """
        fully_qualified_name = 'ncompass.examples.basic_example.model'
        local_name = 'model'
        
        # Step 1: Module imported locally BEFORE enable_rewrites (like tmp.py line 5)
        # When you do 'from model import ...', Python stores it as 'model', NOT the fully qualified name
        old_module = type(sys)(fully_qualified_name)
        old_module.run_model_inference = lambda: "original"
        sys.modules[local_name] = old_module  # ONLY stored under local name
        # sys.modules[fully_qualified_name] does NOT exist
        
        # Step 2: Importing module has local import reference (like tmp.py)
        importing_module = type(sys)('ncompass.examples.basic_example.tmp')
        importing_module.run_model_inference = old_module.run_model_inference
        importing_module.model = old_module
        sys.modules['ncompass.examples.basic_example.tmp'] = importing_module
        
        # Store reference for testing
        self.tmp_refs = {}
        self.tmp_refs['run_model_inference'] = old_module.run_model_inference
        self.tmp_refs['model'] = old_module
        
        # Step 3: enable_rewrites is called (like tmp.py line 13)
        # clear_cached_modules looks for 'ncompass.examples.basic_example.model' but doesn't find it
        # So old_modules is empty {}
        
        # Mock re-import to return new module
        new_module = type(sys)(fully_qualified_name)
        new_module.run_model_inference = lambda: "reloaded"
        
        def mock_import_side_effect(module_name):
            if module_name == fully_qualified_name:
                sys.modules[fully_qualified_name] = new_module
                return new_module
            return sys.modules.get(module_name)
        
        mock_import_module.side_effect = mock_import_side_effect
        
        # Enable rewrites with fully qualified name (as in config)
        config = RewriteConfig(
            targets={
                fully_qualified_name: ModuleConfig()
            }
        )
        enable_rewrites(config)
        
        # Step 4: Local import references are NOT updated because old_modules was empty
        # This test should FAIL with current implementation
        # The problem: clear_cached_modules doesn't find the module because it's stored under 'model',
        # not 'ncompass.examples.basic_example.model'. So old_modules is empty, and update_module_references
        # has nothing to work with.
        self.assertIs(self.tmp_refs['model'], new_module,
                     "Local import reference should be updated after enable_rewrites")
        self.assertEqual(self.tmp_refs['run_model_inference'](), "reloaded",
                        "Function from local import should be updated")
        self.assertIs(importing_module.model, new_module,
                     "Module reference in tmp module should be updated")
        self.assertEqual(importing_module.run_model_inference(), "reloaded",
                        "Function reference in tmp module should be updated")


if __name__ == '__main__':
    unittest.main()
