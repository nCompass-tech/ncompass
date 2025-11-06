"""
Tests for ncompass.trace.core.rewrite module.
"""

import unittest
import sys
from unittest.mock import patch

from ncompass.trace.core.rewrite import enable_rewrites
from ncompass.trace.core.finder import RewritingFinder


class TestEnableRewrites(unittest.TestCase):
    """Test cases for the enable_rewrites function."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Store original meta_path to restore after tests
        self.original_meta_path = sys.meta_path.copy()
    
    def tearDown(self):
        """Clean up after tests."""
        # Restore original meta_path
        sys.meta_path[:] = self.original_meta_path
    
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


if __name__ == '__main__':
    unittest.main()
