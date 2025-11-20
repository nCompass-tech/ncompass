"""
Tests for ncompass package version.
"""

import unittest

import ncompass


class TestVersion(unittest.TestCase):
    """Test cases for ncompass package version."""

    def test_version_is_not_unknown(self):
        """Test that __version__ is not 'unknown'."""
        self.assertIsNotNone(ncompass.__version__)
        self.assertNotEqual(ncompass.__version__, "unknown")
        self.assertIsInstance(ncompass.__version__, str)
        self.assertGreater(len(ncompass.__version__), 0)

