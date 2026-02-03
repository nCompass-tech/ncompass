"""Utility functions for computing derived trace file paths.

This module provides consistent path computation for derived trace files
(.json.gz, .sqlite, .bkup) to be stored in a hidden directory alongside
the original trace file.
"""

from pathlib import Path

# Hidden directory name for derived trace files
DERIVED_FILES_DIR = ".nc_trace_cache"


def get_derived_dir(source_path: str) -> Path:
    """Get the hidden directory path for derived files.

    Args:
        source_path: Path to the original trace file

    Returns:
        Path to the hidden directory (e.g., /path/to/.nc_trace_cache/)
    """
    source = Path(source_path)
    parent = source.parent

    # If already inside .nc_trace_cache, don't nest another one
    if parent.name == DERIVED_FILES_DIR:
        return parent

    return parent / DERIVED_FILES_DIR


def get_derived_path(source_path: str, new_suffix: str) -> Path:
    """Compute the path for a derived file in the hidden directory.

    Args:
        source_path: Path to the original trace file
        new_suffix: The new suffix for the derived file (e.g., '.json.gz', '.sqlite')

    Returns:
        Path to the derived file in the hidden directory

    Example:
        >>> get_derived_path('/data/traces/profile.nsys-rep', '.json.gz')
        PosixPath('/data/traces/.nc_trace_cache/profile.json.gz')
    """
    source = Path(source_path)
    derived_dir = get_derived_dir(source_path)

    # Get the stem, handling multi-part extensions like .json.gz
    stem = source.stem
    if stem.endswith('.json'):
        stem = stem[:-5]  # Remove .json if present (for .json.gz files)

    return derived_dir / f"{stem}{new_suffix}"


def ensure_derived_dir(source_path: str) -> Path:
    """Ensure the hidden directory exists and return its path.

    Args:
        source_path: Path to the original trace file

    Returns:
        Path to the created hidden directory
    """
    derived_dir = get_derived_dir(source_path)
    derived_dir.mkdir(parents=True, exist_ok=True)
    return derived_dir
