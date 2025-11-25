"""Pytest fixtures for integration tests."""

import tempfile
import shutil
import subprocess
from pathlib import Path
from typing import Generator

import pytest

from tests.integration.utils import (create_symlink,
                                     remove_symlink)

@pytest.fixture(scope="function")
def repo_root() -> Path:
    """Get the repository root directory."""
    # Assuming conftest.py is in tests/integration/
    path = Path(__file__).parent.parent.parent.absolute()
    return path

@pytest.fixture(scope="function")
def symlink_sdk(repo_root: Path, example_dir: Path):
    """Get the repository root directory."""
    sdk_path = "../../ncompass"
    link   = example_dir / "ncompass"
    print("\nSymlinking SDK ...")
    create_symlink(link, sdk_path)
    try:
        yield
    finally:
        print("\nRemoving Symlinked ...")
        remove_symlink(link)

@pytest.fixture(scope="function")
def test_dir(example_dir: Path) -> Path:
    """Get the repository root directory."""
    test_dir = example_dir / ".pytest"
    print(f"\nCreating test dir {test_dir} for test generated files ...")
    subprocess.run(
        ["mkdir", f"{test_dir}"],
        check=True
    )
    try:
        yield test_dir
    finally: 
        print("\nRemoving Test generated files ...")
        subprocess.run(
            [f"rm", "-rf", f"{test_dir}"],
            check=True
        )
