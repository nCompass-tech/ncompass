"""Integration tests for basic_example."""

import sys
import subprocess
import shutil
import pytest
from   pathlib import Path
from   typing import Generator

from tests.integration.utils import (build_docker_image, 
                                     run_docker_command,
                                     create_symlink,
                                     remove_symlink)

@pytest.fixture(scope="function")
def example_dir(repo_root: Path) -> Path: 
    return repo_root / "examples" / "basic_example"

@pytest.fixture(scope="function")
def copy_basic_example_gold_result(test_dir: Path): 
    src = Path(__file__).parent / "golden_references" / "gold.pt.trace.json"
    cmd = ["cp", f"{src}", f"{test_dir}"]
    print(f"\nCopying gold file for testing: {cmd}")
    subprocess.run(
        cmd,
        check=True
    )

@pytest.fixture(scope="function")
def build_and_kill_basic_example_docker_container(example_dir: Path):
    print("Building test docker image...")
    subprocess.run(
        [sys.executable, "nc_pkg.py", "--build"],
        cwd = example_dir,
        check=True
    )
    try:
        yield
    finally:
        print("Taking down docker container...")
        subprocess.run(
            [sys.executable, "nc_pkg.py", "--down"],
            cwd = example_dir,
            check=True
        )

@pytest.mark.integration
def test_basic_example_link_only(example_dir: Path, 
                                 symlink_sdk,
                                 build_and_kill_basic_example_docker_container,
                                 copy_basic_example_gold_result,
                                 test_dir: Path):
    print("Running Test ...")
    test_and_check_cmd = f"""
    python main.py --link-only test_files/test_trace.pt.trace.json &&
    mv test_files/test_trace.linked.pt.trace.json .pytest/ &&
    diff .pytest/test_trace.linked.pt.trace.json .pytest/gold.pt.trace.json
    """
    subprocess.run(
        [sys.executable, "nc_pkg.py", "--exec", test_and_check_cmd],
        cwd = example_dir,
        check=True
    )
