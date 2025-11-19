"""Integration tests for nsys_example."""

import sys
import subprocess
import pytest
from pathlib import Path

@pytest.fixture(scope="function")
def example_dir(repo_root: Path) -> Path: 
    return repo_root / "examples" / "nsys_example"

@pytest.fixture(scope="function")
def copy_nsys_example_gold_result(test_dir: Path): 
    src = Path(__file__).parent / "golden_references" / "gold_nsys_example.json"
    cmd = ["cp", f"{src}", f"{test_dir}"]
    print(f"\nCopying gold file for testing: {cmd}")
    subprocess.run(
        cmd,
        check=True
    )

@pytest.fixture(scope="function")
def build_and_kill_nsys_example_docker_container(example_dir: Path):
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
def test_nsys_example(example_dir: Path, 
                     symlink_sdk,
                     build_and_kill_nsys_example_docker_container,
                     copy_nsys_example_gold_result,
                     test_dir: Path):
    print("Running Test ...")
    test_and_check_cmd = f"""
    python convert_nsys.py --input test_files/test_trace.nsys-rep &&
    test -f test_trace.sqlite &&
    mv test_trace.json test_trace.sqlite .pytest/ &&
    diff .pytest/test_trace.json .pytest/gold_nsys_example.json
    """
    subprocess.run(
        [sys.executable, "nc_pkg.py", "--exec", test_and_check_cmd],
        cwd = example_dir,
        check=True
    )

