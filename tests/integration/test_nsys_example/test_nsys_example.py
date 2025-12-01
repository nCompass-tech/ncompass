"""Integration test for nsys_example conversion.

Tests the nsys-rep to Chrome trace conversion pipeline:
1. Run main.py --convert-only on a pre-existing .nsys-rep file
2. Convert to Chrome trace JSON format (.json.gz)
3. Compare output against golden reference
"""

import sys
import subprocess
import pytest
from pathlib import Path

from tests.integration.utils.compare import compare_json_files


@pytest.fixture(scope="function")
def example_dir(repo_root: Path) -> Path:
    """Return the nsys_example directory."""
    return repo_root / "examples" / "nsys_example"


@pytest.fixture(scope="function")
def build_and_kill_nsys_example_docker_container(example_dir: Path):
    """Build Docker container before test, tear down after."""
    print("Building test docker image...")
    subprocess.run(
        [sys.executable, "nc_pkg.py", "--build"],
        cwd=example_dir,
        check=True
    )
    try:
        yield
    finally:
        print("Taking down docker container...")
        subprocess.run(
            [sys.executable, "nc_pkg.py", "--down"],
            cwd=example_dir,
            check=True
        )


@pytest.mark.integration
def test_nsys_convert(
    example_dir: Path,
    symlink_sdk,
    build_and_kill_nsys_example_docker_container,
    test_dir: Path
):
    """
    Test nsys-rep to Chrome trace conversion.
    
    This test verifies:
    1. main.py --convert-only converts .nsys-rep to .json.gz
    2. Output matches golden reference after normalization
    """
    print("Running nsys conversion test (main.py --convert-only)...")
    
    # Run main.py with --convert-only flag inside Docker
    convert_cmd = "python main.py --convert-only --input test_files/test_trace.nsys-rep --output-dir .pytest"
    
    result = subprocess.run(
        [sys.executable, "nc_pkg.py", "--exec", convert_cmd],
        cwd=example_dir,
        check=True,
        capture_output=True,
        text=True
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    
    # Compare generated output against golden reference
    generated_path = test_dir / "test_trace.json.gz"
    gold_path = Path(__file__).parent / "golden_references" / "golden_convert.json.gz"
    
    assert generated_path.exists(), f"Generated file not found: {generated_path}"
    assert gold_path.exists(), f"Gold file not found: {gold_path}"
    
    match, message = compare_json_files(generated_path, gold_path)
    
    if not match:
        pytest.fail(f"Output does not match golden reference:\n{message}")
    
    print("Nsys conversion test passed!")
