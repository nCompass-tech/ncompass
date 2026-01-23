# Copyright 2025 nCompass Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Integration tests for the nsys-rep to Chrome trace converter.

This test suite validates the conversion pipeline by:
1. Discovering all .nsys-rep files in trace_converter/test_files/
2. Converting each using both Rust and Python backends via Docker
3. Comparing converted files against golden references
4. Validating kernel event counts using Perfetto's trace processor

The tests use Docker containers to ensure a consistent environment
with all required dependencies (nsys CLI, etc.).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from tests.integration.utils.compare import compare_json_files
from tests.integration.utils.trace_validation import count_events_by_category_json


# ---------------------------------------------------------------------------
# Test file discovery for parametrization
# ---------------------------------------------------------------------------

def _get_repo_root() -> Path:
    """Get repo root for parametrization (runs at collection time)."""
    return Path(__file__).parent.parent.parent.parent


def _get_test_files() -> list[str]:
    """Discover test file stems at collection time."""
    test_files_dir = _get_repo_root() / "examples" / "trace_converter" / "test_files"
    if not test_files_dir.exists():
        return []
    return [f.stem for f in sorted(test_files_dir.glob("*.nsys-rep"))]


# Get test file stems for parametrization
TEST_FILE_STEMS = _get_test_files()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def example_dir() -> Path:
    """Return the trace_converter example directory."""
    return _get_repo_root() / "examples" / "trace_converter"


@pytest.fixture(scope="module")
def test_files_dir(example_dir: Path) -> Path:
    """Return the test_files directory containing .nsys-rep files."""
    return example_dir / "test_files"


@pytest.fixture(scope="module")
def golden_dir() -> Path:
    """Return the golden_references directory."""
    return Path(__file__).parent / "golden_references"


@pytest.fixture(scope="module")
def output_dir(example_dir: Path) -> Path:
    """Create and return a temporary output directory for test results."""
    out_dir = example_dir / ".pytest_output"
    out_dir.mkdir(exist_ok=True)
    yield out_dir
    # Cleanup is optional - leave files for debugging
    import shutil
    # shutil.rmtree(out_dir, ignore_errors=True)


@pytest.fixture(scope="module")
def build_docker_container(example_dir: Path):
    """Build Docker container and install ncompass once before tests, tear down after."""
    # Step 1: Build the docker image
    print("\nBuilding test docker image...")
    subprocess.run(
        [sys.executable, "nc_pkg.py", "--build"],
        cwd=example_dir,
        check=True
    )
    
    # Step 2: Start container and install ncompass (--run --no-exec)
    # This starts the container and installs ncompass from the symlinked source
    print("\nStarting container and installing ncompass...")
    subprocess.run(
        [sys.executable, "nc_pkg.py", "--run", "--no-exec"],
        cwd=example_dir,
        check=True
    )
    
    try:
        yield
    finally:
        print("\nTaking down docker container...")
        subprocess.run(
            [sys.executable, "nc_pkg.py", "--down"],
            cwd=example_dir,
            check=True
        )


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def convert_trace_in_docker(
    example_dir: Path,
    input_file: str,
    output_dir: Path,
    use_rust: bool,
) -> Path:
    """
    Convert a .nsys-rep file using ncompass CLI inside Docker.
    
    Args:
        example_dir: Path to trace_converter example directory
        input_file: Filename (stem) of the .nsys-rep file in test_files/
        output_dir: Directory for output files
        use_rust: Whether to use Rust backend (False = --python-fallback)
        
    Returns:
        Path to the converted .json.gz file
        
    Raises:
        RuntimeError: If conversion fails
    """
    # Build the ncompass convert command
    # Use different output names for rust vs python to avoid overwrites
    backend = "rust" if use_rust else "python"
    input_path = f"test_files/{input_file}.nsys-rep"
    output_path = str(output_dir.relative_to(example_dir))
    output_name = f"{input_file}.{backend}"
    
    if use_rust:
        convert_cmd = f"ncompass convert {input_path} --output-dir {output_path} --output {output_name}"
    else:
        convert_cmd = f"ncompass convert {input_path} --output-dir {output_path} --output {output_name} --python-fallback"
    
    # Execute via nc_pkg.py --exec
    result = subprocess.run(
        [sys.executable, "nc_pkg.py", "--exec", convert_cmd],
        cwd=example_dir,
        capture_output=True,
        text=True,
    )
    
    if result.returncode != 0:
        raise RuntimeError(
            f"Conversion failed for {input_file} (rust={use_rust}):\n"
            f"Command: {convert_cmd}\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )
    
    # The output file should be in output_dir with backend suffix
    converted_path = output_dir / f"{output_name}.json.gz"
    
    if not converted_path.exists():
        raise FileNotFoundError(
            f"Expected output file not found: {converted_path}\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )
    
    return converted_path


def get_golden_path(golden_dir: Path, trace_stem: str, use_rust: bool) -> Path:
    """Get the path to the golden reference file."""
    backend = "rust" if use_rust else "python"
    return golden_dir / f"{trace_stem}.{backend}.json.gz"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    len(TEST_FILE_STEMS) == 0,
    reason="No .nsys-rep files found in test_files/"
)
@pytest.mark.integration
class TestConverterRust:
    """Test suite for the Rust converter backend."""
    
    @pytest.mark.parametrize("trace_stem", TEST_FILE_STEMS)
    def test_conversion_matches_golden(
        self,
        example_dir: Path,
        output_dir: Path,
        golden_dir: Path,
        build_docker_container,
        trace_stem: str,
    ):
        """
        Test that Rust conversion output matches golden reference.
        """
        print(f"\n[Rust] Converting: {trace_stem}")
        
        # Convert the trace
        converted_path = convert_trace_in_docker(
            example_dir=example_dir,
            input_file=trace_stem,
            output_dir=output_dir,
            use_rust=True,
        )
        
        # Get golden reference
        golden_path = get_golden_path(golden_dir, trace_stem, use_rust=True)
        assert golden_path.exists(), f"Golden reference not found: {golden_path}"
        
        # Compare against golden reference
        print(f"[Rust] Comparing against golden: {golden_path.name}")
        match, message = compare_json_files(converted_path, golden_path)
        
        assert match, f"[Rust] {trace_stem}: {message}"
        print(f"[Rust] {trace_stem}: ✓ Matches golden reference")
    
    @pytest.mark.parametrize("trace_stem", TEST_FILE_STEMS)
    def test_kernel_count_nonzero(
        self,
        example_dir: Path,
        output_dir: Path,
        build_docker_container,
        trace_stem: str,
    ):
        """Verify there are kernel events in the converted trace."""
        converted_path = output_dir / f"{trace_stem}.rust.json.gz"
        
        # Skip if file doesn't exist (conversion test may have failed)
        if not converted_path.exists():
            pytest.skip(f"Converted file not found: {converted_path}")
        
        kernel_count = count_events_by_category_json(converted_path, "kernel")
        print(f"\n[Rust] {trace_stem}: {kernel_count} kernel events")
        
        assert kernel_count > 0, (
            f"[Rust] {trace_stem}: No kernel events found. "
            "The trace may be empty or conversion may have failed."
        )


@pytest.mark.skipif(
    len(TEST_FILE_STEMS) == 0,
    reason="No .nsys-rep files found in test_files/"
)
@pytest.mark.integration
class TestConverterPython:
    """Test suite for the Python converter backend."""
    
    @pytest.mark.parametrize("trace_stem", TEST_FILE_STEMS)
    def test_conversion_matches_golden(
        self,
        example_dir: Path,
        output_dir: Path,
        golden_dir: Path,
        build_docker_container,
        trace_stem: str,
    ):
        """
        Test that Python conversion output matches golden reference.
        """
        print(f"\n[Python] Converting: {trace_stem}")
        
        # Convert the trace
        converted_path = convert_trace_in_docker(
            example_dir=example_dir,
            input_file=trace_stem,
            output_dir=output_dir,
            use_rust=False,
        )
        
        # Get golden reference
        golden_path = get_golden_path(golden_dir, trace_stem, use_rust=False)
        assert golden_path.exists(), f"Golden reference not found: {golden_path}"
        
        # Compare against golden reference
        print(f"[Python] Comparing against golden: {golden_path.name}")
        match, message = compare_json_files(converted_path, golden_path)
        
        assert match, f"[Python] {trace_stem}: {message}"
        print(f"[Python] {trace_stem}: ✓ Matches golden reference")
    
    @pytest.mark.parametrize("trace_stem", TEST_FILE_STEMS)
    def test_kernel_count_nonzero(
        self,
        example_dir: Path,
        output_dir: Path,
        build_docker_container,
        trace_stem: str,
    ):
        """Verify there are kernel events in the converted trace."""
        converted_path = output_dir / f"{trace_stem}.python.json.gz"
        
        # Skip if file doesn't exist (conversion test may have failed)
        if not converted_path.exists():
            pytest.skip(f"Converted file not found: {converted_path}")
        
        kernel_count = count_events_by_category_json(converted_path, "kernel")
        print(f"\n[Python] {trace_stem}: {kernel_count} kernel events")
        
        assert kernel_count > 0, (
            f"[Python] {trace_stem}: No kernel events found. "
            "The trace may be empty or conversion may have failed."
        )


@pytest.mark.skipif(
    len(TEST_FILE_STEMS) == 0,
    reason="No .nsys-rep files found in test_files/"
)
@pytest.mark.integration
class TestPerfettoValidation:
    """Test suite for validating converted traces with Perfetto.
    
    Note: Perfetto validation runs inside Docker because the perfetto package
    is installed there, not on the host.
    """
    
    @pytest.mark.parametrize("trace_stem", TEST_FILE_STEMS)
    @pytest.mark.parametrize("use_rust", [True, False], ids=["rust", "python"])
    def test_kernel_count_matches_perfetto(
        self,
        example_dir: Path,
        output_dir: Path,
        build_docker_container,
        trace_stem: str,
        use_rust: bool,
    ):
        """
        Validate that kernel event counts match between JSON and Perfetto SQL.
        
        This ensures our converter produces traces that Perfetto interprets
        consistently with the raw JSON.
        """
        backend_name = "rust" if use_rust else "python"
        backend_label = "Rust" if use_rust else "Python"
        converted_path = output_dir / f"{trace_stem}.{backend_name}.json.gz"
        
        # Skip if file doesn't exist (conversion test may have failed)
        if not converted_path.exists():
            pytest.skip(f"Converted file not found: {converted_path}")
        
        # Count JSON events on host (just parsing JSON, no special deps needed)
        json_count = count_events_by_category_json(converted_path, "kernel")
        
        # Run Perfetto validation inside Docker where perfetto is installed
        # The trace file is accessible via the mounted volume
        trace_path_in_container = converted_path  # Same path due to volume mount
        perfetto_cmd = (
            f"python -c \""
            f"from perfetto.trace_processor import TraceProcessor; "
            f"tp = TraceProcessor(trace='{trace_path_in_container}'); "
            f"result = list(tp.query(\\\"SELECT count(*) as cnt FROM slice WHERE category='kernel'\\\")); "
            f"print(result[0].cnt if result else 0)"
            f"\""
        )
        
        result = subprocess.run(
            [sys.executable, "nc_pkg.py", "--exec", perfetto_cmd],
            cwd=example_dir,
            capture_output=True,
            text=True,
        )
        
        if result.returncode != 0:
            pytest.fail(
                f"[{backend_label}] {trace_stem}: Perfetto validation failed\n"
                f"stdout: {result.stdout}\n"
                f"stderr: {result.stderr}"
            )
        
        # Parse the perfetto count from output (first line contains the count)
        try:
            # The actual output from the python command is on the first line
            # nc_pkg.py prints additional messages after
            first_line = result.stdout.strip().split('\n')[0]
            perfetto_count = int(first_line)
        except (ValueError, IndexError) as e:
            pytest.fail(
                f"[{backend_label}] {trace_stem}: Failed to parse Perfetto output\n"
                f"stdout: {result.stdout}\n"
                f"stderr: {result.stderr}\n"
                f"error: {e}"
            )
        
        print(f"\n[{backend_label}] {trace_stem}:")
        print(f"  JSON count: {json_count}")
        print(f"  Perfetto count: {perfetto_count}")
        
        match = json_count == perfetto_count
        if match:
            print(f"  ✓ Counts match: {json_count}")
        else:
            print(f"  ✗ Mismatch: JSON={json_count}, Perfetto={perfetto_count}")
        
        assert match, (
            f"[{backend_label}] {trace_stem}: Kernel count mismatch - "
            f"JSON={json_count}, Perfetto={perfetto_count}"
        )


@pytest.mark.integration
class TestOverlapDetection:
    """Test suite for overlap detection in trace processing."""

    def test_overlapping_kernel_moved_to_overflow_track(self, golden_dir: Path):
        """
        Verify that partially overlapping kernel events are moved to overflow tracks.

        This tests the overlap detection logic in process_chrome_trace_file().
        The golden reference file contains these events on tid=7:
          1. ts=1656230196442.499, dur=2.463 → ends at 444.962
          2. ts=1656230196443.714, dur=8.065 → ends at 451.779 (overlaps with #1)

        Event #2 starts BEFORE event #1 ends but ends AFTER, so it's a partial
        overlap and should be moved to the overflow track "↳ 7".
        """
        from ncompass.trace.converters.utils import process_chrome_trace_file
        import gzip
        import orjson
        import tempfile

        input_path = golden_dir / "5reqs-50in-5out-0.12.0-1759.json"
        assert input_path.exists(), f"Golden reference not found: {input_path}"

        with tempfile.NamedTemporaryFile(suffix=".json.gz", delete=False) as tmp:
            output_path = tmp.name

        try:
            process_chrome_trace_file(str(input_path), output_path)

            with gzip.open(output_path, "rb") as f:
                trace_data = orjson.loads(f.read())

            events = trace_data.get("traceEvents", [])

            target_ts = 1656230196443.714
            target_events = [
                e for e in events
                if e.get("ph") == "X"
                and e.get("cat") == "kernel"
                and abs(e.get("ts", 0) - target_ts) < 0.001
            ]

            assert len(target_events) == 1, (
                f"Expected exactly 1 event at ts={target_ts}, found {len(target_events)}"
            )

            event = target_events[0]
            tid = event.get("tid")

            # Overflow TID for 7 is 107 (original + 100)
            expected_tid = 107
            assert tid == expected_tid, (
                f"Event at ts={target_ts} should be on overflow track {expected_tid}, "
                f"but found tid={tid!r}. This indicates the overlap detection "
                f"failed to move the partially overlapping event."
            )

            print(f"\n✓ Event at ts={target_ts} correctly moved to overflow track '{tid}'")

        finally:
            import os
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_overlap_detection_unit(self):
        """
        Unit test for _process_event_for_overlap with minimal example.

        Tests the exact scenario from the bug report:
          Event 1: ts=442.499, dur=2.463 → ends at 444.962
          Event 2: ts=443.714, dur=8.065 → ends at 451.779 (partial overlap)

        Event 2 should be moved to overflow track.
        """
        from ncompass.trace.converters.utils import _process_event_for_overlap, OverflowState

        state = OverflowState()

        event1 = {
            "ph": "X", "cat": "kernel", "name": "kernel1",
            "pid": 0, "tid": 7,
            "ts": 1656230196442.499, "dur": 2.463,
        }
        event2 = {
            "ph": "X", "cat": "kernel", "name": "kernel2",
            "pid": 0, "tid": 7,
            "ts": 1656230196443.714, "dur": 8.065,
        }

        result1 = _process_event_for_overlap(event1, state)
        print(f"\nAfter event1: max_end = {state.max_end}")
        print(f"  Event1 tid: {result1.get('tid')}")

        result2 = _process_event_for_overlap(event2, state)
        print(f"After event2: max_end = {state.max_end}")
        print(f"  Event2 tid: {result2.get('tid')}")

        event1_end = event1["ts"] + event1["dur"]
        event2_end = event2["ts"] + event2["dur"]
        print(f"\nEvent1: ts={event1['ts']}, end={event1_end}")
        print(f"Event2: ts={event2['ts']}, end={event2_end}")
        print(f"Event2 starts before Event1 ends: {event2['ts']} < {event1_end} = {event2['ts'] < event1_end}")
        print(f"Event2 ends after Event1 ends: {event2_end} > {event1_end} = {event2_end > event1_end}")

        assert result1.get("tid") == 7, "Event1 should stay on original track"
        # Overflow TID for 7 is 107
        assert result2.get("tid") == 107, (
            f"Event2 should be moved to overflow track 107, "
            f"but found tid={result2.get('tid')!r}"
        )

    def test_non_overlapping_events_unit(self):
        """
        Negative test: Events that don't overlap should stay on original track.
        """
        from ncompass.trace.converters.utils import _process_event_for_overlap, OverflowState
        state = OverflowState()

        # Event 2 starts exactly when Event 1 ends
        event1 = {"ph": "X", "cat": "kernel", "pid": 0, "tid": 7, "ts": 100.0, "dur": 50.0}
        event2 = {"ph": "X", "cat": "kernel", "pid": 0, "tid": 7, "ts": 150.0, "dur": 50.0}

        r1 = _process_event_for_overlap(event1, state)
        r2 = _process_event_for_overlap(event2, state)

        assert r1.get("tid") == 7
        assert r2.get("tid") == 7

        # Event 3 starts well after Event 2 ends
        event3 = {"ph": "X", "cat": "kernel", "pid": 0, "tid": 7, "ts": 300.0, "dur": 50.0}
        r3 = _process_event_for_overlap(event3, state)
        assert r3.get("tid") == 7

    def test_nested_events_unit(self):
        """
        Negative test: Events completely inside another should stay on original track.
        """
        from ncompass.trace.converters.utils import _process_event_for_overlap, OverflowState
        state = OverflowState()

        event1 = {"ph": "X", "cat": "kernel", "pid": 0, "tid": 7, "ts": 100.0, "dur": 100.0}
        event2 = {"ph": "X", "cat": "kernel", "pid": 0, "tid": 7, "ts": 120.0, "dur": 50.0}

        r1 = _process_event_for_overlap(event1, state)
        r2 = _process_event_for_overlap(event2, state)

        assert r1.get("tid") == 7
        assert r2.get("tid") == 7
        assert state.max_end[(0, 7)] == 200.0

    def test_different_tracks_unit(self):
        """
        Negative test: Events on different tracks should not cause overflows for each other.
        """
        from ncompass.trace.converters.utils import _process_event_for_overlap, OverflowState
        state = OverflowState()

        event1 = {"ph": "X", "cat": "kernel", "pid": 0, "tid": 7, "ts": 100.0, "dur": 100.0}
        event2 = {"ph": "X", "cat": "kernel", "pid": 0, "tid": 8, "ts": 150.0, "dur": 100.0}

        r1 = _process_event_for_overlap(event1, state)
        r2 = _process_event_for_overlap(event2, state)

        assert r1.get("tid") == 7
        assert r2.get("tid") == 8

    def test_ignored_events_unit(self):
        """
        Negative test: Non-X phase or user_annotation category should be ignored.
        """
        from ncompass.trace.converters.utils import _process_event_for_overlap, OverflowState
        state = OverflowState()

        # Initial event to set max_end
        _process_event_for_overlap({"ph": "X", "cat": "kernel", "pid": 0, "tid": 7, "ts": 100.0, "dur": 100.0}, state)

        # Event with ph='B' (Begin) - should be ignored even if overlapping
        event_b = {"ph": "B", "cat": "kernel", "pid": 0, "tid": 7, "ts": 150.0}
        r_b = _process_event_for_overlap(event_b, state)
        assert r_b.get("tid") == 7

        # Event with user_annotation in category - should be ignored
        event_ua = {"ph": "X", "cat": "user_annotation", "pid": 0, "tid": 7, "ts": 150.0, "dur": 100.0}
        r_ua = _process_event_for_overlap(event_ua, state)
        assert r_ua.get("tid") == 7

    def test_string_tid_unit(self):
        """
        Test overlap detection with string TIDs.
        """
        from ncompass.trace.converters.utils import _process_event_for_overlap, OverflowState
        state = OverflowState()

        event1 = {"ph": "X", "cat": "kernel", "pid": 0, "tid": "stream_1", "ts": 100.0, "dur": 100.0}
        event2 = {"ph": "X", "cat": "kernel", "pid": 0, "tid": "stream_1", "ts": 150.0, "dur": 100.0}

        r1 = _process_event_for_overlap(event1, state)
        r2 = _process_event_for_overlap(event2, state)

        assert r1.get("tid") == "stream_1"
        assert r2.get("tid") == "↳ stream_1"


@pytest.mark.skipif(
    len(TEST_FILE_STEMS) == 0,
    reason="No .nsys-rep files found in test_files/"
)
@pytest.mark.integration
class TestBackendConsistency:
    """Test suite for verifying Rust and Python backends produce consistent results."""
    
    @pytest.mark.parametrize("trace_stem", TEST_FILE_STEMS)
    def test_rust_python_kernel_count_match(
        self,
        output_dir: Path,
        golden_dir: Path,
        build_docker_container,
        trace_stem: str,
    ):
        """
        Verify Rust and Python backends produce the same kernel count.
        
        This catches any discrepancies between the two converter implementations.
        Uses converted files or golden references to ensure consistency.
        """
        rust_path = output_dir / f"{trace_stem}.rust.json.gz"
        python_path = output_dir / f"{trace_stem}.python.json.gz"
        
        # Try converted files first, fall back to golden references
        if rust_path.exists() and python_path.exists():
            rust_count = count_events_by_category_json(rust_path, "kernel")
            python_count = count_events_by_category_json(python_path, "kernel")
        else:
            # Use golden references
            rust_golden = golden_dir / f"{trace_stem}.rust.json.gz"
            python_golden = golden_dir / f"{trace_stem}.python.json.gz"
            
            if not rust_golden.exists() or not python_golden.exists():
                pytest.skip("Golden references not found for both backends")
            
            rust_count = count_events_by_category_json(rust_golden, "kernel")
            python_count = count_events_by_category_json(python_golden, "kernel")
        
        print(f"\n{trace_stem}: Rust={rust_count}, Python={python_count}")
        
        assert rust_count == python_count, (
            f"{trace_stem}: Backend kernel count mismatch - "
            f"Rust={rust_count}, Python={python_count}"
        )
    
    @pytest.mark.skip(
        reason=("Known issue - args: {} key that isn't really used "
                "is different between python and rust")
    )
    @pytest.mark.parametrize("trace_stem", TEST_FILE_STEMS)
    def test_rust_python_trace_diff(
        self,
        output_dir: Path,
        golden_dir: Path,
        build_docker_container,
        trace_stem: str,
    ):
        """
        Verify Rust and Python backends produce structurally identical traces.
        
        This performs a full diff comparison (with normalization of timestamps,
        durations, and IDs) to ensure both backends produce equivalent output.
        """
        rust_path = output_dir / f"{trace_stem}.rust.json.gz"
        python_path = output_dir / f"{trace_stem}.python.json.gz"
        
        # Try converted files first, fall back to golden references
        if not rust_path.exists() or not python_path.exists():
            rust_path = golden_dir / f"{trace_stem}.rust.json.gz"
            python_path = golden_dir / f"{trace_stem}.python.json.gz"
            
            if not rust_path.exists() or not python_path.exists():
                pytest.skip("Files not found for both backends")
        
        print(f"\n{trace_stem}: Comparing Rust vs Python trace content...")
        
        # Use the compare_json_files utility which normalizes timestamps/IDs
        # and sorts events for deterministic comparison
        match, message = compare_json_files(rust_path, python_path)
        
        if match:
            print(f"  ✓ Traces are structurally identical")
        else:
            print(f"  ✗ Traces differ")
        
        assert match, (
            f"{trace_stem}: Backend trace diff detected - "
            f"Rust and Python outputs are not equivalent.\n{message}"
        )

