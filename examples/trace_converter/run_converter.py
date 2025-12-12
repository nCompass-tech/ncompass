#!/usr/bin/env python3
"""
Profile the nsys (.nsys-rep) to Chrome trace converter.

This script runs the ncompass conversion pipeline under cProfile and writes:
- cumulative_time_top50.txt: top 50 by cumulative time
- total_time_top50.txt: top 50 by total (self) time
"""

from __future__ import annotations

import cProfile
import pstats
import sys
from pathlib import Path
import time

THIS_DIR = Path(__file__).resolve().parent
# Allow running from a repo checkout without installing the package.
REPO_ROOT = THIS_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ncompass.trace.converters import convert_nsys_report, ConversionOptions

# _FILE_NAME = ".traces/python_profile_20251210_081737"
_FILE_NAME = ".traces/nsys_h200_vllm_qwen30ba3b_TP1_quant"

INPUT_REP = THIS_DIR / f"{_FILE_NAME}.nsys-rep"
OUTPUT_TRACE = THIS_DIR / f"{_FILE_NAME}.json.gz"

N_STATS_LINES = 10

def run_conversion() -> None:
    options = ConversionOptions(
        activity_types=["kernel", "nvtx", "nvtx-kernel", "cuda-api", "osrt", "sched"],
        include_metadata=True,
    )
    convert_nsys_report(
        nsys_rep_path=str(INPUT_REP),
        output_path=str(OUTPUT_TRACE),
        options=options,
        keep_sqlite=False,
        use_rust=True
    )


def profile_and_dump(profiler_type: str) -> None:
    if not INPUT_REP.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_REP}")

    exc: Exception | None = None
    if profiler_type == "cProfile":
        profiler = cProfile.Profile()
        profiler.enable()
    elif profiler_type == "time":
        t1 = time.time()
    try:
        run_conversion()
    except Exception as e:
        exc = e
    finally:
        if profiler_type == "cProfile":
            profiler.disable()
        elif profiler_type == "time":
            t2 = time.time()
            print(f"Time taken: {(t2 - t1):.2f} seconds")

    if profiler_type == "cProfile":
        cumulative_path = THIS_DIR / f"cumulative_time_top{N_STATS_LINES}.txt"
        total_path = THIS_DIR / f"total_time_top{N_STATS_LINES}.txt"

        with cumulative_path.open("w", encoding="utf-8") as f:
            stats = pstats.Stats(profiler, stream=f)
            stats.strip_dirs().sort_stats("cumulative").print_stats(N_STATS_LINES)

        with total_path.open("w", encoding="utf-8") as f:
            stats = pstats.Stats(profiler, stream=f)
            stats.strip_dirs().sort_stats("tottime").print_stats(N_STATS_LINES)

    if exc is not None:
        raise exc


def main(profiler_type: str) -> int:
    try:
        profile_and_dump(profiler_type)
        print(f"Wrote profile stats to {THIS_DIR}")
        return 0
    except Exception as e:
        print(f"Profiling failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    profiler_type = "time"
    raise SystemExit(main(profiler_type))
