#!/usr/bin/env python3
"""
Profile the nsys (.nsys-rep) to Chrome trace converter.

This script runs the ncompass conversion pipeline under cProfile and writes:
- cumulative_time_top50.txt: top 50 by cumulative time
- total_time_top50.txt: top 50 by total (self) time
"""

from __future__ import annotations

import argparse
import cProfile
import pstats
import sys
from pathlib import Path
import time

THIS_DIR = Path(__file__).resolve().parent

# Because we need to package a binary and this gets built during pip install, before running this
# file, you should first run `uv pip install ../../`
from ncompass.trace.converters import convert_nsys_report, ConversionOptions

def run_conversion(input_rep: Path, output_trace: Path, use_rust: bool) -> None:
    options = ConversionOptions(
        activity_types=["kernel", "nvtx", "nvtx-kernel", "cuda-api", "osrt", "sched"],
        include_metadata=True,
    )
    convert_nsys_report(
        nsys_rep_path=str(input_rep),
        output_path=str(output_trace),
        options=options,
        keep_sqlite=False,
        use_rust=use_rust
    )


def profile_and_dump(
    input_rep: Path,
    output_trace: Path,
    profiler_type: str,
    n_stats_lines: int,
    use_rust: bool,
) -> None:
    if not input_rep.exists():
        raise FileNotFoundError(f"Input file not found: {input_rep}")

    exc: Exception | None = None
    if profiler_type == "cProfile":
        profiler = cProfile.Profile()
        profiler.enable()
    elif profiler_type == "time":
        t1 = time.time()
    try:
        run_conversion(input_rep, output_trace, use_rust)
    except Exception as e:
        exc = e
    finally:
        if profiler_type == "cProfile":
            profiler.disable()
        elif profiler_type == "time":
            t2 = time.time()
            print(f"Time taken: {(t2 - t1):.2f} seconds")

    if profiler_type == "cProfile":
        cumulative_path = THIS_DIR / f"cumulative_time_top{n_stats_lines}.txt"
        total_path = THIS_DIR / f"total_time_top{n_stats_lines}.txt"

        with cumulative_path.open("w", encoding="utf-8") as f:
            stats = pstats.Stats(profiler, stream=f)
            stats.strip_dirs().sort_stats("cumulative").print_stats(n_stats_lines)

        with total_path.open("w", encoding="utf-8") as f:
            stats = pstats.Stats(profiler, stream=f)
            stats.strip_dirs().sort_stats("tottime").print_stats(n_stats_lines)

    if exc is not None:
        raise exc


def main(
    file_name:     str,
    profiler_type: str,
    n_stats_lines: int,
    use_rust:      bool = True,
) -> int:
    input_rep = THIS_DIR / f"{file_name}.nsys-rep"
    output_trace = THIS_DIR / f"{file_name}.json.gz"

    try:
        profile_and_dump(input_rep, output_trace, profiler_type, n_stats_lines, use_rust)
        print(f"Wrote profile stats to {THIS_DIR}")
        return 0
    except Exception as e:
        print(f"Profiling failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile nsys to Chrome trace conversion")
    parser.add_argument(
        "--file", "-f",
        default=".traces/nsys_h200_vllm_qwen30ba3b_TP4_quant",
        help="Trace file name (without extension, relative to script dir)",
    )
    parser.add_argument(
        "--profiler", "-p",
        choices=["time", "cProfile"],
        default="time",
        help="Profiler type to use",
    )
    parser.add_argument(
        "--stats-lines", "-n",
        type=int,
        default=10,
        help="Number of stats lines to output (for cProfile)",
    )
    parser.add_argument(
        "--use-rust",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use Rust backend for conversion (default: True)",
    )
    args = parser.parse_args()

    raise SystemExit(main(args.file, args.profiler, args.stats_lines, args.use_rust))
