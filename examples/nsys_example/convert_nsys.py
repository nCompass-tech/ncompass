#!/usr/bin/env python3
"""
Convert nsys report (.nsys-rep) to SQLite and then to Chrome trace JSON format.

This script demonstrates how to use the ncompass SDK to convert nsys profiling reports
into Chrome trace format for visualization in Perfetto or Chrome DevTools.
"""

import argparse
import subprocess
import sys
from pathlib import Path

from ncompass.trace.converters import convert_file, ConversionOptions


def run_sqlite_step(nsys_rep_file: Path, sqlite_file: Path) -> int:
    """Step 1: Convert nsys-rep to SQLite."""
    # Check if input file exists
    if not nsys_rep_file.exists():
        print(f"Error: Input file not found: {nsys_rep_file}", file=sys.stderr)
        return 1
    
    print("-" * 80)
    print(f"Converting nsys report to SQLite...")
    print(f"Input: {nsys_rep_file}")
    print(f"Output: {sqlite_file}")
    
    try:
        export_command = (
            f"nsys export --type sqlite --include-json true "
            f"--force-overwrite true -o {sqlite_file} {nsys_rep_file}"
        )
        result = subprocess.run(export_command, shell=True, check=True)
        print(f"SQLite conversion completed successfully!")
        print(f"SQLite file saved as: {sqlite_file}")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"Error converting to SQLite: {e}", file=sys.stderr)
        return e.returncode
    except FileNotFoundError:
        print(
            "Error: 'nsys' command not found. Please ensure nsys CLI is installed "
            "and available in your PATH.",
            file=sys.stderr
        )
        return 1


def run_chrome_step(sqlite_file: Path, chrome_trace_file: Path) -> int:
    """Step 2: Convert SQLite to Chrome trace."""
    # Check if input file exists
    if not sqlite_file.exists():
        print(f"Error: Input file not found: {sqlite_file}", file=sys.stderr)
        return 1
    
    print("-" * 80)
    print(f"Converting SQLite to Chrome trace format...")
    print(f"Input: {sqlite_file}")
    print(f"Output: {chrome_trace_file}")
    
    try:
        # Create conversion options with common activity types
        options = ConversionOptions(
            activity_types=["kernel", "nvtx", "nvtx-kernel", "cuda-api", "osrt", "sched"],
            include_metadata=True
        )
        
        convert_file(str(sqlite_file), str(chrome_trace_file), options)
        print(f"Chrome trace conversion completed successfully!")
        print(f"Chrome trace file saved as: {chrome_trace_file}")
        return 0
    except Exception as e:
        print(f"Error converting to Chrome trace: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


def parse_args():
    """Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Convert nsys report (.nsys-rep) to SQLite and Chrome trace JSON format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert default file (vllm_trace_profile.nsys-rep)
  python convert_nsys.py
  
  # Convert a specific nsys report file
  python convert_nsys.py --input my_profile.nsys-rep
  
  # Specify custom output base name
  python convert_nsys.py --input my_profile.nsys-rep --output my_trace
  
  # Run only SQLite conversion step
  python convert_nsys.py --step sqlite
  
  # Run only Chrome trace conversion step (requires existing SQLite file)
  python convert_nsys.py --step chrome --input vllm_trace_profile.sqlite
        """
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="vllm_trace_profile.nsys-rep",
        help="Input nsys report file (.nsys-rep) or SQLite file (.sqlite) if --step chrome. "
             "Default: vllm_trace_profile.nsys-rep"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Base name for output files (without extensions). "
             "If not specified, uses input filename without extension."
    )
    parser.add_argument(
        "--step", "-s",
        choices=["sqlite", "chrome", "all"],
        default="all",
        help="Which step to run: 'sqlite' (nsys-rep -> sqlite), "
             "'chrome' (sqlite -> json), or 'all' (run all steps). Default: all"
    )
    
    return parser.parse_args()


def resolve_file_paths(args, script_dir: Path):
    """Resolve file paths based on arguments and step.
    
    Args:
        args: Parsed command-line arguments
        script_dir: Directory where the script is located
        
    Returns:
        tuple: (nsys_rep_file, sqlite_file, chrome_trace_file)
               Some values may be None depending on the step
    """
    # Resolve input file path
    input_file = script_dir / args.input
    
    # Determine output base name
    if args.output:
        output_base = args.output
    else:
        # Use input filename without extension
        output_base = input_file.stem
    
    # Determine file paths based on step
    if args.step == "chrome":
        # For chrome step, input should be SQLite file
        if not input_file.suffix == ".sqlite":
            print(
                "Warning: For --step chrome, input should be a .sqlite file. "
                f"Got: {input_file}",
                file=sys.stderr
            )
        sqlite_file = input_file
        chrome_trace_file = script_dir / f"{output_base}.json"
        return (None, sqlite_file, chrome_trace_file)
    else:
        # For sqlite or all steps, input should be nsys-rep file
        if not input_file.suffix == ".nsys-rep":
            print(
                "Warning: Input should be a .nsys-rep file. "
                f"Got: {input_file}",
                file=sys.stderr
            )
        nsys_rep_file = input_file
        sqlite_file = script_dir / f"{output_base}.sqlite"
        chrome_trace_file = script_dir / f"{output_base}.json"
        return (nsys_rep_file, sqlite_file, chrome_trace_file)


def main():
    args = parse_args()
    
    # Get the directory where this script is located
    script_dir = Path(__file__).parent.absolute()
    
    # Resolve file paths based on arguments
    nsys_rep_file, sqlite_file, chrome_trace_file = resolve_file_paths(args, script_dir)
    
    # Run the requested step(s)
    if args.step == "all":
        print("=" * 80)
        print("Running all steps: sqlite -> chrome")
        print("=" * 80)
    
    if args.step == "all" or args.step == "sqlite":
        ret = run_sqlite_step(nsys_rep_file, sqlite_file)
        if ret != 0:
            return ret
    
    if args.step == "all" or args.step == "chrome":
        ret = run_chrome_step(sqlite_file, chrome_trace_file)
        if ret != 0:
            return ret
    
    if args.step == "all":
        print("-" * 80)
        print("All conversions completed!")
        print(f"SQLite file: {sqlite_file}")
        print(f"Chrome trace file: {chrome_trace_file}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

