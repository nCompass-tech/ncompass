#!/usr/bin/env python3
"""
Convert nsys report (.nsys-rep) to Chrome trace JSON format.

This script demonstrates how to use the ncompass SDK to convert nsys profiling reports
into Chrome trace format for visualization in Perfetto or Chrome DevTools.

Uses the unified convert_nsys_report function which handles the full pipeline:
1. Converts .nsys-rep to SQLite using nsys CLI
2. Converts SQLite to Chrome Trace format
3. Writes output as gzip-compressed JSON (.json.gz)
"""

import argparse
import sys
from pathlib import Path

from ncompass.trace.converters import convert_nsys_report, ConversionOptions


def run_conversion(
    nsys_rep_file: Path,
    output_file: Path,
    keep_sqlite: bool = False,
) -> int:
    """Convert nsys report to Chrome trace format.
    
    Args:
        nsys_rep_file: Path to input .nsys-rep file
        output_file: Path to output .json.gz file
        keep_sqlite: If True, keep intermediate SQLite file
        
    Returns:
        0 on success, 1 on failure
    """
    # Check if input file exists
    if not nsys_rep_file.exists():
        print(f"Error: Input file not found: {nsys_rep_file}", file=sys.stderr)
        return 1
    
    print("-" * 80)
    print(f"Converting nsys report to Chrome trace format...")
    print(f"Input: {nsys_rep_file}")
    print(f"Output: {output_file}")
    if keep_sqlite:
        print(f"SQLite file will be preserved")
    
    try:
        # Create conversion options with common activity types
        options = ConversionOptions(
            activity_types=["kernel", "nvtx", "nvtx-kernel", "cuda-api", "osrt", "sched"],
            include_metadata=True
        )
        
        # Use the ncompass library's unified conversion function
        convert_nsys_report(
            nsys_rep_path=str(nsys_rep_file),
            output_path=str(output_file),
            options=options,
            keep_sqlite=keep_sqlite,
        )
        
        print(f"Conversion completed successfully!")
        print(f"Chrome trace file saved as: {output_file}")
        return 0
        
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except RuntimeError as e:
        print(f"Error during conversion: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


def parse_args():
    """Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Convert nsys report (.nsys-rep) to Chrome trace JSON format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert default file (test_trace.nsys-rep)
  python convert_nsys.py
  
  # Convert a specific nsys report file
  python convert_nsys.py --input my_profile.nsys-rep
  
  # Specify custom output file
  python convert_nsys.py --input my_profile.nsys-rep --output my_trace.json.gz
  
  # Keep intermediate SQLite file
  python convert_nsys.py --input my_profile.nsys-rep --keep-sqlite
        """
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="test_files/test_trace.nsys-rep",
        help="Input nsys report file (.nsys-rep). Default: test_files/test_trace.nsys-rep"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output file path (.json.gz). If not specified, uses input filename with .json.gz extension."
    )
    parser.add_argument(
        "--output-dir", "-d",
        type=str,
        default=None,
        help="Output directory for generated files. If not specified, uses the script's directory."
    )
    parser.add_argument(
        "--keep-sqlite", "-k",
        action="store_true",
        help="Keep the intermediate SQLite file after conversion."
    )
    
    return parser.parse_args()


def resolve_file_paths(args, script_dir: Path):
    """Resolve file paths based on arguments.
    
    Args:
        args: Parsed command-line arguments
        script_dir: Directory where the script is located
        
    Returns:
        tuple: (nsys_rep_file, output_file)
    """
    # Resolve input file path (could be absolute or relative)
    input_path = Path(args.input)
    if input_path.is_absolute():
        input_file = input_path
    else:
        input_file = script_dir / args.input
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
        if not output_dir.is_absolute():
            output_dir = script_dir / output_dir
    else:
        output_dir = script_dir
    
    # Determine output file
    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_file = output_dir / output_path
        else:
            output_file = output_path
    else:
        # Use input filename with .json.gz extension
        output_base = input_file.stem
        output_file = output_dir / f"{output_base}.json.gz"
    
    # Validate input file extension
    if not input_file.suffix == ".nsys-rep":
        print(
            f"Warning: Input should be a .nsys-rep file. Got: {input_file}",
            file=sys.stderr
        )
    
    return input_file, output_file


def main():
    args = parse_args()
    
    # Get the directory where this script is located
    script_dir = Path(__file__).parent.absolute()
    
    # Resolve file paths based on arguments
    nsys_rep_file, output_file = resolve_file_paths(args, script_dir)
    
    print("=" * 80)
    print("nsys report to Chrome trace conversion")
    print("=" * 80)
    
    ret = run_conversion(
        nsys_rep_file=nsys_rep_file,
        output_file=output_file,
        keep_sqlite=args.keep_sqlite,
    )
    
    if ret == 0:
        print("-" * 80)
        print("Conversion complete!")
        print(f"Chrome trace file: {output_file}")
    
    return ret


if __name__ == "__main__":
    sys.exit(main())
