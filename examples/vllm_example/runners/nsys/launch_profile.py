#!/usr/bin/env python3
"""
Launcher script that uses nsys_cmd to profile get_vllm_trace.py, generate an nsys report,
convert it to SQLite, and then convert to Chrome trace format.
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path

# Add parent directory to path to import nsys2chrome
# Script is at: vllm_examples/runners/nsys/launch_profile.py
# nsys2chrome is at: vllm_examples/nsys2chrome/
nsys2chrome_path = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(0, str(nsys2chrome_path))
from nsys2chrome import convert_file
from nsys2chrome.models import ConversionOptions

def nsys_cmd(output_name: str, with_range: bool=True):
    # Path to pyfunc_config.json in the same directory as this script
    pyfunc_config_path = Path(__file__).parent.absolute() / "pyfunc_config.json"
    
    command = ""
    command += "nsys profile "
    command += " --trace=" + 'cuda,nvtx,osrt,cudnn,cublas,opengl,cudla'
    command += " --output=" + output_name
    command += " --sample=" + 'process-tree'
    command += " --session-new=" + 'nc0'
    command += " --gpuctxsw=" + str(True).lower()
    command += " --cuda-graph-trace=" + 'node'
    command += " --show-output=" + str(True).lower()
    command += " --stop-on-exit=" + str(True).lower()
    command += " --gpu-metrics-devices=" + 'all'
    command += " --force-overwrite=" + str(True).lower()
    command += " --cuda-memory-usage=" + str(True).lower()
    command += " --trace-fork-before-exec=true"
    # command += " --cudabacktrace=kernel"
    # command += " --python-backtrace=cuda"
    # command += " --pytorch=functions-trace"
    # command += " --python-sampling=true"
    # command += f" --python-functions-trace={pyfunc_config_path}"
    if with_range:
        command += " --capture-range=" + "nvtx"
        command += " --nvtx-capture=" + "nc_start_capture"
        command += " --env-var=NSYS_NVTX_PROFILER_REGISTER_ONLY=" + "0"
        command += " --capture-range-end=" + "repeat"
    
    return command

def run_profile_step(script_dir: Path, target_script: Path, output_name: str) -> int:
    """Step 1: Run nsys profile to generate .nsys-rep file."""
    # Build the nsys command
    nsys_command = nsys_cmd(output_name=output_name, with_range=True)
    
    # Construct the full command: nsys command + python script
    python_executable = sys.executable
    # full_command = f"{nsys_command} {python_executable} -m nvtx {target_script}"
    full_command = f"{nsys_command} {python_executable} {target_script}"
    
    print(f"Running command: {full_command}")
    print(f"Output will be saved as: {output_name}.nsys-rep")
    print("-" * 80)
    
    try:
        result = subprocess.run(full_command, shell=True, check=True, cwd=script_dir)
        print("-" * 80)
        print(f"Profile completed successfully!")
        print(f"Report saved as: {output_name}.nsys-rep")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"Error running nsys profile: {e}", file=sys.stderr)
        return e.returncode

def run_sqlite_step(script_dir: Path, output_name: str) -> int:
    """Step 2: Convert nsys-rep to SQLite."""
    nsys_rep_file = script_dir / f"{output_name}.nsys-rep"
    sqlite_file = script_dir / f"{output_name}.sqlite"
    
    # Check if input file exists
    if not nsys_rep_file.exists():
        print(f"Error: Input file not found: {nsys_rep_file}", file=sys.stderr)
        return 1
    
    print("-" * 80)
    print(f"Converting nsys report to SQLite...")
    print(f"Input: {nsys_rep_file}")
    print(f"Output: {sqlite_file}")
    
    try:
        export_command = f"nsys export --type sqlite --include-json true --force-overwrite true -o {sqlite_file} {nsys_rep_file}"
        result = subprocess.run(export_command, shell=True, check=True, cwd=script_dir)
        print(f"SQLite conversion completed successfully!")
        print(f"SQLite file saved as: {sqlite_file}")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"Error converting to SQLite: {e}", file=sys.stderr)
        return e.returncode

def run_chrome_step(script_dir: Path, output_name: str) -> int:
    """Step 3: Convert SQLite to Chrome trace."""
    sqlite_file = script_dir / f"{output_name}.sqlite"
    chrome_trace_file = script_dir / f"{output_name}.json"
    
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

def main():
    parser = argparse.ArgumentParser(
        description="Profile vLLM with nsys and convert to SQLite and Chrome trace formats.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all steps (default)
  python launch_profile.py
  
  # Run only nsys profile step
  python launch_profile.py --step profile
  
  # Run only SQLite conversion step
  python launch_profile.py --step sqlite
  
  # Run only Chrome trace conversion step
  python launch_profile.py --step chrome
        """
    )
    parser.add_argument(
        "--step", "-s",
        choices=["profile", "sqlite", "chrome", "all"],
        default="all",
        help="Which step to run: 'profile' (nsys profile), 'sqlite' (nsys-rep -> sqlite), "
             "'chrome' (sqlite -> json), or 'all' (run all steps). Default: all"
    )
    parser.add_argument(
        "--output-name", "-o",
        default="vllm_trace_profile",
        help="Base name for output files (without extensions). Default: vllm_trace_profile"
    )
    
    args = parser.parse_args()
    
    # Get the directory where this script is located
    script_dir = Path(__file__).parent.absolute()
    
    # Path to the script to profile
    target_script = script_dir / "get_vllm_trace.py"
    
    output_name = args.output_name
    
    # Run the requested step(s) - cascading if statements
    if args.step == "all":
        print("=" * 80)
        print("Running all steps: profile -> sqlite -> chrome")
        print("=" * 80)
    
    if args.step == "all" or args.step == "profile":
        ret = run_profile_step(script_dir, target_script, output_name)
        if ret != 0:
            return ret
    
    if args.step == "all" or args.step == "sqlite":
        ret = run_sqlite_step(script_dir, output_name)
        if ret != 0:
            return ret
    
    if args.step == "all" or args.step == "chrome":
        ret = run_chrome_step(script_dir, output_name)
        if ret != 0:
            return ret
    
    if args.step == "all":
        print("-" * 80)
        print("All conversions completed!")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

