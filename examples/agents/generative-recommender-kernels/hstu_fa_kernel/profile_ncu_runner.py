#!/usr/bin/env python3
"""
NCU profiling wrapper for the from-scratch HSTU attention kernel.
Python replacement for profile_ncu.sh.

Usage:
    python hstu_fa_kernel/profile_ncu_runner.py
    python hstu_fa_kernel/profile_ncu_runner.py -o my_report
    python hstu_fa_kernel/profile_ncu_runner.py --kernel-regex "FlashAttn"
    python hstu_fa_kernel/profile_ncu_runner.py -- --batch-size 256
"""

import argparse
import os
import subprocess
from pathlib import Path


def get_paths():
    """Return (script_dir, repo_root) as Path objects."""
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    return script_dir, repo_root


def parse_args():
    """Parse wrapper arguments."""
    parser = argparse.ArgumentParser(
        description="NCU profiling wrapper for HSTU attention kernel",
        usage="python profile_ncu_runner.py [-o NAME] [--kernel-regex REGEX] [-- EXTRA_ARGS...]",
    )
    parser.add_argument("-o", "--output", default="hstu_scratch_profile",
                        help="Output report name (default: hstu_scratch_profile)")
    parser.add_argument("--kernel-regex", default=None,
                        help="Custom kernel filter regex for ncu")
    parser.add_argument("extra_args", nargs=argparse.REMAINDER,
                        help="Extra args passed to profile_ncu.py (after --)")
    return parser.parse_args()


def build_ncu_command(script_dir: Path, output_path: Path, python_cmd: str,
                      extra_args: list):
    """Build the ncu command line."""
    cmd = [
        "ncu",
        "--set", "full",
        "--profile-from-start", "off",
        "-o", str(output_path),
        "--force-overwrite",
        "--kernel-id=::device_kernel:",
        python_cmd, str(script_dir / "profile_ncu.py"),
    ]
    cmd.extend(extra_args)
    return cmd


def main():
    script_dir, _ = get_paths()
    args = parse_args()
    python_cmd = os.environ.get("PYTHON", "python")

    output_dir = script_dir / "ncu_reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.output

    # Strip leading '--' from extra_args if present
    extra_args = args.extra_args
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]

    print(f"NCU profiling: {output_path}.ncu-rep")
    print(f"Extra args: {' '.join(extra_args) if extra_args else 'none'}")

    cmd = build_ncu_command(script_dir, output_path, python_cmd, extra_args)
    subprocess.run(cmd, check=True)

    print()
    print(f"Report saved: {output_path}.ncu-rep")
    print(f"View with: ncu-ui {output_path}.ncu-rep")


if __name__ == "__main__":
    main()
