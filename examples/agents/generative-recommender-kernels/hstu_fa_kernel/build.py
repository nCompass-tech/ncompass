#!/usr/bin/env python3
"""
Build the from-scratch kernel (hstu_fa_kernel/kernel/) as the `hstu_ai_optimized` package.
Registers ops under torch.ops.hstu_ai_optimized.

Usage:
    python hstu_fa_kernel/build.py
    PIP=pip3 PYTHON=python3 python hstu_fa_kernel/build.py
"""

import glob
import os
import subprocess
import sys
from pathlib import Path


def get_paths():
    """Return (script_dir, repo_root, ref_cpp_dir) as Path objects."""
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    ref_cpp_dir = repo_root / "generative-recommenders" / "generative_recommenders" / "ops" / "cpp"
    return script_dir, repo_root, ref_cpp_dir


def validate_kernel_sources(script_dir: Path):
    """Ensure hstu_fa_kernel/kernel/ is populated with header files."""
    kernel_dir = script_dir / "kernel"
    if not kernel_dir.is_dir() or not glob.glob(str(kernel_dir / "*.h")):
        print("Error: hstu_fa_kernel/kernel/ not populated. Kernel source should be git-tracked.")
        print("  Check that the hstu_fa_kernel/kernel/ directory exists in the repository.")
        sys.exit(1)


def ensure_cutlass_submodule(repo_root: Path, ref_cpp_dir: Path):
    """Initialize the cutlass submodule if not already present."""
    cutlass_header = ref_cpp_dir / "cutlass" / "include" / "cutlass" / "cutlass.h"
    if not cutlass_header.exists():
        print("Initializing cutlass submodule...")
        subprocess.run(
            ["git", "submodule", "update", "--init", "generative_recommenders/ops/cpp/cutlass"],
            cwd=repo_root / "generative-recommenders",
            check=True,
        )


def set_build_env():
    """Set environment variables for the build."""
    env_vars = {
        "FLASH_ATTENTION_DISABLE_BACKWARD": "TRUE",
        "FLASH_ATTENTION_DISABLE_FP16": "TRUE",
        "FLASH_ATTENTION_DISABLE_FP8": "TRUE",
        "FLASH_ATTENTION_DISABLE_HDIM64": "TRUE",
        "FLASH_ATTENTION_DISABLE_HDIM96": "TRUE",
        "FLASH_ATTENTION_DISABLE_HDIM128": "FALSE",
        "FLASH_ATTENTION_DISABLE_HDIM192": "TRUE",
        "FLASH_ATTENTION_DISABLE_HDIM256": "TRUE",
        "FLASH_ATTENTION_DISABLE_SM80": "TRUE",
    }
    os.environ.update(env_vars)


def ensure_package_dir(script_dir: Path):
    """Create the hstu_ai_optimized package directory if it doesn't exist."""
    init_file = script_dir / "hstu_ai_optimized" / "__init__.py"
    if not init_file.exists():
        init_file.parent.mkdir(parents=True, exist_ok=True)
        init_file.touch()


def run_pip_install(script_dir: Path):
    """Run pip install -e . --no-build-isolation."""
    pip_cmd = os.environ.get("PIP", "pip")
    python_cmd = os.environ.get("PYTHON", "python")

    print("Building hstu_ai_optimized (from-scratch kernel)...")
    subprocess.run(
        [pip_cmd, "install", "-e", ".", "--no-build-isolation"],
        cwd=script_dir,
        check=True,
    )
    print(f"Done. Verify: {python_cmd} -c 'import hstu_ai_optimized._C; print(\"hstu_ai_optimized OK\")'")


def main():
    script_dir, repo_root, ref_cpp_dir = get_paths()

    validate_kernel_sources(script_dir)
    ensure_cutlass_submodule(repo_root, ref_cpp_dir)
    set_build_env()
    ensure_package_dir(script_dir)
    run_pip_install(script_dir)


if __name__ == "__main__":
    main()
