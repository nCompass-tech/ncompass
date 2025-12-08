"""
Custom setup.py for ncompass package.

This setup.py handles:
1. Building Rust binary (nsys-chrome) during pip install
2. Installing .pth files to site-packages root for auto-initialization

The pyproject.toml handles the main package configuration, but these custom
build steps require setup.py with custom commands.
"""

import subprocess
import shutil
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py


class CustomBuildPy(build_py):
    """Custom build command that builds Rust binary and handles .pth files."""

    def run(self):
        # Build the Rust binary first
        rust_dir = Path(__file__).parent / "ncompass_rust" / "trace_converters"

        if rust_dir.exists():
            print("Building Rust binary (nsys-chrome)...")
            try:
                subprocess.run(
                    ["cargo", "build", "--release", "--target=x86_64-unknown-linux-musl"],
                    cwd=rust_dir,
                    check=True,
                )
                print("Rust binary built successfully.")
            except subprocess.CalledProcessError as e:
                print(f"Warning: Failed to build Rust binary: {e}")
                print("The package will work but use the slower Python implementation.")
            except FileNotFoundError:
                print("Warning: 'cargo' not found. Skipping Rust binary build.")
                print("The package will work but use the slower Python implementation.")
                print("To build the Rust binary, install Rust and run:")
                print("  cd ncompass_rust/trace_converters && cargo build --release --target=x86_64-unknown-linux-musl")

        # Run the standard build
        super().run()

        # Copy the Rust binary to the build directory
        rust_binary = rust_dir / "target" / "x86_64-unknown-linux-musl" / "release" / "nsys-chrome"
        if rust_binary.exists():
            # Copy to package data directory
            dest_dir = Path(self.build_lib) / "ncompass" / "bin"
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_file = dest_dir / "nsys-chrome"
            shutil.copy2(rust_binary, dest_file)
            # Make executable
            dest_file.chmod(0o755)
            print(f"Copied Rust binary to {dest_file}")
        else:
            print("Warning: Rust binary not found after build. Python fallback will be used.")

        # Copy ncompass.pth and ncompass_init.py to the build lib directory
        # These need to be at the root of site-packages, not inside ncompass/
        src_dir = Path(__file__).parent
        build_lib = Path(self.build_lib)

        for filename in ["ncompass.pth", "ncompass_init.py"]:
            src_file = src_dir / filename
            if src_file.exists():
                dst_file = build_lib / filename
                self.copy_file(str(src_file), str(dst_file))
                print(f"Copied {filename} to {build_lib}")


setup(
    # All metadata comes from pyproject.toml
    cmdclass={
        "build_py": CustomBuildPy,
    },
    # Install .pth and init files to site-packages root
    # The empty string '' means the root of the installation directory (site-packages)
    data_files=[
        ("", ["ncompass.pth", "ncompass_init.py"]),
    ],
)
