"""Custom setup.py for building Rust binary during pip install."""

import subprocess
import shutil
from pathlib import Path
from setuptools import setup
from setuptools.command.build_py import build_py


class BuildRustCommand(build_py):
    """Custom build command that builds the Rust binary."""

    def run(self):
        # Build the Rust binary
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


setup(
    cmdclass={
        "build_py": BuildRustCommand,
    },
)

