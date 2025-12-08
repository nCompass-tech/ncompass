"""
Setup script for ncompass package.

This setup.py is required to properly install .pth files to site-packages root.
The pyproject.toml handles the main package configuration, but .pth files need
special handling to be placed at the site-packages root level (not inside the
package directory) for Python to process them during startup.
"""

from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py


class CustomBuildPy(build_py):
    """Custom build_py that copies .pth and init files to build directory."""

    def run(self):
        # Run the standard build_py first
        super().run()

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
    # This setup.py only handles special file installation
    cmdclass={
        "build_py": CustomBuildPy,
    },
    # Install .pth and init files to site-packages root
    # The empty string '' means the root of the installation directory (site-packages)
    data_files=[
        ("", ["ncompass.pth", "ncompass_init.py"]),
    ],
)
