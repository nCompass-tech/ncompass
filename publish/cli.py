#!/usr/bin/env python3
"""
nCompass PyPI Publishing Script

Converts the bash publish.sh script to Python with YAML configuration support.
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

# Try to import tomllib (Python 3.11+) or fallback to tomli
try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        print("ERROR: tomllib not available. Install tomli: pip install tomli", file=sys.stderr)
        sys.exit(2)

# Try to import yaml
try:
    import yaml
except ImportError:
    print("ERROR: PyYAML not available. Install it: pip install pyyaml", file=sys.stderr)
    sys.exit(2)

# ANSI color codes
RED = '\033[0;31m'
GREEN = '\033[0;32m'
YELLOW = '\033[1;33m'
BLUE = '\033[0;34m'
NC = '\033[0m'  # No Color

# Constants
PACKAGE_NAME = "ncompass"
PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
CONFIG_FILE = PACKAGE_DIR / "config.yaml"


def log_info(message: str) -> None:
    """Print info message with blue color."""
    print(f"{BLUE}[INFO]{NC} {message}")


def log_success(message: str) -> None:
    """Print success message with green color."""
    print(f"{GREEN}[SUCCESS]{NC} {message}")


def log_warning(message: str) -> None:
    """Print warning message with yellow color."""
    print(f"{YELLOW}[WARNING]{NC} {message}")


def log_error(message: str) -> None:
    """Print error message with red color."""
    print(f"{RED}[ERROR]{NC} {message}")


def check_command(cmd: str) -> None:
    """Check if a command is available in PATH."""
    if not shutil.which(cmd):
        log_error(f"Required command '{cmd}' not found. Please install it first.")
        sys.exit(1)


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    if not config_path.exists():
        log_warning(f"Config file {config_path} not found. Using environment variables and defaults.")
        return {}
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f) or {}
        return config
    except yaml.YAMLError as e:
        log_error(f"Failed to parse {config_path}: {e}")
        sys.exit(1)
    except Exception as e:
        log_error(f"Failed to load {config_path}: {e}")
        sys.exit(1)


def get_config_value(config: dict, key: str, env_key: Optional[str] = None, default: Optional[str] = None) -> str:
    """Get config value from YAML config, environment variable, or default."""
    # First check environment variable (takes precedence)
    if env_key:
        env_value = os.environ.get(env_key)
        if env_value:
            return env_value
    
    # Then check YAML config
    if key in config:
        return str(config[key])
    
    # Finally use default
    return default if default is not None else ""


def get_package_version() -> str:
    """Extract package version from pyproject.toml."""
    pyproject_path = PROJECT_ROOT / "pyproject.toml"
    if not pyproject_path.exists():
        log_error("pyproject.toml not found. Run from project root.")
        sys.exit(1)
    
    try:
        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)
        return data["project"]["version"]
    except KeyError:
        log_error("Version not found in pyproject.toml")
        sys.exit(1)
    except Exception as e:
        log_error(f"Failed to read pyproject.toml: {e}")
        sys.exit(1)


def validate_env(config: dict, mode: str, dry_run: bool) -> tuple[str, str, str]:
    """Validate environment and return PyPI repository info."""
    if mode == "test":
        repo_name = "testpypi"
        simple_url = "https://test.pypi.org/simple/"
        project_url = f"https://test.pypi.org/project/{PACKAGE_NAME}/"
        
        if not dry_run:
            testpypi_token = get_config_value(config, "testpypi_token", "TESTPYPI_TOKEN")
            if not testpypi_token:
                log_error("TESTPYPI_TOKEN is required for --test (set in publish/config.yaml or env var)")
                sys.exit(1)
            if not testpypi_token.startswith("pypi-"):
                log_error("TESTPYPI_TOKEN must start with 'pypi-'")
                sys.exit(1)
    
    elif mode == "prod":
        repo_name = "pypi"
        simple_url = "https://pypi.org/simple/"
        project_url = f"https://pypi.org/project/{PACKAGE_NAME}/"
        
        if not dry_run:
            pypi_token = get_config_value(config, "pypi_token", "PYPI_TOKEN")
            pypi_username = get_config_value(config, "pypi_username", "PYPI_USERNAME")
            pypi_password = get_config_value(config, "pypi_password", "PYPI_PASSWORD")
            
            if pypi_token:
                if not pypi_token.startswith("pypi-"):
                    log_error("PYPI_TOKEN must start with 'pypi-'")
                    sys.exit(1)
            elif not (pypi_username and pypi_password):
                log_error("Provide PYPI_TOKEN or PYPI_USERNAME/PYPI_PASSWORD for --prod")
                log_error("Set in publish/config.yaml or environment variables")
                sys.exit(1)
    else:
        log_error("Internal: MODE not set")
        sys.exit(1)
    
    return repo_name, simple_url, project_url


def check_git_status(package_version: str) -> None:
    """Check git status and tags."""
    log_info("Checking git status...")
    
    # Check for uncommitted changes
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True,
        text=True
    )
    
    if result.stdout.strip():
        log_warning("Working directory has uncommitted changes")
        reply = input("Continue anyway? (y/N) ").strip()
        if reply.lower() != 'y':
            log_info("Aborted by user")
            sys.exit(1)
    else:
        log_success("Working directory is clean")
    
    # Check for existing tag
    tag_name = f"v{package_version}"
    result = subprocess.run(
        ["git", "rev-parse", tag_name],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        log_warning(f"Git tag {tag_name} already exists")
        reply = input("Continue anyway? (y/N) ").strip()
        if reply.lower() != 'y':
            log_info("Aborted by user")
            sys.exit(1)
    
    print()


def install_build_tools() -> None:
    """Install/upgrade build tools."""
    log_info("Installing/upgrading build tools...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--upgrade", "pip", "build", "twine", "setuptools", "wheel"],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        log_error("Failed to install build tools")
        sys.exit(1)
    log_success("Build tools ready")
    print()


def run_tests(skip_tests: bool) -> None:
    """Run test suite."""
    if skip_tests:
        log_warning("Skipping tests (skip_tests=true in config or SKIP_TESTS=true)")
        return
    
    log_info("Running test suite...")
    # Install dev dependencies (ignore errors)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", ".[dev]"],
        capture_output=True,
        text=True
    )
    
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "--cov=ncompass", "--cov-fail-under=80", "-q"]
    )
    
    if result.returncode != 0:
        log_error("Tests failed")
        sys.exit(1)
    
    log_success("All tests passed")
    print()


def run_quality_checks(skip_checks: bool) -> None:
    """Run code quality checks."""
    if skip_checks:
        log_warning("Skipping code quality checks (skip_checks=true in config or SKIP_CHECKS=true)")
        return
    
    log_info("Running code quality checks...")
    
    if not shutil.which("make"):
        log_error("make command not found")
        sys.exit(1)
    
    # Run type-stats
    log_info("  - type-stats")
    result = subprocess.run(
        ["make", "type-stats"],
        cwd="tools"
    )
    if result.returncode != 0:
        log_error("make type-stats failed")
        sys.exit(1)
    
    # Run docstring-coverage
    log_info("  - docstring-coverage")
    result = subprocess.run(
        ["make", "docstring-coverage"],
        cwd="tools"
    )
    if result.returncode != 0:
        log_error("make docstring-coverage failed")
        sys.exit(1)
    
    log_success("Code quality checks passed")
    print()


def build_package() -> None:
    """Build the package."""
    log_info("Cleaning previous builds...")
    
    # Clean build artifacts
    for path in ["build", "dist", "ncompass.egg-info"]:
        if Path(path).exists():
            if Path(path).is_dir():
                shutil.rmtree(path)
            else:
                Path(path).unlink()
    
    # Clean any .egg-info files
    for egg_info in Path(".").glob("*.egg-info"):
        if egg_info.is_dir():
            shutil.rmtree(egg_info)
    
    log_success("Clean complete")
    print()
    
    log_info("Building package...")
    result = subprocess.run([sys.executable, "-m", "build"])
    
    if result.returncode != 0:
        log_error("Build failed")
        sys.exit(1)
    
    dist_dir = Path("dist")
    if not dist_dir.exists() or not any(dist_dir.iterdir()):
        log_error("Build failed - no distribution files created")
        sys.exit(1)
    
    log_success("Package built")
    log_info("Artifacts:")
    for artifact in dist_dir.iterdir():
        size = artifact.stat().st_size
        size_str = f"{size / 1024:.1f}K" if size < 1024 * 1024 else f"{size / (1024 * 1024):.1f}M"
        print(f"  {artifact.name} ({size_str})")
    print()
    
    log_info("Checking distribution with twine...")
    result = subprocess.run(
        [sys.executable, "-m", "twine", "check", "dist/*"]
    )
    
    if result.returncode != 0:
        log_error("twine check failed")
        sys.exit(1)
    
    log_success("Distribution check passed")
    print()


def upload_package(config: dict, mode: str, repo_name: str, dry_run: bool) -> None:
    """Upload package to PyPI."""
    if dry_run:
        log_success("DRY RUN complete. Skipping upload.")
        log_info("Artifacts in dist/")
        return
    
    log_info(f"Uploading to {mode.upper()} ({repo_name})...")
    
    dist_files = list(Path("dist").glob("*"))
    if not dist_files:
        log_error("No distribution files found in dist/")
        sys.exit(1)
    
    if mode == "test":
        testpypi_token = get_config_value(config, "testpypi_token", "TESTPYPI_TOKEN")
        cmd = [
            sys.executable, "-m", "twine", "upload",
            "--repository", repo_name,
            "--username", "__token__",
            "--password", testpypi_token,
        ] + [str(f) for f in dist_files]
    else:  # prod
        pypi_token = get_config_value(config, "pypi_token", "PYPI_TOKEN")
        if pypi_token:
            cmd = [
                sys.executable, "-m", "twine", "upload",
                "--username", "__token__",
                "--password", pypi_token,
            ] + [str(f) for f in dist_files]
        else:
            pypi_username = get_config_value(config, "pypi_username", "PYPI_USERNAME")
            pypi_password = get_config_value(config, "pypi_password", "PYPI_PASSWORD")
            cmd = [
                sys.executable, "-m", "twine", "upload",
                "--username", pypi_username,
                "--password", pypi_password,
            ] + [str(f) for f in dist_files]
    
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        log_error("Upload failed")
        sys.exit(1)
    
    log_success("Upload complete")
    print()


def print_summary(package_name: str, package_version: str, simple_url: str, project_url: str, mode: str) -> None:
    """Print release summary."""
    log_success("Release complete")
    print()
    print("=" * 50)
    print(f"  Package: {package_name}")
    print(f"  Version: {package_version}")
    print(f"  Index:   {simple_url}")
    print(f"  Project: {project_url}")
    print("=" * 50)
    print()
    
    if mode == "test":
        log_info(f"Test install: pip install --index-url {simple_url} {package_name}")
    else:
        log_info(f"Install with: pip install {package_name}")


def print_help(script_name: str) -> None:
    """Print help message."""
    help_text = f"""
Usage: publish [--test | --prod] [options]

Modes (required):
  --test                Upload to TestPyPI
  --prod                Upload to PyPI

Configuration:
  Configuration is loaded from publish/config.yaml (if present) or environment variables.
  Environment variables take precedence over YAML config.

Options in publish/config.yaml or env vars:
  pypi_token / PYPI_TOKEN
                        PyPI API token (prod). Must start with 'pypi-'
  pypi_username / PYPI_USERNAME
                        PyPI username (legacy prod auth)
  pypi_password / PYPI_PASSWORD
                        PyPI password (legacy prod auth)
  testpypi_token / TESTPYPI_TOKEN
                        TestPyPI API token (test). Must start with 'pypi-'
  skip_tests / SKIP_TESTS
                        Skip pytest (true/false)
  skip_checks / SKIP_CHECKS
                        Skip coverage/pyright checks (true/false)
  dry_run / DRY_RUN    Build and check, but do not upload (true/false)
  use_ai_profiling / USE_AI_PROFILING
                        Enable AI profiling during tests (true/false)

Examples:
  SKIP_TESTS=true {script_name} --test
  PYPI_TOKEN=... {script_name} --prod
"""
    print(help_text)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="nCompass PyPI Publishing Script",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Upload to TestPyPI"
    )
    parser.add_argument(
        "--prod",
        action="store_true",
        help="Upload to PyPI"
    )
    
    args = parser.parse_args()
    
    # Determine mode
    if args.test and args.prod:
        log_error("Cannot specify both --test and --prod")
        sys.exit(1)
    elif not args.test and not args.prod:
        print_help(Path(sys.argv[0]).name)
        sys.exit(1)
    
    mode = "test" if args.test else "prod"
    
    # Load configuration
    config_path = CONFIG_FILE
    config = load_config(config_path)
    
    # Get configuration values (with env var fallback)
    skip_tests = get_config_value(config, "skip_tests", "SKIP_TESTS", "false").lower() == "true"
    skip_checks = get_config_value(config, "skip_checks", "SKIP_CHECKS", "false").lower() == "true"
    dry_run = get_config_value(config, "dry_run", "DRY_RUN", "false").lower() == "true"
    use_ai_profiling = get_config_value(
        config,
        "use_ai_profiling",
        "USE_AI_PROFILING",
        "false"
    ).lower() in ("true", "1", "yes")
    
    os.environ["USE_AI_PROFILING"] = "true" if use_ai_profiling else "false"
    log_info(f"USE_AI_PROFILING set to {os.environ['USE_AI_PROFILING']} via publish config")
    
    # Pre-flight checks
    log_info(f"Starting publish workflow for {PACKAGE_NAME} in mode: {mode}")
    os.chdir(PROJECT_ROOT)
    log_info(f"Working directory set to {PROJECT_ROOT}")
    
    if not (PROJECT_ROOT / "pyproject.toml").exists():
        log_error("pyproject.toml not found. Run from project root.")
        sys.exit(1)
    
    check_command("python")
    check_command("git")
    
    python_version = subprocess.run(
        [sys.executable, "--version"],
        capture_output=True,
        text=True
    ).stdout.strip()
    log_info(f"Using {python_version}")
    
    package_version = get_package_version()
    log_info(f"Package version: {package_version}")
    
    repo_name, simple_url, project_url = validate_env(config, mode, dry_run)
    print()
    
    # Git checks
    check_git_status(package_version)
    
    # Install build tools
    install_build_tools()
    
    # Run tests
    run_tests(skip_tests)
    
    # Run quality checks
    run_quality_checks(skip_checks)
    
    # Build package
    build_package()
    
    # Upload package
    upload_package(config, mode, repo_name, dry_run)
    
    # Print summary
    print_summary(PACKAGE_NAME, package_version, simple_url, project_url, mode)


if __name__ == "__main__":
    main()

