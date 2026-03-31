#!/usr/bin/env python3
"""Docker management script for vllm_example.

This example has additional --wheel argument for vllm wheel installation.
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path

VLLM_REPO_URL = "https://github.com/vllm-project/vllm.git"
VLLM_SRC_DIR = "vllm_src"
VERSION_MARKER_FILE = ".vllm_version"
VLLM_NIGHTLY_INDEX = "https://wheels.vllm.ai/nightly/cu130/vllm/"


@dataclass(frozen=True)
class VllmConfig:
    """CLI options that flow through to install hooks."""
    wheel_name: str | None = None
    nightly: bool = False


def parse_wheel_version(wheel_filename: str) -> str:
    """Extract version from wheel filename and return corresponding git tag.

    Args:
        wheel_filename: Wheel filename (e.g., 'vllm-0.12.0+cu130-...-x86_64.whl')

    Returns:
        Git tag (e.g., 'v0.12.0')

    Raises:
        ValueError: If version cannot be parsed from filename
    """
    pattern = r'^vllm-(\d+\.\d+\.\d+(?:\.post\d+)?)\+'
    match = re.match(pattern, wheel_filename)

    if not match:
        raise ValueError(f"Cannot parse version from wheel filename: {wheel_filename}")

    version = match.group(1)
    return f"v{version}"


def get_current_vllm_version(vllm_src_path: Path) -> str | None:
    """Get the version of currently cloned vLLM source.

    Args:
        vllm_src_path: Path to vllm_src directory

    Returns:
        Version tag string if found, None otherwise
    """
    version_file = vllm_src_path / VERSION_MARKER_FILE

    if version_file.exists():
        return version_file.read_text().strip()

    # Fallback: try git describe
    try:
        result = subprocess.run(
            ["git", "describe", "--tags", "--exact-match"],
            cwd=vllm_src_path,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def clone_vllm_source(target_dir: Path, git_tag: str) -> None:
    """Clone vLLM source code at the specified tag.

    Args:
        target_dir: Directory to clone into (e.g., /path/to/vllm_src)
        git_tag: Git tag to checkout (e.g., 'v0.12.0')

    Raises:
        subprocess.CalledProcessError: If git clone fails
    """
    print(f"Cloning vLLM source code (tag: {git_tag})...")

    subprocess.run(
        [
            "git", "clone",
            "--depth", "1",
            "--branch", git_tag,
            VLLM_REPO_URL,
            str(target_dir)
        ],
        check=True
    )

    # Write version marker for quick future checks
    version_file = target_dir / VERSION_MARKER_FILE
    version_file.write_text(git_tag)

    print(f"vLLM source cloned successfully to {target_dir}")


def resolve_nightly_info() -> tuple[str, str]:
    """Fetch the nightly wheel index and return (full_commit_hash, wheel_filename).

    Parses the index page at wheels.vllm.ai to find the latest x86_64 nightly
    wheel and extract the git commit it was built from.

    Returns:
        Tuple of (full_commit_hash, wheel_filename)

    Raises:
        RuntimeError: If the index cannot be fetched or parsed
    """
    import urllib.request

    print(f"Fetching nightly wheel index from {VLLM_NIGHTLY_INDEX}...")
    try:
        with urllib.request.urlopen(VLLM_NIGHTLY_INDEX, timeout=15) as resp:
            html = resp.read().decode()
    except Exception as e:
        raise RuntimeError(f"Failed to fetch nightly index: {e}") from e

    # Find x86_64 wheel links: href contains a full commit hash in the path
    # e.g. ../../../<commit>/vllm-...-manylinux_2_35_x86_64.whl
    pattern = r'href="[^"]*?/([0-9a-f]{40})/([^"]*x86_64\.whl)"'
    matches = re.findall(pattern, html)
    if not matches:
        raise RuntimeError("No x86_64 nightly wheel found in index")

    # Take the last match (most recent)
    commit_hash, wheel_filename = matches[-1]
    # URL-decode the filename (e.g. %2B -> +)
    wheel_filename = urllib.request.url2pathname(wheel_filename).split("/")[-1]

    print(f"  Latest nightly: {wheel_filename}")
    print(f"  Commit: {commit_hash}")
    return commit_hash, wheel_filename


def clone_vllm_at_commit(target_dir: Path, commit_hash: str) -> None:
    """Clone vLLM source at a specific commit (shallow).

    Args:
        target_dir: Directory to clone into
        commit_hash: Full git commit hash
    """
    print(f"Cloning vLLM source at commit {commit_hash[:10]}...")

    target_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init"], cwd=target_dir, check=True,
                   capture_output=True)
    subprocess.run(["git", "remote", "add", "origin", VLLM_REPO_URL],
                   cwd=target_dir, check=True, capture_output=True)
    subprocess.run(["git", "fetch", "--depth", "1", "origin", commit_hash],
                   cwd=target_dir, check=True)
    subprocess.run(["git", "checkout", "FETCH_HEAD"],
                   cwd=target_dir, check=True, capture_output=True)

    # Write version marker
    (target_dir / VERSION_MARKER_FILE).write_text(f"nightly:{commit_hash}")
    print(f"vLLM source cloned at {commit_hash[:10]} to {target_dir}")


def prepare_nightly_source() -> tuple[str, str]:
    """Resolve nightly info and clone source. Returns (commit_hash, wheel_filename)."""
    vllm_src_path = Path.cwd() / VLLM_SRC_DIR

    commit_hash, wheel_filename = resolve_nightly_info()
    marker = f"nightly:{commit_hash}"

    if vllm_src_path.exists():
        current = get_current_vllm_version(vllm_src_path)
        if current == marker:
            print(f"vLLM source already at nightly commit ({commit_hash[:10]}), skipping clone.")
            return commit_hash, wheel_filename
        print(f"vLLM source version mismatch: have {current}, need {marker}")
        print("Removing existing vllm_src directory...")
        shutil.rmtree(vllm_src_path)

    clone_vllm_at_commit(vllm_src_path, commit_hash)
    return commit_hash, wheel_filename


def prepare_vllm_source(wheel_file: Path) -> None:
    """Ensure vLLM source code is available at the correct version.

    This function should be called on the HOST before container operations.
    It clones or updates the vLLM source to match the wheel version.

    Args:
        wheel_file: Path to the wheel file being used
    """
    # Determine the target directory (same directory as wheel_file's parent's parent)
    example_dir = wheel_file.parent.parent  # wheels/ -> vllm_example/
    vllm_src_path = example_dir / VLLM_SRC_DIR

    # Parse required version from wheel filename
    required_tag = parse_wheel_version(wheel_file.name)

    if vllm_src_path.exists():
        # Check if existing source matches required version
        current_version = get_current_vllm_version(vllm_src_path)

        if current_version == required_tag:
            print(f"vLLM source already at correct version ({required_tag}), skipping clone.")
            return
        else:
            print(f"vLLM source version mismatch: have {current_version}, need {required_tag}")
            print("Removing existing vllm_src directory...")
            shutil.rmtree(vllm_src_path)

    # Clone fresh with correct tag
    clone_vllm_source(vllm_src_path, required_tag)


def find_wheel_from_vllm_src(vllm_src_path: Path, wheels_dir: Path) -> Path:
    """Find the wheel file matching the vllm_src version.

    Args:
        vllm_src_path: Path to vllm_src directory
        wheels_dir: Path to wheels directory

    Returns:
        Path to matching wheel file

    Raises:
        FileNotFoundError: If vllm_src doesn't exist or no matching wheel
    """
    version_file = vllm_src_path / VERSION_MARKER_FILE
    if not version_file.exists():
        raise FileNotFoundError(
            f"vllm_src not found or not properly set up.\n"
            f"Run: python nc_pkg.py --setup --docker-dir ../docker --wheel <wheel_file>"
        )

    # Read version (e.g., "v0.10.2")
    version_tag = version_file.read_text().strip()
    # Convert to wheel version pattern (e.g., "0.10.2")
    wheel_version = version_tag.lstrip('v')

    # Find matching wheel
    wheel_files = list(wheels_dir.glob(f"vllm-{wheel_version}+*.whl"))
    if not wheel_files:
        raise FileNotFoundError(
            f"No wheel found matching vllm_src version {version_tag}.\n"
            f"Available wheels: {[f.name for f in sorted(wheels_dir.glob('vllm*.whl'))]}"
        )

    return wheel_files[0]


def find_docker_dir(required: bool = True) -> Path | None:
    """Find docker directory via local symlink (created by --setup)."""
    local = Path(__file__).parent / 'docker'
    if local.exists() and local.is_symlink():
        return local.resolve()

    if required:
        print("Error: Docker symlink not found.")
        print("Run: python nc_pkg.py --setup --docker-dir <path> --wheel <wheel_file>")
        sys.exit(1)

    return None


def _setup_docker_imports():
    """Add docker directory to Python path and import nc_pkg_lib."""
    docker_dir = find_docker_dir()
    sys.path.insert(0, str(docker_dir))

    from nc_pkg_lib import (
        main as base_main,
        get_compose_files,
        get_compose_env,
        execute_in_container,
    )
    return base_main, get_compose_files, get_compose_env, execute_in_container

def install_vllm(
    compose_files: list[str], env: dict[str, str], service_name: str,
    *, cfg: VllmConfig,
) -> None:
    """Install vllm using the precompiled wheel (release or nightly)."""
    _, _, _, execute_in_container = _setup_docker_imports()

    if cfg.nightly:
        _install_vllm_nightly(compose_files, env, service_name, execute_in_container)
    else:
        _install_vllm_release(compose_files, env, service_name, execute_in_container, cfg=cfg)


def _install_vllm_nightly(
    compose_files: list[str], env: dict[str, str],
    service_name: str, execute_in_container,
) -> None:
    """Install vLLM from nightly wheel + editable source."""
    try:
        commit_hash, wheel_filename = prepare_nightly_source()
    except (RuntimeError, subprocess.CalledProcessError) as e:
        print(f"Error preparing nightly source: {e}", file=sys.stderr)
        sys.exit(1)

    # Download the nightly wheel via curl, then install editably with precompiled binaries
    wheel_url = f"https://wheels.vllm.ai/nightly/cu130/{commit_hash}/{wheel_filename}"
    install_cmd = (
        f"curl -fsSL -o /tmp/{wheel_filename} '{wheel_url}'"
        f" && VLLM_PRECOMPILED_WHEEL_LOCATION=/tmp/{wheel_filename}"
        " uv pip install -e vllm_src/"
    )

    print(f"Installing vLLM nightly (commit {commit_hash[:10]}) editably...")
    result = execute_in_container(
        compose_files, env, service_name,
        ["/bin/bash", "-c", install_cmd],
        interactive=False,
    )

    if result.stdout:
        print(result.stdout, end='')
    if result.stderr:
        print(result.stderr, end='', file=sys.stderr)

    if result.returncode != 0:
        print(f"Warning: nightly vllm installation failed with exit code {result.returncode}")
    else:
        print("vLLM nightly installation complete.")


def _install_vllm_release(
    compose_files: list[str], env: dict[str, str],
    service_name: str, execute_in_container,
    *, cfg: VllmConfig,
) -> None:
    """Install vLLM from a local precompiled release wheel."""
    wheels_dir = Path.cwd() / "wheels"
    vllm_src_path = Path.cwd() / VLLM_SRC_DIR

    if not wheels_dir.exists():
        print(f"Error: wheels directory not found at {wheels_dir}", file=sys.stderr)
        sys.exit(1)

    if cfg.wheel_name:
        wheel_file = wheels_dir / cfg.wheel_name
        if not wheel_file.exists():
            wheel_files = list(wheels_dir.glob("vllm*.whl"))
            print(f"Error: Specified wheel '{cfg.wheel_name}' not found in {wheels_dir}", file=sys.stderr)
            print(f"Available wheels: {[f.name for f in sorted(wheel_files)]}", file=sys.stderr)
            sys.exit(1)
    else:
        try:
            wheel_file = find_wheel_from_vllm_src(vllm_src_path, wheels_dir)
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    wheel_file = wheel_file.absolute()
    print(f"Selected vLLM wheel: {wheel_file.name}")

    try:
        prepare_vllm_source(wheel_file)
    except (ValueError, subprocess.CalledProcessError) as e:
        print(f"Error preparing vLLM source: {e}", file=sys.stderr)
        sys.exit(1)

    install_cmd = f"VLLM_PRECOMPILED_WHEEL_LOCATION={wheel_file} uv pip install -e vllm_src/"

    print(f"Installing vllm with precompiled wheel: {wheel_file}...")
    result = execute_in_container(
        compose_files, env, service_name,
        ["/bin/bash", "-c", install_cmd],
        interactive=False,
    )

    if result.stdout:
        print(result.stdout, end='')
    if result.stderr:
        print(result.stderr, end='', file=sys.stderr)

    if result.returncode != 0:
        print(f"Warning: vllm installation failed with exit code {result.returncode}")
    else:
        print("vllm installation complete.")


def add_wheel_arg(parser: argparse.ArgumentParser) -> None:
    """Add wheel and nightly arguments to parser."""
    parser.add_argument(
        '--wheel', type=str, metavar='<wheel_file>',
        help='Specify which vllm wheel file to use (e.g., vllm-0.12.0+cu130-cp38-abi3-manylinux_2_31_x86_64.whl). '
             'Required with --setup. Optional during --run/--exec (auto-detected from vllm_src).'
    )
    parser.add_argument(
        '--nightly', action='store_true',
        help='Build with vLLM nightly support (Python 3.12, cu130 PyTorch index, system cuBLAS LD_PRELOAD).'
    )


def vllm_setup_hook(cfg: VllmConfig) -> None:
    """Setup hook for vLLM example - clone source matching wheel."""
    wheels_dir = Path.cwd() / "wheels"

    if not cfg.wheel_name:
        print("Error: --wheel is required with --setup for vllm_example", file=sys.stderr)
        if wheels_dir.exists():
            wheel_files = sorted(wheels_dir.glob("vllm*.whl"))
            if wheel_files:
                print("Available wheels:", file=sys.stderr)
                for f in wheel_files:
                    print(f"  {f.name}", file=sys.stderr)
        sys.exit(1)

    wheel_file = (wheels_dir / cfg.wheel_name).absolute()
    if not wheel_file.exists():
        print(f"Error: Wheel not found: {wheel_file}", file=sys.stderr)
        if wheels_dir.exists():
            wheel_files = sorted(wheels_dir.glob("vllm*.whl"))
            if wheel_files:
                print("Available wheels:", file=sys.stderr)
                for f in wheel_files:
                    print(f"  {f.name}", file=sys.stderr)
        sys.exit(1)

    try:
        prepare_vllm_source(wheel_file)
    except (ValueError, subprocess.CalledProcessError) as e:
        print(f"Error preparing vLLM source: {e}", file=sys.stderr)
        sys.exit(1)


def _handle_setup_locally(docker_dir: Path, cfg: VllmConfig) -> None:
    """Handle --setup without importing from nc_pkg_lib (which needs the symlink)."""
    example_dir = Path.cwd()
    symlink_path = example_dir / "docker"

    # Resolve the docker_dir to absolute path
    docker_dir = docker_dir.resolve()

    if not docker_dir.exists():
        print(f"Error: Docker directory does not exist: {docker_dir}", file=sys.stderr)
        sys.exit(1)

    # Create/verify symlink
    if symlink_path.exists():
        if symlink_path.is_symlink():
            current_target = symlink_path.resolve()
            if current_target == docker_dir:
                print(f"Docker symlink already configured: {symlink_path} -> {docker_dir}")
            else:
                print(f"Updating docker symlink: {symlink_path} -> {docker_dir}")
                symlink_path.unlink()
                symlink_path.symlink_to(docker_dir)
        else:
            print(f"Error: {symlink_path} exists but is not a symlink", file=sys.stderr)
            sys.exit(1)
    else:
        print(f"Creating docker symlink: {symlink_path} -> {docker_dir}")
        symlink_path.symlink_to(docker_dir)

    # Run vllm-specific setup hook (clones vLLM source)
    vllm_setup_hook(cfg)

    print("Setup complete!")


def main():
    # Pre-parse to get wheel name and detect --setup before main parsing
    # This is needed because --setup must be handled before importing nc_pkg_lib
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument('--wheel', type=str)
    pre_parser.add_argument('--setup', action='store_true')
    pre_parser.add_argument('--docker-dir', type=Path)
    pre_parser.add_argument('--nightly', action='store_true')
    pre_args, _ = pre_parser.parse_known_args()

    cfg = VllmConfig(wheel_name=pre_args.wheel, nightly=pre_args.nightly)

    # Set build-arg env vars for docker compose when --nightly is used
    if cfg.nightly:
        os.environ['VLLM_NIGHTLY'] = '1'
        os.environ['PYTHON_VERSION'] = '3.12'

    # Handle --setup locally (before importing nc_pkg_lib which needs the docker symlink)
    if pre_args.setup:
        existing_docker = find_docker_dir(required=False)
        if pre_args.docker_dir:
            docker_dir = pre_args.docker_dir
        elif existing_docker:
            docker_dir = existing_docker
        else:
            print("Error: --docker-dir is required with --setup (no existing docker symlink found)", file=sys.stderr)
            sys.exit(1)
        _handle_setup_locally(docker_dir, cfg)
        return

    # For non-setup operations, import and use base_main
    base_main, _, _, _ = _setup_docker_imports()

    # Get env_config.yaml path
    env_config_path = Path(__file__).parent / "env_config.yaml"

    # Call base main with vllm-specific hooks
    base_main(
        service_name="vllm_example",
        env_config_path=env_config_path if env_config_path.exists() else None,
        post_install_hook=partial(install_vllm, cfg=cfg),
        extra_args_handler=add_wheel_arg,
        args=sys.argv[1:]  # Pass original args
    )


if __name__ == '__main__':
    main()
