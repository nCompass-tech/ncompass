#!/usr/bin/env python3
"""Docker management script for generative-recommender-kernels.

Simplified wrapper around the shared Docker infrastructure.
Installs generative-recommenders dependencies as a post-install hook.
"""
import argparse
import subprocess
import sys
from pathlib import Path

def find_docker_dir(required: bool = True) -> Path | None:
    """Find docker directory via local symlink (created by --setup)."""
    local = Path(__file__).parent / 'docker'
    if local.exists() and local.is_symlink():
        return local.resolve()

    if required:
        print("Error: Docker symlink not found.")
        print("Run: python nc_pkg.py --setup --docker-dir ../../docker")
        sys.exit(1)

    return None


def _setup_docker_imports():
    """Add docker directory to Python path and import nc_pkg_lib."""
    docker_dir = find_docker_dir()
    sys.path.insert(0, str(docker_dir))

    from nc_pkg_lib import (
        main as base_main,
        execute_in_container,
    )
    return base_main, execute_in_container


def patch_reference_sources() -> None:
    """Apply patches to generative-recommenders source via git apply.

    Idempotent: skips if submodule already has local changes.
    """
    submodule = Path(__file__).parent / "generative-recommenders"
    patch_dir = Path(__file__).parent / "fa3" / "reference"
    patch_files = [
        patch_dir / "cpp_compat.patch",
        patch_dir / "hstu_attention_import.patch",
    ]

    if not submodule.exists():
        return

    # Skip if already patched (submodule has local changes)
    result = subprocess.run(
        ["git", "diff", "--quiet"],
        cwd=submodule,
        capture_output=True,
    )
    if result.returncode != 0:
        # Already has local changes — assume patched
        return

    print("Applying patches to generative-recommenders...")
    for patch_file in patch_files:
        if not patch_file.exists():
            print(f"Warning: {patch_file.name} not found, skipping.")
            continue
        result = subprocess.run(
            ["git", "apply", str(patch_file.resolve())],
            cwd=submodule,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"Warning: {patch_file.name} failed: {result.stderr}")
        else:
            print(f"Applied {patch_file.name}.")


def install_gr_deps(compose_files: list[str], env: dict[str, str], service_name: str) -> None:
    """Install generative-recommenders dependencies inside the container.

    Args:
        compose_files: List of compose file flags
        env: Environment variables dictionary
        service_name: Service name
    """
    patch_reference_sources()  # host-side, no container needed

    _, execute_in_container = _setup_docker_imports()

    requirements_file = Path.cwd() / "requirements.txt"
    if not requirements_file.exists():
        print(f"Warning: {requirements_file} not found, skipping dependency install.")
        return

    torch_install_cmd = \
            f"uv pip install torch --index-url https://download.pytorch.org/whl/cu130 --reinstall"
    install_cmd =  torch_install_cmd + " && " + "uv pip install -r requirements.txt"
    print("Installing generative-recommenders dependencies...")

    result = execute_in_container(
        compose_files,
        env,
        service_name,
        ["/bin/bash", "-c", install_cmd],
        interactive=False
    )

    if result.stdout:
        print(result.stdout, end='')
    if result.stderr:
        print(result.stderr, end='', file=sys.stderr)

    if result.returncode != 0:
        print(f"Warning: dependency installation failed with exit code {result.returncode}")
    else:
        print("Dependency installation complete.")


def _handle_setup_locally(docker_dir: Path) -> None:
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

    print("Setup complete!")


def main():
    # Pre-parse to detect --setup before main parsing
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument('--setup', action='store_true')
    pre_parser.add_argument('--docker-dir', type=Path)
    pre_args, _ = pre_parser.parse_known_args()

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
        _handle_setup_locally(docker_dir)
        return

    # For non-setup operations, import and use base_main
    base_main, _ = _setup_docker_imports()

    # Get env_config.yaml path
    env_config_path = Path(__file__).parent / "env_config.yaml"

    # Call base main with gr-specific hooks
    base_main(
        service_name="gr_kernels",
        env_config_path=env_config_path if env_config_path.exists() else None,
        post_install_hook=install_gr_deps,
        args=sys.argv[1:]
    )


if __name__ == '__main__':
    main()
