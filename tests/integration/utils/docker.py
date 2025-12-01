"""Utility functions for integration tests."""

import os
import subprocess
from pathlib import Path
from typing import Optional

def _get_compose_env(example_dir: Path) -> dict[str, str]:
    """
    Get environment variables needed for docker compose commands.
    
    Sets up CURRENT_DIR, HOME, UID, GID, and DISPLAY for docker compose.
    
    Args:
        example_dir: Example directory path
    
    Returns:
        Dictionary of environment variables
    """
    env = os.environ.copy()
    
    # Set current directory (paths are now identical between host and container)
    env['CURRENT_DIR'] = str(example_dir.absolute())
    
    # Set HOME if not already set
    if 'HOME' not in env:
        env['HOME'] = str(Path.home())
    
    # Set UID, GID, DISPLAY
    env['UID'] = str(os.getuid())
    env['GID'] = str(os.getgid())
    if 'DISPLAY' not in env:
        env['DISPLAY'] = ':0'

    return env

def build_docker_image(example_dir: Path, tag: str = "0.0.1", name: str = "test") -> str:
    """
    Build Docker image using docker compose.
    
    Args:
        example_dir: Directory containing docker-compose.yaml
        tag: Docker image tag
        name: Service name
    
    Returns:
        Image name identifier (e.g., "nc_{name}:{tag}")
    """
    image_name = f"nc_{name}:{tag}"
    
    # Build using docker compose
    original_cwd = os.getcwd()
    try:
        os.chdir(str(example_dir))
        
        # Check if ncompass directory exists to determine which compose files to use
        ncompass_dir = example_dir / "ncompass"
        compose_files = ["-f", "docker-compose.yaml"]
        
        if ncompass_dir.exists():
            compose_files.extend(["-f", "docker-compose.ncompass.yaml"])
        
        # Set up environment variables
        env = _get_compose_env(example_dir)
        
        build_cmd = ["docker", "compose"] + compose_files + ["build"]
        result = subprocess.run(
            build_cmd,
            capture_output=True,
            text=True,
            check=False,
            env=env
        )
        
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to build Docker image with docker compose:\n"
                f"stdout: {result.stdout}\nstderr: {result.stderr}"
            )
    finally:
        os.chdir(original_cwd)
    
    return image_name

def run_docker_command(
    image_name: str,
    command: list[str],
    workdir: Path,
    example_dir: Path,
    mounts: Optional[dict[str, str]] = None,
    env_vars: Optional[dict[str, str]] = None,
) -> subprocess.CompletedProcess:
    """
    Run a command inside a Docker container using docker compose.
    
    Args:
        image_name: Docker image name (used to identify service in docker-compose)
        command: Command to run (list of strings)
        workdir: Working directory on host
        example_dir: Example directory containing docker-compose.yaml
        mounts: Additional mounts as {host_path: container_path} (not used with docker-compose)
        env_vars: Environment variables as {key: value}
    
    Returns:
        CompletedProcess with stdout, stderr, and returncode
    """
    # Service name matches the directory name (as defined in docker-compose.yaml)
    service_name = example_dir.name
    
    # Set up base environment variables
    env = _get_compose_env(example_dir)
    
    # Override CURRENT_DIR with workdir (for test runs, we mount workdir, not example_dir)
    env['CURRENT_DIR'] = str(workdir.absolute())
    
    # Add additional environment variables if provided
    if env_vars:
        env.update(env_vars)
    
    # Determine example name from example_dir to find ncompass directory in workdir
    example_name = example_dir.name
    ncompass_dir = workdir / example_name / "ncompass"
    
    # Build compose file list: base + ncompass (if exists)
    compose_files = ["-f", "docker-compose.yaml"]
    
    # Check if ncompass exists in workdir
    if ncompass_dir.exists():
        compose_files.extend(["-f", "docker-compose.ncompass.yaml"])
    
    # Change to example directory to run docker compose
    original_cwd = os.getcwd()
    try:
        os.chdir(str(example_dir))
        
        # Use docker compose run to execute the command
        # --rm removes the container after execution
        # -T disables pseudo-TTY allocation for non-interactive use
        compose_cmd = [
            "docker", "compose"
        ] + compose_files + [
            "run",
            "--rm",
            "-T",
            service_name
        ] + command
        
        result = subprocess.run(
            compose_cmd,
            capture_output=True,
            text=True,
            check=False,
            env=env
        )
        
    finally:
        os.chdir(original_cwd)
    
    return result

def create_symlink(link: Path, target: Path) -> None:
    """Create a symlink from source to target."""
    if link.exists(): remove_symlink(link)
    subprocess.run(
            ["ln", "-s", f"{target}", f"{link}"],
            check=True
    )

def remove_symlink(path: Path) -> None:
    """Remove a symlink safely."""
    subprocess.run(
            ["rm", f"{path}"],
            check=True
    )

