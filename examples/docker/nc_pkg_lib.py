"""
Shared Docker compose utilities for nCompass examples.

This module provides common functions for managing Docker containers across
all nCompass examples. Individual example nc_pkg.py files import from here
and configure with their specific service names.

Usage in example nc_pkg.py:
    from docker.nc_pkg_lib import main

    if __name__ == '__main__':
        main(service_name="my_example")
"""
import os
import sys
import argparse
import subprocess
from pathlib import Path
from typing import Callable, Optional


def get_docker_dir() -> Path:
    """Get the path to the docker directory containing this file."""
    return Path(__file__).parent


def get_compose_files(example_dir: Path) -> list[str]:
    """
    Get the list of docker compose files to use.

    Args:
        example_dir: Path to the example directory

    Returns:
        List of compose file flags (e.g., ["-f", "docker-compose.yaml"])
    """
    compose_file = example_dir / "docker-compose.yaml"
    if not compose_file.exists():
        print(f"Error: docker-compose.yaml not found in {example_dir}")
        sys.exit(1)

    return ["-f", str(compose_file)]


def get_compose_env(env_config_path: Optional[Path] = None) -> dict[str, str]:
    """
    Get environment variables needed for docker compose commands.

    Reads from env_config.yaml if provided and sets up CURRENT_DIR, HOME,
    UID, GID, and DISPLAY.

    Args:
        env_config_path: Optional path to env_config.yaml file

    Returns:
        Dictionary of environment variables
    """
    env = os.environ.copy()

    # Read from env_config.yaml if provided
    if env_config_path and env_config_path.exists():
        try:
            import yaml
            with open(env_config_path) as f:
                config = yaml.safe_load(f) or {}
            # Convert all values to strings
            env.update({k: str(v) for k, v in config.items() if v is not None})
        except ImportError:
            print("Warning: pyyaml not installed, skipping env_config.yaml")

    # Set current directory (paths are identical between host and container)
    env['CURRENT_DIR'] = str(Path.cwd().absolute())

    # Set HOME if not already set
    if 'HOME' not in env:
        env['HOME'] = str(Path.home())

    # Set UID, GID, DISPLAY
    env['UID'] = str(os.getuid())
    env['GID'] = str(os.getgid())
    if 'DISPLAY' not in env:
        env['DISPLAY'] = ':0'

    return env


def run_compose_command(
    compose_files: list[str],
    command: list[str],
    env: dict[str, str],
    capture_output: bool = True
) -> subprocess.CompletedProcess:
    """
    Run a docker compose command with the given compose files and environment.

    Args:
        compose_files: List of compose file flags (e.g., ["-f", "docker-compose.yaml"])
        command: Docker compose command to run (e.g., ["build"], ["up", "-d"])
        env: Environment variables dictionary
        capture_output: Whether to capture stdout/stderr (default: True)

    Returns:
        CompletedProcess from subprocess.run
    """
    compose_cmd = ["docker", "compose"] + compose_files + command
    return subprocess.run(
        compose_cmd,
        check=False,
        env=env,
        capture_output=capture_output,
        text=True
    )


def build_image(example_dir: Path, env: dict[str, str]) -> None:
    """
    Build the Docker container using docker compose.

    Args:
        example_dir: Path to the example directory
        env: Environment variables dictionary
    """
    print("Building the Docker container with docker compose...")

    compose_files = get_compose_files(example_dir)
    print(f"Working directory: {env['CURRENT_DIR']}")

    build_args = ["docker", "compose"] + compose_files + ["build"]
    subprocess.run(build_args, check=True, cwd=str(example_dir), env=env)


def down_container(compose_files: list[str], env: dict[str, str]) -> None:
    """
    Stop and remove the container.

    Args:
        compose_files: List of compose file flags
        env: Environment variables dictionary
    """
    result = run_compose_command(compose_files, ["ps", "-q"], env)

    if result.stdout and result.stdout.strip():
        print("Stopping and removing container...")
        run_compose_command(compose_files, ["down"], env, capture_output=False).check_returncode()
    else:
        print("No running container found.")


def force_restart_container(compose_files: list[str], env: dict[str, str]) -> None:
    """
    Ensure the container is running, starting it if necessary.

    Stops and removes any existing container, then starts a fresh one.

    Args:
        compose_files: List of compose file flags
        env: Environment variables dictionary
    """
    result = run_compose_command(compose_files, ["ps", "-q"], env)

    if result.stdout and result.stdout.strip():
        print("Stopping existing container...")
        run_compose_command(compose_files, ["down"], env)

    print("Starting container...")
    run_compose_command(compose_files, ["up", "-d"], env, capture_output=False).check_returncode()


def execute_in_container(
    compose_files: list[str],
    env: dict[str, str],
    service_name: str,
    command: list[str],
    interactive: bool = False
) -> subprocess.CompletedProcess:
    """
    Execute a command in the running container.

    Args:
        compose_files: List of compose file flags
        env: Environment variables dictionary
        service_name: Name of the docker compose service
        command: Command to execute (list of strings)
        interactive: Whether to run interactively (default: False)

    Returns:
        CompletedProcess from subprocess.run
    """
    exec_cmd = ["docker", "compose"] + compose_files + ["exec"]

    if not interactive:
        exec_cmd.append("-T")  # Disable pseudo-TTY for non-interactive

    exec_cmd.extend([service_name] + command)

    return subprocess.run(
        exec_cmd,
        env=env,
        check=False,
        capture_output=not interactive
    )


def install_ncompass(
    compose_files: list[str],
    env: dict[str, str],
    service_name: str,
    ncompass_dir: str
) -> None:
    """
    Install ncompass in the container.

    Args:
        compose_files: List of compose file flags
        env: Environment variables dictionary
        service_name: Service name
        ncompass_dir: Path to ncompass directory
    """
    ncompass_path = Path(ncompass_dir).absolute()
    if not ncompass_path.exists() or not ncompass_path.is_dir():
        print(f"Error: Path '{ncompass_path}' does not exist or is not a directory.")
        sys.exit(1)
    elif ncompass_path.name != "ncompass":
        print(f"Error: Path must end with '/ncompass' (got '{ncompass_path.name}').")
        sys.exit(1)

    install_cmd = f"uv pip install {ncompass_path}"

    print("Installing ncompass...")
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
        print(f"Warning: ncompass installation failed with exit code {result.returncode}")
    else:
        print("ncompass installation complete.")


def run_container(
    example_dir: Path,
    service_name: str,
    ncompass_dir: str,
    auto_exec: bool = True,
    env_config_path: Optional[Path] = None,
    post_install_hook: Optional[Callable[[list[str], dict[str, str], str], None]] = None
) -> None:
    """
    Run the Docker container using docker compose.

    Args:
        example_dir: Path to the example directory
        service_name: Service name (must match docker-compose service name)
        ncompass_dir: Path to ncompass directory
        auto_exec: Whether to automatically exec into the container
        env_config_path: Optional path to env_config.yaml
        post_install_hook: Optional callback for additional installation steps
    """
    print("Running the Docker container with docker compose...")

    compose_files = get_compose_files(example_dir)
    env = get_compose_env(env_config_path)

    print(f"Mounting current directory: {env['CURRENT_DIR']}")

    # Ensure container is running
    force_restart_container(compose_files, env)

    # Run post-install hook if provided (e.g., vllm installation)
    if post_install_hook:
        post_install_hook(compose_files, env, service_name)

    # Install ncompass
    install_ncompass(compose_files, env, service_name, ncompass_dir)

    if auto_exec:
        print(f"Executing interactive shell in container '{service_name}'...")
        execute_in_container(
            compose_files,
            env,
            service_name,
            ["/bin/bash"],
            interactive=True
        )
    else:
        print(f"\nTo connect to the container, run: docker exec -it {service_name} /bin/bash")


def exec_command(
    example_dir: Path,
    service_name: str,
    ncompass_dir: str,
    command: str,
    env_config_path: Optional[Path] = None,
    post_install_hook: Optional[Callable[[list[str], dict[str, str], str], None]] = None
) -> None:
    """
    Execute a command in the running container.

    Args:
        example_dir: Path to the example directory
        service_name: Service name (must match docker-compose service name)
        ncompass_dir: Path to ncompass directory
        command: Command string to execute in bash shell
        env_config_path: Optional path to env_config.yaml
        post_install_hook: Optional callback for additional installation steps
    """
    compose_files = get_compose_files(example_dir)
    env = get_compose_env(env_config_path)

    # Ensure container is running
    force_restart_container(compose_files, env)

    # Run post-install hook if provided
    if post_install_hook:
        post_install_hook(compose_files, env, service_name)

    # Install ncompass
    install_ncompass(compose_files, env, service_name, ncompass_dir)

    # Execute the command in bash
    print(f"Executing command in container '{service_name}': {command}")
    result = execute_in_container(
        compose_files,
        env,
        service_name,
        ["/bin/bash", "-c", command],
        interactive=False
    )

    # Print output
    if result.stdout:
        print(result.stdout, end='')
    if result.stderr:
        print(result.stderr, end='', file=sys.stderr)

    # Exit with the same code as the command
    if result.returncode != 0:
        sys.exit(result.returncode)


def create_parser(service_name: str) -> argparse.ArgumentParser:
    """
    Create argument parser for nc_pkg.py.

    Args:
        service_name: Default service name for this example

    Returns:
        Configured ArgumentParser
    """
    parser = argparse.ArgumentParser(
        description='Docker compose management for nCompass examples.'
    )
    parser.add_argument(
        '--build', action='store_true',
        help='Build the Docker image'
    )
    parser.add_argument(
        '--run', action='store_true',
        help='Run the Docker container'
    )
    parser.add_argument(
        '--down', action='store_true',
        help='Stop and remove the Docker container'
    )
    parser.add_argument(
        '--exec', type=str, metavar='<cmd>',
        help='Execute a command in a bash shell inside the container'
    )
    parser.add_argument(
        '--tag', type=str, default='latest',
        help='Tag for the Docker container (default: latest)'
    )
    parser.add_argument(
        '--no-exec', action='store_true',
        help='Do not automatically exec into the container'
    )
    parser.add_argument(
        '--ncompass-dir', type=str,
        help='Path to the ncompass directory (required for --run and --exec)'
    )

    return parser


def main(
    service_name: str,
    env_config_path: Optional[Path] = None,
    post_install_hook: Optional[Callable[[list[str], dict[str, str], str], None]] = None,
    extra_args_handler: Optional[Callable[[argparse.ArgumentParser], None]] = None,
    args: Optional[list[str]] = None
) -> None:
    """
    Main entry point for nc_pkg.py scripts.

    Args:
        service_name: Default service name for this example
        env_config_path: Optional path to env_config.yaml
        post_install_hook: Optional callback for additional installation steps
        extra_args_handler: Optional callback to add extra arguments to parser
        args: Optional argument list (defaults to sys.argv)
    """
    parser = create_parser(service_name)

    # Allow examples to add their own arguments
    if extra_args_handler:
        extra_args_handler(parser)

    parsed_args = parser.parse_args(args)

    # Determine example directory (where this script is called from)
    example_dir = Path.cwd()

    env = get_compose_env(env_config_path)

    # Set COMPOSE_PROJECT_NAME for session isolation when --tag is provided.
    # Set on os.environ so downstream functions (run_container, exec_command)
    # that call get_compose_env() also pick it up.
    if parsed_args.tag != 'latest':
        os.environ['COMPOSE_PROJECT_NAME'] = \
                f"{example_dir.name}-{parsed_args.tag}"
        env['COMPOSE_PROJECT_NAME'] = os.environ['COMPOSE_PROJECT_NAME']

    if parsed_args.build:
        build_image(example_dir, env)

    if parsed_args.down:
        compose_files = get_compose_files(example_dir)
        down_container(compose_files, env)

    if parsed_args.run or getattr(parsed_args, 'exec', None) is not None:
        if not parsed_args.ncompass_dir:
            print("Error: --ncompass-dir is required when using --run or --exec")
            sys.exit(1)

        if getattr(parsed_args, 'exec', None) is not None:
            exec_command(
                example_dir=example_dir,
                service_name=service_name,
                ncompass_dir=parsed_args.ncompass_dir,
                command=parsed_args.exec,
                env_config_path=env_config_path,
                post_install_hook=post_install_hook
            )

        if parsed_args.run:
            run_container(
                example_dir=example_dir,
                service_name=service_name,
                ncompass_dir=parsed_args.ncompass_dir,
                auto_exec=not parsed_args.no_exec,
                env_config_path=env_config_path,
                post_install_hook=post_install_hook
            )
