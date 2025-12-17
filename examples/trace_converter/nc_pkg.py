import os
import sys
import argparse
import subprocess

import yaml
from pathlib import Path

def get_compose_files() -> list[str]:
    """
    Get the list of docker compose files to use based on what exists.
    
    Returns:
        List of compose file flags (e.g., ["-f", "docker-compose.yaml", "-f", "docker-compose.ncompass.yaml"])
    """
    compose_files = ["-f", "docker-compose.yaml"]
    
    
    return compose_files

def get_compose_env() -> dict[str, str]:
    """
    Get environment variables needed for docker compose commands.
    
    Reads from env_config.yaml and sets up CURRENT_DIR, HOME, UID, GID, and DISPLAY.
    
    Returns:
        Dictionary of environment variables
    """
    # Read environment config from yaml
    config_path = Path(__file__).parent / "env_config.yaml"
    
    if config_path.exists():
        with open(config_path) as f:
            env = yaml.safe_load(f) or {}
        # Convert all values to strings
        env = {k: str(v) for k, v in env.items() if v is not None}
    else:
        print(f"Warning: {config_path} not found. Copy env_config.yaml.example to env_config.yaml and fill in your values.")
        env = {}
    
    # Set current directory (paths are now identical between host and container)
    env['CURRENT_DIR'] = str(Path.cwd().absolute())
    
    # Set HOME
    env['HOME'] = str(Path.home())
    
    # Set UID, GID, DISPLAY
    env['UID'] = str(os.getuid())
    env['GID'] = str(os.getgid())
    env['DISPLAY'] = os.environ.get('DISPLAY', ':0')
    
    return env

def run_compose_command(compose_files: list[str], 
                        command: list[str], 
                        env: dict[str, str], 
                        capture_output: bool = True) -> subprocess.CompletedProcess:
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
    return subprocess.run(compose_cmd, 
                          check=False, 
                          env=env, 
                          capture_output=capture_output, 
                          text=True)

def build_image(tag: str, name: str, installdir: str):
    """Build the Docker container using docker compose."""
    print("Building the Docker container with docker compose...")
    
    compose_files = get_compose_files()
    env = get_compose_env()
    
    print(f"Mounting current directory: {env['CURRENT_DIR']}")
    
    build_args = ["docker", "compose"] + compose_files + ["build"]
    
    subprocess.run(build_args, check=True, cwd=".", env=env)

def down_container(compose_files: list[str], env: dict[str, str]) -> None:
    """
    Stop and remove the container.
    
    Args:
        compose_files: List of compose file flags
        env: Environment variables dictionary
    """
    # Check if container is running
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
    # Check if container is already running
    result = run_compose_command(compose_files, ["ps", "-q"], env)
    
    if result.stdout and result.stdout.strip():
        print("Stopping existing container...")
        run_compose_command(compose_files, ["down"], env)
    
    # Start the container
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

def install_ncompass(compose_files: list[str], env: dict[str, str], name: str) -> None:
    """
    Install ncompass
    
    Args:
        compose_files: List of compose file flags
        env: Environment variables dictionary
        name: Service name
    """
    install_cmd = "uv pip install ../../"
    
    print("Installing ncompass ...")
    result = execute_in_container(
        compose_files,
        env,
        name,
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

def run_container(tag: str, name: str, auto_exec: bool = True):
    """
    Run the Docker container using docker compose.
    
    Args:
        tag: Docker image tag
        name: Service name (must match docker-compose service name)
        auto_exec: Whether to automatically exec into the container
    """
    print("Running the Docker container with docker compose...")
    
    compose_files = get_compose_files()
    env = get_compose_env()
    
    print(f"Mounting current directory: {env['CURRENT_DIR']}")
    
    # Ensure container is running
    force_restart_container(compose_files, env)
    
    # Install ncompass in editable mode
    install_ncompass(compose_files, env, name)

    if auto_exec:
        print(f"Executing interactive shell in container '{name}'...")
        execute_in_container(
            compose_files,
            env,
            name,
            ["/bin/bash"],
            interactive=True
        )
    else:
        print(f"\nTo connect to the container, run: docker exec -it {name} /bin/bash")

def exec_command(tag: str, name: str, command: str):
    """
    Execute a command in the running container.
    
    Args:
        tag: Docker image tag (unused, kept for consistency)
        name: Service name (must match docker-compose service name)
        command: Command string to execute in bash shell
    """
    compose_files = get_compose_files()
    env = get_compose_env()
    
    # Execute the command in bash
    print(f"Executing command in container '{name}': {command}")
    result = execute_in_container(
        compose_files,
        env,
        name,
        ["/bin/bash", "-c", command],
        interactive=True
    )
    
    # Print output
    if result.stdout:
        print(result.stdout, end='')
    if result.stderr:
        print(result.stderr, end='', file=sys.stderr)
    
    # Exit with the same code as the command
    if result.returncode != 0:
        sys.exit(result.returncode)

def parse_args():
    parser = argparse.ArgumentParser(description='Process build and run options.')
    parser.add_argument('--build', action='store_true', help='Build the Docker image')
    parser.add_argument('--run', action='store_true', help='Run the Docker container')
    parser.add_argument('--down', action='store_true', help='Stop and remove the Docker container')
    parser.add_argument(
        '--exec', type=str, metavar='<cmd>',
        help='Execute a command in a bash shell inside the container'
    )
    parser.add_argument(
        '--tag', type=str, default='0.0.1',
        help='Tag for the Docker container (default: 0.0.1)'
    )
    parser.add_argument(
        '--name', type=str, default='nsys_example',
        help='Name for the Docker container (default: nsys_example)'
    )
    parser.add_argument(
        '--no-exec', action='store_true',
        help='Do not automatically exec into the container'
    )
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    installdir = os.path.abspath(".")
    
    if args.build:
        build_image(
            tag=args.tag,
            name=args.name,
            installdir=installdir
        )
    
    if args.down:
        compose_files = get_compose_files()
        env = get_compose_env()
        down_container(compose_files, env)
    
    if args.exec is not None:
        exec_command(tag=args.tag, name=args.name, command=args.exec)
    
    if args.run:
        run_container(tag=args.tag, name=args.name, auto_exec=not args.no_exec)

if __name__ == '__main__':
    main()
