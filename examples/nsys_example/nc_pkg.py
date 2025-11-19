import os
import sys
import argparse
import subprocess

from pathlib import Path

def resolve_host_path(path: Path) -> Path:
    """
    Resolve a path using HOST_BASE logic.
    
    Replaces '/workspace' in the path with HOST_BASE if HOST_BASE is set.
    
    Args:
        path: Path to resolve
        
    Returns:
        Resolved absolute path
    """
    host_dir = Path(os.getenv('HOST_BASE', '/workspace'))
    resolved = Path(str(path.absolute()).replace('/workspace', str(host_dir)))
    return resolved

def get_compose_files() -> list[str]:
    """
    Get the list of docker compose files to use based on what exists.
    
    Returns:
        List of compose file flags (e.g., ["-f", "docker-compose.yaml", "-f", "docker-compose.ncompass.yaml"])
    """
    compose_files = ["-f", "docker-compose.yaml"]
    
    ncompass_dir = Path("./ncompass")
    if ncompass_dir.exists():
        compose_files.extend(["-f", "docker-compose.ncompass.yaml"])
        print("Using docker-compose.ncompass.yaml for ncompass mount")
    
    return compose_files

def get_compose_env() -> dict[str, str]:
    """
    Get environment variables needed for docker compose commands.
    
    Sets up CURRENT_DIR and NCOMPASS_DIR using HOST_BASE resolution logic.
    
    Returns:
        Dictionary of environment variables
    """
    env = os.environ.copy()
    
    # Resolve current directory using HOST_BASE logic
    current_dir = resolve_host_path(Path.cwd())
    env['CURRENT_DIR'] = str(current_dir.absolute())
    
    # Set UID, GID, DISPLAY
    env['UID'] = str(os.getuid())
    env['GID'] = str(os.getgid())
    env['DISPLAY'] = os.environ.get('DISPLAY', ':0')
    
    # Check if ncompass directory exists and set NCOMPASS_DIR
    ncompass_dir = Path("./ncompass")
    if ncompass_dir.exists():
        ncompass_resolved = resolve_host_path(ncompass_dir)
        env['NCOMPASS_DIR'] = str(ncompass_resolved.absolute())
    
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

def ensure_container_running(compose_files: list[str], env: dict[str, str]) -> None:
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
    ensure_container_running(compose_files, env)

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
    
    # Ensure container is running
    ensure_container_running(compose_files, env)
    
    # Execute the command in bash
    print(f"Executing command in container '{name}': {command}")
    result = execute_in_container(
        compose_files,
        env,
        name,
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
