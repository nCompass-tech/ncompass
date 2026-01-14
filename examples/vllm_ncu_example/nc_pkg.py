#!/usr/bin/env python3
"""
Docker helper script for vllm_ncu_example.

Provides commands to build, run, and manage the Docker container for
vLLM profiling with NCU and Nsys.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

import yaml


def get_compose_files() -> list[str]:
    """Get the list of docker compose files to use."""
    return ["-f", "docker-compose.yaml"]


def get_compose_env() -> dict[str, str]:
    """
    Get environment variables needed for docker compose commands.

    Reads from env_config.yaml and sets up required environment variables.
    """
    config_path = Path(__file__).parent / "env_config.yaml"

    if config_path.exists():
        with open(config_path) as f:
            env = yaml.safe_load(f) or {}
        env = {k: str(v) for k, v in env.items() if v is not None}
    else:
        print(f"Warning: {config_path} not found. Copy env_config.yaml.example to env_config.yaml and fill in your values.")
        env = {}

    # Set current directory
    env['CURRENT_DIR'] = str(Path.cwd().absolute())

    # Set HOME, UID, GID, DISPLAY
    env['HOME'] = str(Path.home())
    env['UID'] = str(os.getuid())
    env['GID'] = str(os.getgid())
    env['DISPLAY'] = os.environ.get('DISPLAY', ':0')

    return env


def run_compose_command(
    compose_files: list[str],
    command: list[str],
    env: dict[str, str],
    capture_output: bool = True
) -> subprocess.CompletedProcess:
    """Run a docker compose command."""
    compose_cmd = ["docker", "compose"] + compose_files + command
    return subprocess.run(
        compose_cmd,
        check=False,
        env=env,
        capture_output=capture_output,
        text=True
    )


def build_image():
    """Build the Docker container using docker compose."""
    print("Building the Docker container...")

    compose_files = get_compose_files()
    env = get_compose_env()

    print(f"Mounting current directory: {env['CURRENT_DIR']}")

    build_args = ["docker", "compose"] + compose_files + ["build"]
    subprocess.run(build_args, check=True, cwd=".", env=env)


def down_container(compose_files: list[str], env: dict[str, str]) -> None:
    """Stop and remove the container."""
    result = run_compose_command(compose_files, ["ps", "-q"], env)

    if result.stdout and result.stdout.strip():
        print("Stopping and removing container...")
        run_compose_command(compose_files, ["down"], env, capture_output=False).check_returncode()
    else:
        print("No running container found.")


def force_restart_container(compose_files: list[str], env: dict[str, str]) -> None:
    """Ensure the container is running, starting it if necessary."""
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
    """Execute a command in the running container."""
    exec_cmd = ["docker", "compose"] + compose_files + ["exec"]

    if not interactive:
        exec_cmd.append("-T")

    exec_cmd.extend([service_name] + command)

    return subprocess.run(
        exec_cmd,
        env=env,
        check=False,
        capture_output=not interactive
    )


def install_vllm(compose_files: list[str], env: dict[str, str], name: str, vllm_src: str) -> None:
    """
    Install vllm from source or wheel.

    Args:
        compose_files: List of compose file flags
        env: Environment variables dictionary
        name: Service name
        vllm_src: Path to vllm source directory or wheel file
    """
    vllm_path = Path(vllm_src).absolute()

    if vllm_path.is_dir():
        # Check for wheel file in the directory
        wheel_files = list(vllm_path.glob("vllm*.whl"))
        if wheel_files:
            wheel_file = wheel_files[0]
            install_cmd = f"VLLM_PRECOMPILED_WHEEL_LOCATION={wheel_file} uv pip install -e {vllm_path}"
            print(f"Installing vllm with precompiled wheel: {wheel_file}...")
        else:
            install_cmd = f"uv pip install -e {vllm_path}"
            print(f"Installing vllm from source: {vllm_path}...")
    elif vllm_path.suffix == ".whl":
        install_cmd = f"uv pip install {vllm_path}"
        print(f"Installing vllm from wheel: {vllm_path}...")
    else:
        print(f"Error: {vllm_src} is not a valid vllm source directory or wheel file")
        sys.exit(1)

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
        print(f"Warning: vllm installation failed with exit code {result.returncode}")
    else:
        print("vllm installation complete.")


def install_ncompass(compose_files: list[str], env: dict[str, str], name: str, ncompass_dir: str) -> None:
    """Install ncompass package."""
    ncompass_path = Path(ncompass_dir).absolute()
    if not ncompass_path.exists() or not ncompass_path.is_dir():
        print(f"Error: Path '{ncompass_path}' does not exist or is not a directory.")
        sys.exit(1)

    install_cmd = f"uv pip install {ncompass_path}"

    print(f"Installing ncompass from: {ncompass_path}...")
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


def run_container(name: str, ncompass_dir: str, vllm_src: str | None, auto_exec: bool = True):
    """
    Run the Docker container using docker compose.

    Args:
        name: Service name (must match docker-compose service name)
        ncompass_dir: Path to ncompass directory
        vllm_src: Optional path to vllm source directory or wheel
        auto_exec: Whether to automatically exec into the container
    """
    print("Running the Docker container with docker compose...")

    compose_files = get_compose_files()
    env = get_compose_env()

    print(f"Mounting current directory: {env['CURRENT_DIR']}")

    # Ensure container is running
    force_restart_container(compose_files, env)

    # Install vllm if source provided
    if vllm_src:
        install_vllm(compose_files, env, name, vllm_src)

    # Install ncompass
    install_ncompass(compose_files, env, name, ncompass_dir)

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


def exec_command(name: str, ncompass_dir: str, command: str):
    """
    Execute a command in the running container.

    Args:
        name: Service name (must match docker-compose service name)
        ncompass_dir: Path to ncompass directory
        command: Command string to execute in bash shell
    """
    compose_files = get_compose_files()
    env = get_compose_env()

    # Ensure container is running
    force_restart_container(compose_files, env)

    # Install ncompass
    install_ncompass(compose_files, env, name, ncompass_dir)

    # Execute the command in bash
    print(f"Executing command in container '{name}': {command}")
    result = execute_in_container(
        compose_files,
        env,
        name,
        ["/bin/bash", "-c", command],
        interactive=False
    )

    if result.stdout:
        print(result.stdout, end='')
    if result.stderr:
        print(result.stderr, end='', file=sys.stderr)

    if result.returncode != 0:
        sys.exit(result.returncode)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Docker helper for vLLM NCU/Nsys profiling example.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build the Docker image
  python nc_pkg.py --build

  # Build and run (interactive shell)
  python nc_pkg.py --build --run --ncompass-dir ../../..

  # Run with vllm installation
  python nc_pkg.py --run --ncompass-dir ../../.. --vllm-src /path/to/vllm

  # Execute a profiling command
  python nc_pkg.py --exec "python main.py --ncu" --ncompass-dir ../../..

  # Stop the container
  python nc_pkg.py --down
        """
    )
    parser.add_argument('--build', action='store_true', help='Build the Docker image')
    parser.add_argument('--run', action='store_true', help='Run the Docker container')
    parser.add_argument('--down', action='store_true', help='Stop and remove the Docker container')
    parser.add_argument(
        '--exec', type=str, metavar='<cmd>',
        help='Execute a command in a bash shell inside the container'
    )
    parser.add_argument(
        '--name', type=str, default='vllm_ncu_example',
        help='Name for the Docker container (default: vllm_ncu_example)'
    )
    parser.add_argument(
        '--no-exec', action='store_true',
        help='Do not automatically exec into the container after --run'
    )
    parser.add_argument(
        '--ncompass-dir', type=str,
        help='Path to the ncompass directory (required for --run and --exec)'
    )
    parser.add_argument(
        '--vllm-src', type=str,
        help='Path to vllm source directory or wheel file (optional)'
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.build:
        build_image()

    if args.down:
        compose_files = get_compose_files()
        env = get_compose_env()
        down_container(compose_files, env)

    if args.run or args.exec is not None:
        if not args.ncompass_dir:
            print("Error: --ncompass-dir is required when using --run or --exec")
            sys.exit(1)

        if args.exec is not None:
            exec_command(name=args.name, ncompass_dir=args.ncompass_dir, command=args.exec)

        if args.run:
            run_container(
                name=args.name,
                ncompass_dir=args.ncompass_dir,
                vllm_src=args.vllm_src,
                auto_exec=not args.no_exec
            )


if __name__ == '__main__':
    main()
