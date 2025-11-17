import os
import argparse
import subprocess

from pathlib import Path

def build_image(tag: str, name: str, installdir: str):
    """Build the Docker container with the specified tag."""
    print("Building the Docker container...")
    
    build_args = [
        "docker", "build",
        "-t", f"nc_{name}:{tag}",
        "."
    ]
    
    subprocess.run(build_args, check=True, cwd=".")

def run_container(tag:str, name:str, auto_exec=True):
    print("Running the Docker container...")
    container_name = f"nc_{name}:{tag}"
    
    # Check if the container is already running and stop it if so
    result = subprocess.run(
        ["docker", "ps", "-a", "--filter", f"name={name}", "--format", "{{.Names}}"],
        capture_output=True,
        text=True,
        check=False
    )
    
    if name in result.stdout:
        print(f"Stopping existing container '{name}'...")
        subprocess.run(["docker", "stop", name], check=False)
        subprocess.run(["docker", "rm", name], check=False)
    
    # Mount current directory to /workspace
    host_dir = Path(os.getenv('HOST_BASE', '/workspace'))
    current_dir = Path(str(Path.cwd()).replace('/workspace', str(host_dir)))
    print(f"Mounting current directory: {current_dir}")
    
    run_cmd = [
        "docker", "run",
        "--detach",
        "--privileged",
        "--network=host",
        "--ipc=host",
        "--gpus", "all",
        "-v", f"{current_dir}:/workspace",
        "-v", f"{current_dir}/.cache:/root/.cache",
        "--name", name,
        "-e", f"HOST_UID={os.getuid()}",
        "-e", f"HOST_GID={os.getgid()}",
        "-e", f"DISPLAY={os.environ.get('DISPLAY', ':0')}",
        container_name,
        "sleep", "infinity"
    ]

    # Run the container using subprocess
    subprocess.run(run_cmd, check=True)

    if auto_exec:
        print(f"Executing interactive shell in container '{name}'...")
        subprocess.run(["docker", "exec", "-it", name, "/bin/bash"])
    else:
        print(f"\nTo connect to the container, run: docker exec -it {name} /bin/bash")

def parse_args():
    parser = argparse.ArgumentParser(description='Process build and run options.')
    parser.add_argument('--build', action='store_true', help='Build the Docker image')
    parser.add_argument('--run', action='store_true', help='Run the Docker container')
    parser.add_argument(
        '--tag', type=str, default='0.0.1',
        help='Tag for the Docker container (default: 0.0.1)'
    )
    parser.add_argument(
        '--name', type=str, default='basic_example',
        help='Name for the Docker container (default: basic_example)'
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

    if args.run:
        run_container(tag=args.tag, name=args.name, auto_exec=not args.no_exec)

if __name__ == '__main__':
    main()
