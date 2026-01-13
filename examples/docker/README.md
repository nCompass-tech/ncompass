# Unified Docker Setup for nCompass Examples

This directory contains a unified Docker configuration that all nCompass examples share. The unified image includes all profiling tools, allowing any example to use PyTorch profiler, Nsight Systems (nsys), or Nsight Compute (ncu).

## Quick Start

### Building the Image

From any example directory:

```bash
python nc_pkg.py --build
```

Or directly from this directory:

```bash
docker compose build
```

### Running an Example

```bash
cd ../basic_example
python nc_pkg.py --run --ncompass-dir /path/to/ncompass
```

## Included Tools

The unified Docker image includes:

| Tool | Version | Description |
|------|---------|-------------|
| Python | 3.10 | Python interpreter |
| PyTorch | 2.0+ | Deep learning framework with profiler |
| CUDA | 13.0.0 (configurable) | GPU toolkit |
| Nsight Systems (nsys) | 2025.2.1 | Timeline-based GPU profiling |
| Nsight Compute (ncu) | Matches CUDA version | Kernel-level GPU profiling |
| CUDA-GDB | Matches CUDA version | GPU debugger |
| Rust/Cargo | stable | For trace_converter profiling |
| flamegraph | latest | Performance visualization |
| perf | system | Linux performance tools |
| uv | latest | Fast Python package manager |

## Directory Structure

```
docker/
├── Dockerfile               # Unified Dockerfile with all tools
├── docker-compose.base.yaml # Base compose config (extended by examples)
├── nc_pkg_lib.py           # Shared Docker management library
├── requirements/
│   ├── base.txt            # Core Python dependencies
│   └── dev.txt             # Development/profiling dependencies
└── README.md               # This file
```

## How Examples Use This

Each example has a `docker` symlink pointing to this directory, allowing examples to be portable and not depend on relative paths above them.

### Directory Structure in Each Example

```
my_example/
├── docker -> ../docker    # Symlink to docker directory
├── docker-compose.yaml    # Extends ./docker/docker-compose.base.yaml
├── nc_pkg.py              # Uses ./docker symlink or --docker-dir
└── ...
```

### docker-compose.yaml

Each example's `docker-compose.yaml` extends the base via the local symlink:

```yaml
services:
  my_example:
    extends:
      file: ./docker/docker-compose.base.yaml
      service: ncompass
    container_name: my_example
```

### nc_pkg.py

Each example's `nc_pkg.py` resolves the docker directory dynamically:

```python
#!/usr/bin/env python3
"""Docker management script for my_example."""
import argparse
import sys
from pathlib import Path


def find_docker_dir() -> Path:
    """Find docker directory via --docker-dir arg or local symlink."""
    # Use argparse to cleanly extract --docker-dir before importing nc_pkg_lib
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--docker-dir', type=Path)
    args, _ = parser.parse_known_args()

    if args.docker_dir:
        return args.docker_dir

    # Check local symlink
    local = Path(__file__).parent / 'docker'
    if local.exists():
        return local.resolve()

    print("Error: Docker directory not found.")
    print("Provide via: --docker-dir PATH or ./docker symlink")
    sys.exit(1)


# Add docker directory to Python path
docker_dir = find_docker_dir()
sys.path.insert(0, str(docker_dir))

from nc_pkg_lib import main

SERVICE_NAME = "my_example"

if __name__ == '__main__':
    main(service_name=SERVICE_NAME)
```

## Docker Directory Resolution

The docker directory is found using this priority order:

1. **`--docker-dir` argument** (highest priority): Explicitly specify the path
2. **`./docker` symlink** (default): Local symlink in the example directory

### Using the Default Symlink

Each example ships with a symlink `./docker -> ../docker`. Simply run:

```bash
python nc_pkg.py --build
```

### Moving the Docker Directory

To use a docker directory at a different location:

```bash
# Option 1: Update the symlink
rm ./docker
ln -s /opt/my-custom-docker ./docker
python nc_pkg.py --build

# Option 2: Override with explicit argument
python nc_pkg.py --build --docker-dir /opt/my-custom-docker
```

### Creating Symlinks for New Examples

When creating a new example, add the docker symlink:

```bash
cd examples/my_new_example
ln -s ../docker ./docker
```

## nc_pkg.py Usage

All examples support these common options:

```bash
# Build the Docker image
python nc_pkg.py --build

# Run the container and enter interactive shell
python nc_pkg.py --run --ncompass-dir /path/to/ncompass

# Run without entering shell
python nc_pkg.py --run --ncompass-dir /path/to/ncompass --no-exec

# Execute a command in the container
python nc_pkg.py --exec "python main.py" --ncompass-dir /path/to/ncompass

# Stop and remove the container
python nc_pkg.py --down

# Override docker directory location
python nc_pkg.py --build --docker-dir /path/to/docker
```

### vllm_example Additional Options

The vllm example supports an additional `--wheel` option:

```bash
python nc_pkg.py --run --ncompass-dir /path/to/ncompass --wheel vllm-0.13.0+cu130-cp38-abi3-manylinux_2_31_x86_64.whl
```

## Profiling in the Container

Once inside the container, you can use any profiler:

### PyTorch Profiler

```python
import torch
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    # Your code here
    pass
prof.export_chrome_trace("trace.json")
```

### Nsight Systems

```bash
nsys profile python my_script.py
nsys profile --trace=cuda,nvtx python my_script.py
```

### Nsight Compute

```bash
ncu python my_script.py
ncu --set full python my_script.py
```

## Environment Variables

The container sets these environment variables:

| Variable | Description |
|----------|-------------|
| `PYTHONPATH` | Set to `/workspace` |
| `LD_LIBRARY_PATH` | Includes CUDA libraries |
| `HOME` | Mounted from host |
| `HF_TOKEN` | HuggingFace token (optional) |
| `VLLM_TORCH_PROFILER_DIR` | vLLM profiler output (optional) |

## Base Image

The unified image is based on:

```
nvidia/cuda:13.0.0-devel-ubuntu24.04
```

This provides CUDA 13.0.0 with development tools on Ubuntu 24.04.

### Customizing CUDA Version

You can build with a different CUDA version using the `CUDA_VERSION` build arg:

```bash
# Build with CUDA 12.9.1
docker compose build --build-arg CUDA_VERSION=12.9.1

# Build with CUDA 12.8.0
docker compose build --build-arg CUDA_VERSION=12.8.0
```

The Dockerfile automatically handles version-specific configurations (e.g., library symlinks) based on the CUDA version you specify.

## Customization

### Adding Python Dependencies

Edit `requirements/base.txt` for core dependencies or `requirements/dev.txt` for development tools, then rebuild.

### Example-Specific Customization

If an example needs custom setup:

1. Create a custom `post_install_hook` function
2. Pass it to `main()` in your `nc_pkg.py`

See `vllm_example/nc_pkg.py` for an example.

## Troubleshooting

### Permission Issues

The container creates a user (`ncuser`) matching your host UID/GID to avoid permission issues with mounted volumes.

### GPU Access

Ensure:
- NVIDIA drivers are installed on host
- Docker has GPU runtime configured
- The container runs with `--privileged` (set in compose)

### nsys/ncu Not Found

If profiling tools aren't found, verify the image was built correctly:

```bash
docker run --rm ncompass-dev:latest nsys --version
docker run --rm ncompass-dev:latest ncu --version
```
