# nCompass SDK Examples

Ready-to-run example scripts demonstrating how to use the nCompass SDK for profiling and tracing AI inference workloads on GPUs and other accelerators.

## 🚀 Getting Started

The best way to use nCompass is through our **[VSCode extension](https://marketplace.visualstudio.com/items?itemName=nCompassTech.ncprof-vscode)**, which provides seamless integration between your codebase and performance traces.

📖 **New to nCompass?** Check out our [quick start guide](https://docs.ncompass.tech/ncprof/quick-start) to get up and running in minutes.

## 📚 Available Examples

Each example is self-contained and demonstrates different profiling workflows:

- **[vLLM Profiling Example](vllm_example/)** — Profile vLLM using .pth-based auto-initialization with NCU, Nsys, and Torch profilers
- **[Running remotely on Modal](modal_example/)** — Run profiling sessions on Modal cloud infrastructure

> 💡 **Tip**: Each example includes a detailed README with step-by-step instructions and explanations.

## Unified Docker Environment

All examples share a **[unified Docker setup](docker/)** that includes all profiling tools:

| Tool | Version |
|------|---------|
| CUDA | 13.0.0 (configurable) |
| Python | 3.10 |
| Nsight Systems (nsys) | 2025.2.1 |
| Nsight Compute (ncu) | Matches CUDA version |
| PyTorch | 2.0+ |

### Building the Docker Image

```bash
cd docker/
docker compose build

# Or build with a specific CUDA version
docker compose build --build-arg CUDA_VERSION=12.9.1
```

### Running an Example with Docker

```bash
cd vllm_example/
python nc_pkg.py --build
python nc_pkg.py --run --ncompass-dir /path/to/ncompass
```

See [docker/README.md](docker/README.md) for detailed Docker documentation.

## Tutorial Videos

Learn how to use nCompass with our video tutorials:

- **[Installation Guide](https://www.loom.com/share/871ac68417c14100b6e6a29df699e857)** — Set up nCompass and the VSCode extension
- **[Feature Tutorial - Automatic TorchRecord Context Injection](https://www.loom.com/share/2604f25cc97e468db0e209e7ef5f8949)** — See how zero-instrumentation profiling works
- **[Feature Tutorial - Running remotely on Modal](https://www.loom.com/share/6c5f9fc56600452b84dd0739e8f251f9)** — How to integrate with Modal and run profiling remotely

## Running Examples

### Prerequisites

Before running any example, ensure you have:

1. Docker installed (recommended) OR local installation of profiling tools
2. NVIDIA GPU with appropriate drivers
3. [VSCode extension](https://marketplace.visualstudio.com/items?itemName=nCompassTech.ncprof-vscode) (for marker injection)

Each example includes its own README with specific setup instructions.

## 💬 Support

Need help with examples or have questions?

- 📚 **[Documentation](https://docs.ncompass.tech)** — Comprehensive guides and API reference
- 💬 **[Community Forum](https://community.ncompass.tech)** — Get help from the community
- 🐛 **[GitHub Issues](https://github.com/ncompass-tech/ncompass/issues)** — Report bugs or request features
