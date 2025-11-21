# nCompass SDK Examples

Ready-to-run example scripts demonstrating how to use the nCompass SDK for profiling and tracing AI inference workloads on GPUs and other accelerators.

## 🚀 Getting Started

The best way to use nCompass is through our **[VSCode extension](https://marketplace.visualstudio.com/items?itemName=nCompassTech.ncprof-vscode)**, which provides seamless integration between your codebase and performance traces.

📖 **New to nCompass?** Check out our [quick start guide](https://docs.ncompass.tech/ncprof/quick-start) to get up and running in minutes.

## 📚 Available Examples

Each example is self-contained and demonstrates different profiling workflows:

- **[Basic Example - TorchRecord Profiling](basic_example/)** — Get started with PyTorch profiling using automatic tracepoint injection
- **[Nsight Systems Example](nsys_example/)** — Profile GPU kernels with Nsight Systems integration and convert nsys traces to chrome traces to view in the VSCode IDE.
- **[Profiling remotely on Modal](profiling_session/)** — Run profiling sessions on remote compute infrastructure

> 💡 **Tip**: Each example includes a detailed README with step-by-step instructions and explanations.

## 🎥 Tutorial Videos

Learn how to use nCompass with our video tutorials:

- **[Installation Guide](https://www.loom.com/share/871ac68417c14100b6e6a29df699e857)** — Set up nCompass and the VSCode extension
- **[Feature Tutorial - Automatic TorchRecord Context Injection](https://www.loom.com/share/2604f25cc97e468db0e209e7ef5f8949)** — See how zero-instrumentation profiling works
- **[Feature Tutorial - Running remotely on Modal](https://www.loom.com/share/6c5f9fc56600452b84dd0739e8f251f9)** — How to integrate with Modal and run profiling remotely

## ⚙️ Running Examples

### Prerequisites

Before running any example, ensure you have:

1. ✅ Installed the [VSCode extension](https://marketplace.visualstudio.com/items?itemName=nCompassTech.ncprof-vscode)
2. ✅ The `ncprof` backend running
3. ✅ Python 3.11+ installed
4. ✅ `Pydantic>=2.0` installed

Each example includes its own README with specific setup instructions and requirements. Navigate to the example directory and follow the instructions there.

## 💬 Support

Need help with examples or have questions?

- 📚 **[Documentation](https://docs.ncompass.tech)** — Comprehensive guides and API reference
- 💬 **[Community Forum](https://community.ncompass.tech)** — Get help from the community
- 🐛 **[GitHub Issues](https://github.com/ncompass-tech/ncompass/issues)** — Report bugs or request features
