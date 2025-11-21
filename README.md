# nCompass Python SDK

[![PyPI](https://img.shields.io/pypi/v/ncompass.svg)](https://pypi.org/project/ncompass/)
[![Downloads](https://static.pepy.tech/badge/ncompass)](https://pepy.tech/project/ncompass)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

The Python SDK powering our Performance Optimization IDE—bringing seamless profiling and performance analysis directly into your development workflow.

Built by [nCompass Technologies](https://ncompass.tech).

## What are we building?

We're building a **Performance Optimization IDE** that improves developer productivity by 100x when profiling and analyzing performance of GPU and other accelerator systems. Our IDE consists of two integrated components:

### 🎯 [VSCode Extension](https://marketplace.visualstudio.com/items?itemName=nCompassTech.ncprof-vscode)

Unify your profiling workflow with seamless integration between traces and codebases:

- **No more context switching** — profile, analyze, and optimize all in one place
- **Zero-copy workflow** — visualize traces directly in your editor without transferring files between machines
- **Code-to-trace navigation** — jump seamlessly between your codebase and performance traces
- **AI-powered insights** — get intelligent suggestions for performance improvements and bottleneck identification

### ⚙️ **SDK (this repo)**

The Python SDK that powers the extension with powerful automation features:

- **Zero-instrumentation profiling** — AST-level code injection means you never need to manually add profiling statements
- **Universal trace conversion** — convert traces from nsys and other formats to Chrome traces for integrated visualization
- **Extensible architecture** — built for customization and extension (contributions welcome!)

## Installation

Install via pip:

```bash
pip install ncompass
```

> ⚠️ **Troubleshooting**: If you run into issues with `ncompasslib` or `pydantic`, ensure that:
> 
> 1. You are running Python 3.11
> 2. You have `Pydantic>=2.0` installed

## Examples

Refer to our [open source GitHub repo](https://github.com/nCompass-tech/ncompass/tree/main/examples) for examples. Our examples are built to work together with the VSCode extension. For instance, with adding tracepoints to the code, you can add/remove tracepoints using the extension and then run profiling using our examples. 

- **[Basic TorchProfile Example](examples/basic_example/)**
- **[Nsight Systems Examples](examples/nsys_example/)**
- **[Running remotely on Modal](examples/modal_example/)**

## Online Resources

- 🌐 **Website**: [ncompass.tech](https://ncompass.tech)
- 📚 **Documentation**: [docs.ncompass.tech](https://docs.ncompass.tech)
- 💬 **Community**: [community.ncompass.tech](https://community.ncompass.tech)
- 🐛 **Issues**: [GitHub Issues](https://github.com/ncompass-tech/ncompass/issues)

## Requirements

- Python 3.11 or higher
- PyTorch 2.0+ (optional, for torch profiling features)
- CUDA-capable GPU (optional, for GPU profiling)

## Development

### Coverage & Quality Tools

All development and coverage tools are in the **`tools/`** directory:

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run coverage checks (from tools/ directory)
cd tools
make all-checks         # Run all checks
make coverage           # Unit test coverage
make docstring-coverage # Docstring coverage
make type-stats         # Type hint coverage
make lint               # Run linters
make format             # Auto-format code
```

See **[tools/COVERAGE.md](tools/COVERAGE.md)** for comprehensive documentation.

### Project Structure

```
ncompass/
├── pyproject.toml      # Project config (only root file)
├── ncompass/           # Main package
├── tests/              # Test suite
├── examples/           # Usage examples
└── tools/              # All development tools
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Support

- **Documentation**: [docs.ncompass.tech](https://docs.ncompass.tech)
- **Community Forum**: [community.ncompass.tech](https://community.ncompass.tech)
- **Email**: aditya.rajagopal@ncompass.tech

Made with ⚡ by [nCompass Technologies](https://ncompass.tech)
