# nCompass Python SDK

[![PyPI](https://img.shields.io/pypi/v/ncompass.svg)](https://pypi.org/project/ncompass/)
[![Downloads](https://static.pepy.tech/badge/ncompass)](https://pepy.tech/project/ncompass)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

Profiling and trace analysis SDK. Built by [nCompass Technologies](https://ncompass.tech).

## Features

- **📊 Advanced Profiling** - Built-in performance monitoring and health metrics
- **🤖 AI-Powered Analysis** - Intelligent trace analysis and bottleneck identification
- **🔄 Iterative Optimization** - Progressive profiling workflow for targeted improvements
- **🎯 AST-Level Instrumentation** - Automatic code instrumentation without manual changes
- **⚡ Production-Ready** - Separate development and production configurations

## Installation

Install via pip:

```bash
pip install ncompass
```

## Examples
<!-- TODO: Update readme after setting up actual examples -->
Refer to our [open source GitHub repo](https://github.com/nCompass-tech/ncompass) for examples.

## Documentation

- **[Getting Started](docs/getting_started.md)** - Installation and basic usage
- **[API Reference](docs/api_reference.md)** - Complete API documentation  
- **[Advanced Usage](docs/advanced_usage.md)** - Advanced features and best practices
- **[Examples](examples/)** - Working code examples

## Online Resources

- 🌐 **Website**: [ncompass.tech](https://ncompass.tech)
- 📚 **Documentation**: [docs.ncompass.tech](https://docs.ncompass.tech)
- 💬 **Community**: [community.ncompass.tech](https://community.ncompass.tech)
- 🐛 **Issues**: [GitHub Issues](https://github.com/ncompass-tech/ncompass/issues)

## Why nCompass?

- AI assisted bottleneck identification
- Intelligent profiling marker suggestions
- Progressive optimization guidance

## Requirements

- Python 3.9 or higher
- PyTorch 2.0+ (optional, for torch profiling features)
- CUDA-capable GPU (optional, for GPU profiling)

## Examples

See the [examples](examples/) directory for complete working examples:

- **[basic_usage.py](examples/basic_usage.py)** - Simple profiling session
- **[profiling_session.py](examples/profiling_session.py)** - Complete workflow
- **[advanced_tracing.py](examples/advanced_tracing.py)** - Iterative optimization

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
├── docs/               # Documentation
└── tools/              # All development tools
    ├── Makefile
    ├── COVERAGE.md
    ├── pyrightconfig.json
    └── .coveragerc
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Support

- **Documentation**: [docs.ncompass.tech](https://docs.ncompass.tech)
- **Community Forum**: [community.ncompass.tech](https://community.ncompass.tech)
- **Email**: support@ncompass.tech

## About nCompass Technologies

We are building tools that allow GPU kernel developers save hours of dev time every week. Write code that runs twice as fast, twice as quickly.

Learn more at [ncompass.tech](https://ncompass.tech).

---

Made with ⚡ by [nCompass Technologies](https://ncompass.tech)
