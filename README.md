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
> 1. You are running Python 3.10+
> 2. You have `Pydantic>=2.0` installed

## Examples

Refer to our [open source GitHub repo](https://github.com/nCompass-tech/ncompass/tree/main/examples) for examples. Our examples are built to work together with the VSCode extension. For instance, with adding tracepoints to the code, you can add/remove tracepoints using the extension and then run profiling using our examples. 

- **[Basic TorchProfile Example](examples/basic_example/)**
- **[Nsight Systems Examples](examples/nsys_example/)**
- **[Running remotely on Modal](examples/modal_example/)**
- **[Fast conversion of .nsys-rep to .json.gz](examples/trace_converter/)**

## Online Resources

- 🌐 **Website**: [ncompass.tech](https://ncompass.tech)
- 📚 **Documentation**: [Documentation](https://round-hardhat-a0a.notion.site/ncprof-Quick-Start-2c4097a5a430805db541c01762ea6922?source=copy_link)
- 💬 **Community**: [community.ncompass.tech](https://community.ncompass.tech)
- 🐛 **Issues**: [GitHub Issues](https://github.com/ncompass-tech/ncompass/issues)
- __ **Discord**: [Join our discord](https://discord.gg/9K48xTxKvN)

## Requirements

- Python 3.10 or higher
- Nsight Systems CLI installed (for .nsys-rep to .json.gz conversion features)

## Building without packaging
Because of Rust dependencies for the fast .nsys-rep to .json.gz converter, `-e` (editable) builds
aren't setup. To build you have to just `pip install ./` and use the package from your python env.

To run tests, run the following:
```bash
nix develop
pytest tests/ # python tests
cd ncompass_rust/trace_converters/
cargo test --target=x86_64-unknown-linux-musl # rust tests
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

Made with ⚡ by [nCompass Technologies](https://ncompass.tech)
