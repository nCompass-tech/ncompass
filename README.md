# nCompass Python SDK

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

High-performance AI inference SDK with built-in tracing and profiling capabilities. Built by [nCompass Technologies](https://ncompass.tech).

## Features

- **🚀 High-Performance GPU Kernels** - Custom-built kernels optimized for AI inference
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

With PyTorch support:

```bash
pip install ncompass[torch]
```

For development:

```bash
pip install ncompass[dev]
```

## Quick Start

```python
from ncompass.trace import ProfilingSession

# Initialize profiling session
session = ProfilingSession(
    trace_output_dir="./traces",
    cache_dir="./cache"
)

# Profile your inference code
def my_inference():
    # Your model inference here
    outputs = model(inputs)
    return outputs

# Run profiling
trace_path = session.run_profile(my_inference)

# Get AI-powered insights
summary = session.get_trace_summary(trace_path)
print(summary['markdown'])
```

## Core Capabilities

### Iterative Profiling Workflow

```python
# 1. Capture initial trace
trace = session.run_profile(inference_fn)

# 2. Get AI analysis
summary = session.get_trace_summary(trace)

# 3. Submit feedback for targeted profiling
session.submit_feedback(
    feedback_text="Why is attention slow?",
    target_module="model.attention",
    start_line=100,
    end_line=200
)

# 4. Apply targeted markers and re-run
session.apply_targeted_markers()
trace = session.run_profile(inference_fn)
```

### AST-Level Instrumentation

```python
from ncompass.trace import enable_rewrites, RewriteConfig

# Enable automatic code instrumentation
config = RewriteConfig(targets={...})
enable_rewrites(config=config)

# Your code runs with automatic profiling markers
run_inference()
```

### Trace Filtering

```python
# Filter traces to focus on key operations
filtered_trace = session.filter_trace(
    include_cuda_kernels=True,
    min_duration_us=100.0
)
```

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

### Built for Performance

- Custom GPU kernels optimized for AI workloads
- One of the fastest open-source inference solutions
- Scalable architecture for production deployments

### Production-Ready Monitoring

- Detailed performance metrics out of the box
- Health monitoring and alerting
- Separate dev/prod configurations for dedicated instances

### AI-Powered Insights

- Automatic bottleneck identification
- Intelligent profiling marker suggestions
- Progressive optimization guidance

## Requirements

- Python 3.8 or higher
- PyTorch 2.0+ (optional, for torch profiling features)
- CUDA-capable GPU (optional, for GPU profiling)

## Examples

See the [examples](examples/) directory for complete working examples:

- **[basic_usage.py](examples/basic_usage.py)** - Simple profiling session
- **[profiling_session.py](examples/profiling_session.py)** - Complete workflow
- **[advanced_tracing.py](examples/advanced_tracing.py)** - Iterative optimization


## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Support

- **Documentation**: [docs.ncompass.tech](https://docs.ncompass.tech)
- **Community Forum**: [community.ncompass.tech](https://community.ncompass.tech)
- **Email**: support@ncompass.tech

## About nCompass Technologies

nCompass Technologies builds high-performance AI inference solutions with custom GPU kernels. Our mission is to make AI inference faster, more scalable, and more accessible.

Learn more at [ncompass.tech](https://ncompass.tech).

---

Made with ⚡ by [nCompass Technologies](https://ncompass.tech)
