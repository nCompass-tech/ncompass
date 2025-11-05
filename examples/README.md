# nCompass SDK Examples

This directory contains example scripts demonstrating how to use the nCompass SDK for profiling and tracing AI inference workloads.

## Available Examples

### Basic Examples

- **[basic_usage.py](basic_usage.py)** - Simple profiling session with trace analysis
- **[profiling_session.py](profiling_session.py)** - Complete ProfilingSession workflow
- **[advanced_tracing.py](advanced_tracing.py)** - Advanced features including iterative profiling

## Running Examples

Each example is self-contained and can be run directly:

```bash
python examples/basic_usage.py
```

Make sure you have nCompass installed:

```bash
pip install ncompass
```

## Example Structure

Each example follows this general pattern:

1. **Setup** - Initialize ProfilingSession and configure
2. **Execution** - Run the profiling target
3. **Analysis** - Get and display trace insights
4. **Iteration** (Advanced) - Submit feedback and refine

## Prerequisites

Some examples may require additional dependencies:

```bash
# For PyTorch examples
pip install ncompass[torch]

# For full development environment
pip install ncompass[dev]
```

## Example Data

Examples use synthetic workloads to demonstrate profiling capabilities. For real-world usage:
- Replace model code with your actual inference code
- Adjust trace output directories as needed
- Configure analysis service URLs if using custom deployment

## Support

For questions about examples:
- Check the [Documentation](https://docs.ncompass.tech)
- Visit the [Community Forum](https://community.ncompass.tech)
- View the [API Reference](../docs/api_reference.md)

## Contributing Examples

We welcome example contributions! Please ensure your example:
- Is self-contained and runnable
- Includes clear comments explaining each step
- Demonstrates a specific feature or use case
- Follows the existing code style

