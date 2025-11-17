# Basic Example: PyTorch Profiling with nCompass SDK

This example demonstrates how to profile PyTorch neural network training using the nCompass SDK. It shows how to:

1. Integrate nCompass SDK for tracing and instrumentation
2. Train a simple neural network with instrumentation
3. Profile GPU-accelerated PyTorch code locally
4. Link user annotations to GPU kernel executions
5. Save and view profiling traces

> **Note**: This example is derived from the [Modal torch_profiling example](https://modal.com/docs/examples/torch_profiling#tracing-and-profiling-gpu-accelerated-pytorch-programs-on-modal), adapted to run locally with nCompass SDK integration instead of Modal's cloud infrastructure.

## Prerequisites

- Python 3.8+
- CUDA-capable GPU (optional, but recommended for best results)

## Quick Start

### Installation
#### Without Docker
```bash
python -m venv venv-basic-example
source venv-basic-example/bin/activate
pip install uv 
uv pip install -r requirements.txt
```

#### With Docker
```bash
python nc_pkg.py --build --run
# This puts you in an envi
```

### Basic Profiling

Run the example with default settings:

```bash
python main.py
```

This will:
- Train a simple neural network for 3 profiling steps
- Automatically link `user_annotation` events to GPU kernels (enabled by default)
- Generate a Chrome trace JSON file in `.traces/` directory
- Save profiling data for analysis

### With Custom Options

```bash
python main.py --label "baseline" --steps 5 --epochs 20 --hidden-size 1024
```

## Command-Line Options

### Main Options

- `--label TEXT`: Optional label for the profiling run (e.g., "baseline", "optimized")
- `--steps INT`: Number of profiling steps (default: 3)
- `--epochs INT`: Number of training epochs per profiling step (default: 10)
- `--hidden-size INT`: Hidden layer size for the neural network (default: 512)
- `--trace-dir TEXT`: Directory to save traces (default: ".traces")

### Profiling Options

- `--record-shapes`: Record tensor shapes during profiling (default: True)
- `--profile-memory`: Profile memory usage
- `--with-stack`: Include Python stack traces (default: True)
- `--print-rows INT`: Number of rows to print in summary table (default: 10)

### Advanced Options

- `--custom-config-path TEXT`: Path to custom profiling targets JSON config
- `--no-link`: Disable linking user_annotation events to kernels (linking is enabled by default)
- `--verbose`, `-v`: Print detailed statistics when linking annotations

## Linking User Annotations to Kernels

The example includes functionality to link CPU-side `user_annotation` events (like `torch.profiler.record_function`) to their corresponding GPU kernel executions. This helps visualize the relationship between high-level operations and actual GPU work.

**Linking is enabled by default** - all profiling runs automatically link annotations unless you disable it with `--no-link`.

### Automatic Linking During Profiling

By default, profiling automatically links annotations:

```bash
python main.py
```

With verbose output to see detailed statistics:

```bash
python main.py --verbose
```

To disable linking:

```bash
python main.py --no-link
```

When linking is enabled (default), the profiler will:
- Run the profiling session
- Automatically link `user_annotation` events to kernels using CUDA runtime correlation IDs
- Overwrite the original trace file with the linked version
- Print detailed statistics (if `--verbose` is used)

### Standalone Linking Script

You can also link annotations in an existing trace file using the standalone script:

```bash
python link_user_annotation_kernels.py input_trace.json output_trace.json
```

The script will:
- Load the input trace file
- Find overlapping intervals between `user_annotation` and CUDA runtime events
- Use correlation IDs to link CUDA runtime calls to kernels
- Create new `gpu_user_annotation` events that span actual kernel execution times
- Print detailed statistics about the linking process
- Save the linked trace to the output file

**Replacement Logic:**
- If both `gpu_user_annotation` and `user_annotation` exist (same name) → replaces `gpu_user_annotation`
- If only `user_annotation` exists → replaces `user_annotation` with new `gpu_user_annotation`
- If only `gpu_user_annotation` exists → leaves it unchanged

## Output

### Trace Files

Profiling generates Chrome trace JSON files in the `.traces/` directory with names like:
```
train_simple_network_baseline_20240101_120000_abc12345/xxx.pt.trace.json
```

### Trace Contents

The trace files contain:
- **CPU events**: Function calls, user annotations, CUDA runtime API calls
- **GPU events**: Kernel executions, memory operations
- **Linked events**: `gpu_user_annotation` events that span kernel execution times (enabled by default, use `--no-link` to disable)

## Viewing Traces

### Using the nCompass VSCode Extension (Recommended)

The easiest way to view traces is using the [nCompass VSCode extension](https://docs.ncompass.tech/ncprof/quick-start). The extension allows you to:

- Open trace files directly in VSCode
- View CPU and GPU activity in an interactive timeline
- See linked user annotations and their corresponding kernel executions
- Filter and analyze performance data

To get started, install the nCompass extension in VSCode and follow the [quick start guide](https://docs.ncompass.tech/ncprof/quick-start) to view your trace files.

## Example Workflows

### 1. Basic Profiling Run

```bash
python main.py --steps 5 --epochs 20
```

### 2. Profiling with Verbose Linking Statistics

```bash
python main.py --verbose --steps 3
```

Note: Linking is enabled by default, so `--link` is not needed. Use `--verbose` to see detailed statistics.

### 3. Compare Different Configurations

```bash
# Baseline
python main.py --label "baseline" --hidden-size 512

# Larger model
python main.py --label "large" --hidden-size 1024

# Compare traces in Perfetto
```

### 4. Link Existing Trace

```bash
python link_user_annotation_kernels.py \
    .traces/train_simple_network_*/xxx.pt.trace.json \
    linked_trace.json
```

## Understanding the Code

### File Structure

- `main.py`: Main profiling script with command-line interface
- `simplenet.py`: Simple neural network model and training function
- `link_user_annotation_kernels.py`: Standalone script for linking annotations
- `utils.py`: Shared utility functions for trace processing and statistics

### Profiling Targets

The example uses nCompass profiling targets to automatically instrument specific code regions. See `PROFILING_TARGETS` in `main.py` for configuration.

The profiling targets configuration mimics the `config.json` format generated by the [ncprof VSCode extension](https://docs.ncompass.tech/ncprof/quick-start). This allows you to define which functions should be automatically wrapped with profiling contexts that appear in PyTorch profiler traces.

### Custom Configuration

You can provide a custom profiling targets JSON file:

```bash
python main.py --custom-config-path my_config.json
```

The JSON should follow the same structure as `PROFILING_TARGETS` in `main.py`, which matches the format used by the ncprof VSCode extension. See the [ncprof quick start guide](https://docs.ncompass.tech/ncprof/quick-start) for more details on the configuration format.

## Running a test with the local version of the SDK
```bash
Run ./setup_local_test_env.sh
```
This will create a symlink of the SDK here. Uninstall ncompass from whatever env (docker / venv)
you're running in.

## Additional Resources
- [ncprof VSCode Extension Documentation](https://docs.ncompass.tech/ncprof/quick-start)
- [PyTorch Profiler Documentation](https://pytorch.org/docs/stable/profiler.html)
- [nCompass SDK Documentation](../README.md)

