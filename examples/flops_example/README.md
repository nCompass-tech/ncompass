# Basic Example: PyTorch Profiling with nCompass SDK

This example demonstrates how to profile PyTorch neural network training using the nCompass SDK with **zero-instrumentation profiling**. It shows how to:

- Automatically inject profiling markers using AST-level code rewriting
- Train a simple neural network with automatic instrumentation
- Profile GPU-accelerated PyTorch code locally
- Link user annotations to GPU kernel executions
- View traces directly in the VSCode extension

> **Note**: This example works best with the [nCompass VSCode extension](https://marketplace.visualstudio.com/items?itemName=nCompassTech.ncprof-vscode), which automatically creates the profiling configuration and provides an integrated trace viewer.

## Prerequisites

Before you begin, ensure you have:

- **Python 3.11+** (required)
- **CUDA-capable GPU**
- **VSCode** with the [nCompass extension](https://marketplace.visualstudio.com/items?itemName=nCompassTech.ncprof-vscode) installed

## Step-by-Step Guide

### Step 1: Install Dependencies

Create a virtual environment and install the required packages:

```bash
# Create a virtual environment
python3.11 -m venv venv-basic-example

# Activate the virtual environment
source venv-basic-example/bin/activate  # On Windows: venv-basic-example\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

> ⚠️ **Troubleshooting**: If you encounter issues with `ncompasslib` or `pydantic`, ensure you're running Python 3.11 and have `Pydantic>=2.0` installed.

### Step 2: Set Up the VSCode Extension

The nCompass SDK requires a profiling configuration file that is automatically created by the VSCode extension:

1. **Install the extension**: Open VSCode and install the [nCompass extension](https://marketplace.visualstudio.com/items?itemName=nCompassTech.ncprof-vscode)
2. **Open the example directory**: Open the `basic_example` folder in VSCode
3. **Add tracepoints**:
   - Open `simplenet.py` in VSCode
   - Use the extension to add tracepoints to functions you want to profile
   - The extension will automatically create the configuration file at `.cache/ncompass/profiles/.default/.default/current/config.json`

> 💡 **Tip**: If you don't add tracepoints manually, the example will still work, but you won't see custom annotations in your traces. The extension's tracepoint UI makes it easy to mark which functions should be profiled.

### Step 3: Run Basic Profiling

Run the example with default settings:

```bash
python main.py
```

This will:
- Train a simple neural network for 3 profiling steps
- Automatically inject profiling markers (if config exists)
- Link `user_annotation` events to GPU kernels (enabled by default)
- Generate a Chrome trace JSON file in the `.traces/` directory
- Print a summary of the profiling session

### Step 4: View Traces in VSCode

1. **Open the trace file**: In VSCode, navigate to the `.traces/` directory and open the generated `.pt.trace.json` file by `right-clicking -> Open With... -> GPU Trace Viewer`
2. **Explore the trace**:
   - See CPU events (function calls, user annotations)
   - See GPU events (kernel executions, memory operations)
   - Navigate between code and trace using the extension's code-to-trace linking

### Step 5: Customize Your Profiling Run

You can customize the profiling session with various command-line options:

```bash
# Profile with a custom label
python main.py --label "baseline" --steps 5 --epochs 20

# Profile a larger model
python main.py --label "large_model" --hidden-size 1024 --steps 3

# Get verbose linking statistics
python main.py --verbose --steps 3

# Disable automatic annotation linking
python main.py --no-link

# Print top operations summary
python main.py --print-rows 20
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
- `--link-only PATH`: Only link an existing trace file (must end with `.pt.trace.json`)

## Understanding the Workflow

### How Zero-Instrumentation Profiling Works

1. **Configuration**: The VSCode extension creates a configuration file at `.cache/ncompass/profiles/.default/.default/current/config.json` that specifies which functions should be instrumented.

2. **AST Rewriting**: When you run `main.py`, it calls `enable_rewrites()` which:
   - Loads the configuration file
   - Uses Python's import system to intercept module loading
   - Automatically wraps specified functions with `torch.profiler.record_function()` contexts
   - No manual code changes needed!

3. **Profiling**: PyTorch's profiler captures:
   - CPU events (function calls, annotations)
   - GPU events (kernel executions, memory operations)
   - CUDA runtime API calls

4. **Linking**: The SDK automatically links CPU-side annotations to their corresponding GPU kernel executions using CUDA correlation IDs.

### File Structure

- `main.py`: Main profiling script with command-line interface
- `simplenet.py`: Simple neural network model and training function (this is what gets instrumented)
- `utils.py`: Shared utility functions for trace processing and statistics
- `nc_pkg.py`: Docker helper script

## Use Case: Link Existing Trace
When you add TorchRecord or NVTX contexts to source code, generally show up on CPU timelines. This
means they'll be connected with the CPU side kernel launch calls, but won't necessarily align
visually with the kernels that are actually running on the GPU.

What we mean here by kernel linking is that we analyze the trace file and link the CPU side events
with their corresponding GPU side events and ensure that the markers you placed visually appear
over the kernel events rather than the CPU kernel launch events.

If you have an existing trace file and want to add kernel linking:

```bash
python main.py --link-only .traces/train_simple_network_*/xxx.pt.trace.json
```

This creates a new file with `.linked.pt.trace.json` suffix containing the linked events.

## Output Files

### Trace Files

Profiling generates Chrome trace JSON files in the `.traces/` directory with names like:
```
train_simple_network_baseline_20240101_120000_abc12345/xxx.pt.trace.json
```

### Trace Contents

The trace files contain:
- **CPU events**: Function calls, user annotations, CUDA runtime API calls
- **GPU events**: Kernel executions, memory operations
- **Linked events**: `gpu_user_annotation` events that span kernel execution times (enabled by default)

## Troubleshooting

### Issue: "No config file found" or rewrites not enabled

**Solution**: Make sure you've set up the VSCode extension. Once you add tracepoints, the config 
file should be generated at `.cache/ncompass/profiles/.default/.default/current/config.json`.

### Issue: Import errors with ncompasslib or pydantic

**Solution**: 
- Ensure you're using Python 3.11+
- Install `Pydantic>=2.0`: `pip install "pydantic>=2.0"`
- Reinstall ncompass: `pip install --upgrade ncompass`

### Issue: CUDA not available

**Solution**: The example will run on CPU, but GPU profiling provides much better insights. Ensure:
- CUDA-capable GPU is available
- PyTorch with CUDA support is installed
- CUDA drivers are properly configured

### Issue: Trace file not found

**Solution**: Check that:
- The profiling completed successfully
- The `.traces/` directory exists
- Look for the most recent trace file in the output directory

## Docker Setup (Optional)

If you prefer to use Docker, you can run the example in a containerized environment:

### Prerequisites

- Docker and Docker Compose installed
- NVIDIA Docker runtime (for GPU support)

### Quick Start

```bash
# Build and run the container
python nc_pkg.py --build --run
```

This will:
- Build a Docker image with all dependencies
- Start a container with GPU access
- Drop you into an interactive shell inside the container

### Docker Commands

```bash
# Build the image
python nc_pkg.py --build

# Run the container (interactive shell)
python nc_pkg.py --run

# Run a command in the container
python nc_pkg.py --exec "python main.py --label test"

# Stop and remove the container
python nc_pkg.py --down
```

### Docker Notes

- The container mounts the current directory to `/workspace`
- Traces are saved to `.traces/` which is accessible from the host
- GPU access is enabled via NVIDIA Docker runtime
- The container uses Python 3.12 and includes all dependencies

## Additional Resources

- **[nCompass VSCode Extension Documentation](https://docs.ncompass.tech/ncprof/quick-start)** - Complete guide to using the extension
- **[PyTorch Profiler Documentation](https://pytorch.org/docs/stable/profiler.html)** - Learn about PyTorch's profiling capabilities
- **[nCompass SDK Documentation](../../README.md)** - More details on SDK documentation
- **[Examples README](../README.md)** - Overview of all available examples

## Support

For questions or issues:
- Check the [Documentation](https://docs.ncompass.tech)
- Visit the [Community Forum](https://community.ncompass.tech)
- Open an issue on [GitHub](https://github.com/ncompass-tech/ncompass/issues)
