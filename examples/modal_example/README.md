# Modal Example: Remote PyTorch Profiling with nCompass SDK

This example demonstrates how to profile GPU-accelerated PyTorch neural network training on [Modal](https://modal.com) with integrated nCompass SDK tracing and instrumentation. Run your profiling workloads on Modal's cloud GPUs without managing infrastructure.

## What This Example Does

This example shows how to:
- Set up a Modal App with GPU support (A10G)
- Integrate nCompass SDK for zero-instrumentation profiling
- Train a simple neural network with automatic tracepoint injection
- Profile GPU-accelerated PyTorch code remotely on Modal
- Link user annotations to GPU kernels for better trace visualization
- Save profiling traces to a Modal Volume and download them locally
- View traces directly in the nCompass VSCode extension

## Prerequisites

Before you begin, ensure you have:

- **Python 3.11+** (required)
- **Modal account**: Sign up at [modal.com](https://modal.com) and authenticate
- **VSCode** with the [nCompass extension](https://marketplace.visualstudio.com/items?itemName=nCompassTech.ncprof-vscode) installed

## Step-by-Step Guide

### Step 1: Install Dependencies

Create a virtual environment and install the required packages:

```bash
# Create a virtual environment
python3.11 -m venv venv-modal-example

# Activate the virtual environment
source venv-modal-example/bin/activate  # On Windows: venv-modal-example\Scripts\activate

# Install Modal and dependencies
pip install -r requirements.txt
```
### Step 2: Set Up Modal Account

Authenticate with Modal:

```bash
modal setup
```

This will:
- Open your browser to sign up or log in
- Save your authentication token locally
- Configure Modal CLI access

> 💡 **Tip**: If you don't have a Modal account, you can sign up for free at [modal.com](https://modal.com).

### Step 3: Set Up the VSCode Extension

The nCompass SDK requires a profiling configuration file that is automatically created by the VSCode extension:

1. **Install the extension**: Open VSCode and install the [nCompass extension](https://marketplace.visualstudio.com/items?itemName=nCompassTech.ncprof-vscode)
2. **Open the example directory**: Open the `modal_example` folder in VSCode
3. **Add tracepoints**:
   - Open `torch_profiling_example.py` in VSCode
   - Navigate to the `train_simple_network` function (around line 67)
   - Use the extension to [add tracepoints](https://docs.ncompass.tech/ncprof/quick-start#step-5-register-a-tracepoint) to functions you want to profile
   - The extension will automatically create the configuration file at `.cache/ncompass/profiles/.default/.default/current/config.json`

> 💡 **Tip**: The config file is automatically copied to the Modal container, so your tracepoints will be active during remote profiling. You can add tracepoints to any function that will run on Modal.

### Step 4: Run Basic Profiling

Run the example with default settings:

```bash
modal run torch_profiling_example.py
```

This will:
- Build a Modal image with PyTorch and nCompass SDK
- Copy your local config.json to the container
- Provision an A10G GPU on Modal
- Train a simple neural network for 3 profiling steps
- Automatically inject profiling markers (if config exists)
- Link `user_annotation` events to GPU kernels (enabled by default)
- Save traces to a Modal Volume
- Download the trace file locally to `.traces/` directory

**Expected output:**
```
✓ Created objects.
✓ Running function profile...
trace saved to train_simple_network/abc123-def456-...
Linking user_annotation events to kernels...
Linked 5 user_annotation events to kernels
Trace updated: train_simple_network/abc123-def456-.../trace.pt.trace.json
trace saved locally at .traces/trace.pt.trace.json - view using the nCompass VSCode extension
```

### Step 5: View Traces in VSCode

1. **Open the trace file**: In VSCode, navigate to the `.traces/` directory and open the generated `.pt.trace.json` file
2. **Open with trace viewer**: Right-click on the `.pt.trace.json` file → **Open With...** → **GPU Trace Viewer**
3. **Explore the trace**:
   - See CPU events (function calls, user annotations)
   - See GPU events (kernel executions, memory operations)
   - Navigate between code and trace using the extension's code-to-trace linking
   - View your custom tracepoints that were injected automatically

## Command-Line Options

### Main Options

- `--function TEXT`: Name of the function to profile (default: "train_simple_network")
- `--label TEXT`: Optional label for the profiling run (e.g., "baseline", "optimized")
- `--steps INT`: Number of profiling steps (default: 3)
- `--local-trace-dir TEXT`: Local directory to save traces (default: ".traces")

### Profiling Options

- `--record-shapes`: Record tensor shapes during profiling
- `--profile-memory`: Profile memory usage
- `--with-stack`: Include Python stack traces
- `--print-rows INT`: Number of rows to print in summary table (default: 10)

### Advanced Options

- `--no-link`: Disable linking user_annotation events to kernels (linking is enabled by default)
- `--verbose`, `-v`: Print detailed statistics when linking annotations

### Examples

```bash
# Basic run with defaults
modal run torch_profiling_example.py

# Custom label and more profiling steps
modal run torch_profiling_example.py --label "baseline" --steps 5

# Show top 20 operations in summary table
modal run torch_profiling_example.py --print-rows 20

# Save traces to a custom local directory
modal run torch_profiling_example.py --local-trace-dir ./my-traces

# Disable annotation linking
modal run torch_profiling_example.py --no-link

# Show detailed linking statistics
modal run torch_profiling_example.py --verbose
```

## Understanding the Workflow

### How Remote Profiling Works

1. **Local Configuration**: The VSCode extension creates a configuration file at `.cache/ncompass/profiles/.default/.default/current/config.json` that specifies which functions should be instrumented.

2. **Modal Image Build**: When you run the script, Modal:
   - Builds a container image with PyTorch, CUDA, and nCompass SDK
   - Copies your local `config.json` file into the container at `/config/ncompass_config.json`
   - Installs all required dependencies

3. **Remote Execution**: Modal provisions a GPU (A10G) and runs your code:
   - The `profile` function initializes nCompass SDK with the copied config
   - AST rewriting automatically wraps specified functions with profiling contexts
   - PyTorch's profiler captures CPU and GPU events
   - Traces are saved to a Modal Volume

4. **Trace Download**: After profiling completes:
   - The trace file is downloaded from the Modal Volume to your local `.traces/` directory
   - You can view it immediately in the VSCode extension

### Key Components

**Modal Volume**: Traces are persisted in a Modal Volume named `example-traces`. This allows:
- Traces to persist across runs
- Multiple runs to be stored
- Easy access and download

**Config File Copying**: The script automatically detects and copies your local config file:
```python
ncompass_local_tracepoint_config = \
    Path(f"{os.getcwd()}/.cache/ncompass/profiles/.default/.default/current/config.json")
if ncompass_local_tracepoint_config.exists(): 
    image = image.add_local_file(ncompass_local_tracepoint_config, 
                                 "/config/ncompass_config.json")
```

**SDK Initialization**: Before profiling, the SDK is initialized in the `profile` function:
```python
if Path(ncompass_remote_tracepoint_config).exists():
    with open(ncompass_remote_tracepoint_config) as f:
        cfg = json.load(f)
        enable_rewrites(config=RewriteConfig.from_dict(cfg))
```

> ⚠️ **Important**: SDK initialization must happen **before** the function you want to profile is called, and **outside** any function you want to add tracepoints to.

## Output Files

### Trace Files

Profiling generates Chrome trace JSON files that are:
1. **Saved to Modal Volume**: Persisted in the cloud for future access
2. **Downloaded locally**: Copied to your `.traces/` directory (or custom directory specified with `--local-trace-dir`)

Trace files are named like:
```
.traces/trace.pt.trace.json
```

### Trace Contents

The trace files contain:
- **CPU events**: Function calls, user annotations, CUDA runtime API calls
- **GPU events**: Kernel executions, memory operations
- **Linked events**: `gpu_user_annotation` events that span kernel execution times (enabled by default)

## Use Cases

### Use Case 1: Profile Without Local GPU

If you don't have a GPU locally, Modal provides cloud GPU access:

```bash
modal run torch_profiling_example.py --label "remote_gpu_test"
```

### Use Case 2: Compare Different Configurations

Run multiple profiling sessions with different parameters:

```bash
# Baseline
modal run torch_profiling_example.py --label "baseline" --hidden-size 512

# Larger model
modal run torch_profiling_example.py --label "large" --hidden-size 1024

# Compare traces in VSCode extension
```

### Use Case 3: Iterative Optimization

1. Add tracepoints to your code using the VSCode extension
2. Run profiling: `modal run torch_profiling_example.py`
3. View traces and identify bottlenecks
4. Optimize your code
5. Repeat steps 2-4

## Troubleshooting

### Issue: "Config file not found" or rewrites not enabled

**Solution**: 
- Make sure you've set up the VSCode extension and added tracepoints
- Verify the config file exists at `.cache/ncompass/profiles/.default/.default/current/config.json`
- The script will automatically copy it to the Modal container if it exists

### Issue: Modal authentication errors

**Solution**:
- Run `modal setup` to authenticate
- Check that your Modal account is active
- Verify your authentication token: `modal token show`

### Issue: GPU not available

**Solution**:
- Check your Modal account has GPU access (free tier includes limited GPU credits)
- Verify GPU availability: `modal gpu list`
- The example uses A10G by default - you can modify the GPU type in the script if needed

### Issue: Import errors with ncompass

**Solution**:
- Ensure you're using Python 3.11+
- The Modal image automatically installs nCompass SDK, but verify the version in the script
- Check Modal logs: `modal app logs example-torch-profiling`

### Issue: Trace files not downloading

**Solution**:
- Check that the `.traces/` directory exists and is writable
- Verify Modal Volume is accessible: `modal volume list`
- Check Modal logs for errors: `modal app logs example-torch-profiling`

### Issue: User annotations not linking to kernels

**Solution**:
- Ensure tracepoints are placed in GPU-executed code
- Check that the profiler captures both CPU and CUDA activities (enabled by default)
- Use `--verbose` flag to see detailed linking statistics
- Verify your code actually runs on GPU (check for `.cuda()` calls)

## File Structure

- `torch_profiling_example.py`: Main example script with Modal app definition
- `requirements.txt`: Python dependencies for local development
- `README.md`: This file

## Additional Resources

- **[nCompass VSCode Extension Documentation](https://docs.ncompass.tech/ncprof/quick-start)** - Complete guide to using the extension
- **[Modal Documentation](https://modal.com/docs)** - Complete Modal platform documentation
- **[Modal PyTorch Profiling Example](https://modal.com/docs/examples/torch_profiling)** - Original Modal example
- **[PyTorch Profiler Documentation](https://pytorch.org/docs/stable/profiler.html)** - Learn about PyTorch's profiling capabilities

## Support

For questions or issues:
- **nCompass Support**: Check the [Documentation](https://docs.ncompass.tech) or visit the [Community Forum](https://community.ncompass.tech)
- **GitHub Issues**: Open an issue on [GitHub](https://github.com/ncompass-tech/ncompass/issues)
- **Modal Support**: Check [Modal Documentation](https://modal.com/docs) or [Modal Community](https://modal.com/community)
