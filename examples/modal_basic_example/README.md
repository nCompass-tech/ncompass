# Modal Example: Remote PyTorch Profiling with nCompass SDK

This example demonstrates how to profile GPU-accelerated PyTorch neural network training on [Modal](https://modal.com) 
It uses the nCompass VSCode / Cursor IDE to add TorchRecords in the code and the SDK to inject the
records at runtime so that the results appear in the trace.

We also run an additional step of linking the created annotations to CUDA kernel launches, so that
in the final generated chrome trace, the annotation appear on top of the kernels launched, not just
the host side calls.

## Prerequisites

Before you begin, ensure you have:

- **Python 3.10+** (required)
- **Modal account**: Sign up at [modal.com](https://modal.com) and authenticate
- **VSCode** / **Cursor** with the [nCompass extension](https://marketplace.visualstudio.com/items?itemName=nCompassTech.ncprof-vscode) installed

## Step-by-Step Guide

### Step 1: Install Dependencies

Create a virtual environment and install the required packages:

```bash
# Create a virtual environment
python3 -m venv venv-modal-example

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
        - Highlight region of the code you want to add the context around
        - `Ctrl/Cmd + ,` opens a dropdown option (lightbulb)
        - Click on `Add Region to Profiel (TorchRecord)`
   - The extension will automatically create the configuration file at `.cache/ncompass/profiles/.default/Torch/current/config.json`

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

### Step 5: View Traces in VSCode

1. **Open the trace file**: In VSCode, navigate to the `.traces/` directory and open the generated `.pt.trace.json` file
2. **Open with trace viewer**: Right-click on the `.pt.trace.json` file → **Open With...** → **GPU Trace Viewer**
3. **Explore the trace**:
   - See CPU events (function calls, user annotations)
   - See GPU events (kernel executions, memory operations)
   - Navigate between code and trace using the extension's code-to-trace linking
   - View your custom tracepoints that were injected automatically

## Understanding the Workflow

### Create the local configuration file

Open the VSCode extension and add the region you want to profile inside the `train_simple_network`
function (following Step 3 above). The VSCode extension creates a configuration file 
at `.cache/ncompass/profiles/.default/Torch/current/config.json` that specifies which functions 
should be instrumented. The file is by default created in a `.cache` directory at the top level 
of the workspace that is currently open in VSCode.

This registers the parts of the code that you want to instrument without actually adding the
torch record contexts in the code itself. 

We welcome new contributions for injectors, which allow you to add code at runtime and not clutter 
your codebase with code you only use for debugging.

### Setup necessary packages, environment variables and remote files

In order to use the nCompass SDK, you need to pip install `ncompass>=0.1.9` and set the following
environment variables: 
- NCOMPASS_CACHE_DIR
- NCOMPASS_PROFILER_TYPE

The lines of code in `torch_profiling_example.py` that do this are here:
```python
image = modal.Image.debian_slim(python_version="3.10")\
    .uv_pip_install("torch")\
    .uv_pip_install("ncompass>=0.1.9")\
    .uv_pip_install("pathlib")\
    .env({"NCOMPASS_CACHE_DIR":     "/config",\
          "NCOMPASS_PROFILER_TYPE": "Torch"})
```

The `NCOMPASS_CACHE_DIR` env variable needs to point to the directory on the remote modal machine
that contains the `.cache` directory which is copied from the local directory generated by the
VSCode extension. The code which does this copy in `torch_profiling_example.py` is:
```python
ncompass_local_tracepoint_config = \
        Path(f"{os.getcwd()}/.cache/ncompass/profiles/.default/Torch/current/config.json")
ncompass_remote_tracepoint_config = "/config/.cache/ncompass/profiles/.default/Torch/current/config.json"
if ncompass_local_tracepoint_config.exists(): 
    image = image.add_local_file(ncompass_local_tracepoint_config, 
                                 ncompass_remote_tracepoint_config)
```

The `NCOMPASS_PROFILER_TYPE` env var just sets what the sub-directory is after `.default/`, i.e. is
it `Torch` or `NVTX` as we support both Torch Records and NVTX markers.

### Post-processing the trace

Once the above setup code is done, you can run the profile and the SDK will automatiaclly inject
the markers into the code base. The generated trace will then have annotations that say
`user_annotated: ...` to mark the regions that you highlighted.

The code in the `profile()` function in `torch_profiling_example.py` is instrumented to save the
resulting trace to the `.traces` directory. However, there is another important step that the
nCompass SDK provides over default torch profiling - linking user annotations to the kernel 
timeline.

By default, Torch Record annotations don't show up in the GPU timeline on the trace, they show up
on the CPU side timeline on the trace. The nCompass SDK provides a
`link_user_annotation_to_kernels` function which reads the trace, finds the overlap between the
user annotation and CPU side CUDA API calls, finds the corresponding kernel launch that the API
call is linked to and displays the annotation on the GPU timeline. To see the effect of this, run
the code with and without the `--no-link` flag, i.e. 

```bash
modal run torch_profiling_example.py --no-link # all annotations won't show up on GPU timeline
modal run torch_profiling_example.py           # all annotations will show up on GPU timeline
```

The following code ensures this happens:

```python
with image.imports():
  from ncompass.trace.converters import link_user_annotation_to_kernels

...

if link_annotations:
    linked_trace = link_user_annotation_to_kernels(trace_path, verbose=verbose)
    with open(trace_path, 'w') as f:
        json.dump(linked_trace, f)
```

## Additional Resources

- **[nCompass VSCode Extension Documentation](https://docs.ncompass.tech)** - Complete guide to using the extension
- **[Modal Documentation](https://modal.com/docs)** - Complete Modal platform documentation
- **[Modal PyTorch Profiling Example](https://modal.com/docs/examples/torch_profiling)** - Original Modal example
- **[PyTorch Profiler Documentation](https://pytorch.org/docs/stable/profiler.html)** - Learn about PyTorch's profiling capabilities

## Support

For questions or issues:
- **nCompass Support**: Check the [Documentation](https://docs.ncompass.tech) or visit the [Community Forum](https://community.ncompass.tech)
- **GitHub Issues**: Open an issue on [GitHub](https://github.com/ncompass-tech/ncompass/issues)
- **Modal Support**: Check [Modal Documentation](https://modal.com/docs) or [Modal Community](https://modal.com/community)
