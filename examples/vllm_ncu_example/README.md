# vLLM Profiling with NCU and Nsys

This example demonstrates how to profile vLLM inference using NVIDIA Nsight Compute (NCU) for kernel-level metrics and Nsight Systems (Nsys) for timeline analysis.

## What This Example Does

- **NCU Profiling**: Generate detailed CSV reports with per-kernel GPU metrics (timing, memory bandwidth, FLOPs)
- **Nsys Profiling**: Create timeline traces for visualizing the execution flow
- **Torch Profiling**: Use vLLM's built-in PyTorch profiler

## Prerequisites

- **Python 3.10+**
- **NVIDIA GPU** with CUDA support
- **Docker** (recommended) or local installation of:
  - NVIDIA Nsight Compute (`ncu`)
  - NVIDIA Nsight Systems (`nsys`)
- **HuggingFace Token** (for gated models)
- [nCompass VSCode Extension](https://docs.ncompass.tech)

## Check NCU Privileges (Host Machine)

On your **host** machine, verify NCU has proper privileges:

```bash
cat /proc/driver/nvidia/params | grep RmProfilingAdminOnly
```

If the output shows `RmProfilingAdminOnly: 1`, run:

```bash
sudo rmmod nvidia_uvm
sudo rmmod nvidia_drm
sudo rmmod nvidia_modeset
sudo rmmod nvidia
sudo modprobe nvidia NVreg_RestrictProfilingToAdminUsers=0
```

## Quick Start (Docker - Recommended)

### Step 1: Setup Environment Config

```bash
cp env_config.yaml.example env_config.yaml
# Edit env_config.yaml and add your HF_TOKEN
```

### Step 2: Build and Run Container

```bash
# Build the Docker image
python nc_pkg.py --build

# Run the container (interactive shell)
python nc_pkg.py --run --ncompass-dir ../../..

# Optional: Install vllm from source
python nc_pkg.py --run --ncompass-dir ../../.. --vllm-src /path/to/vllm
```

### Step 4: Run Profiling (Inside Container)

```bash
# NCU profiling (with VSCode extension markers)
NCOMPASS_CACHE_DIR=../ NCOMPASS_PROFILER_TYPE=NVTX ncompass profile --ncu -- python main.py --nvtx

# Nsys profiling (uses manual ncompass_nsys_range marker in code)
ncompass profile --nsys -- python main.py --nvtx --model Qwen/Qwen2.5-0.5B

# Torch profiling
VLLM_TORCH_PROFILER_DIR=.torch_traces python main.py --torch --model Qwen/Qwen2.5-0.5B
```

## Quick Start (Local)
You can use the commands in `nc_pkg.py` as a guideline to how to setup the environment locally, but
before running you need to have the following installed:
- Packages in `requirements.txt`
- ncompass installed (from source using `pip install ../../` or otherwise)
- vllm installed (from source or otherwise)
- Pre-requisites mentioned above

Once all of these are installed, you can just run the same Step 4 above or based on the details
below to run profiling. 

## Profiling Modes

### NCU Profiling (Kernel-Level Metrics)

NCU provides detailed per-kernel metrics including:
- GPU time duration
- Memory throughput (DRAM, L2, L1)
- Floating-point operations (FP32, FP16, FP64)
- SM utilization

**How NCU range selection works**: The nCompass VSCode extension allows you to add NVTX markers to code without modifying it. These markers control which code regions NCU profiles. The default filter is `regex:user_annotated:.*/`.

**Note:** ncu profiling is expensive and to run it on a large code base like vLLM, you want to
restrict which regions (and sometimes kernels) you want to actually collect for. Because of this,
the ncompass SDK wrapping for ncu forces you to have constrained the program in some way using our
SDK. The constraining capabilities will improve over time.

This is why, to use the ncu profiling example, you need to first use our SDK to add a NVTX
profiling marker - [Tutorial](https://docs.ncompass.tech/Adding-benchmark-debug-code-without-editing-the-codbase-2e3097a5a430801d8c5cca6d0194e99e)

1. Open vLLM source files in VSCode with the nCompass extension
2. Highlight code regions to profile
3. Press `Ctrl+.` and select "Add Region to Profile" > "NVTX"
4. Run the following commands
```bash
# Set environment variables for nCompass NVTX injection
export NCOMPASS_CACHE_DIR=<path to directory containing .cache/>
export NCOMPASS_PROFILER_TYPE=NVTX

# Run NCU profiling
ncompass profile --ncu -- python main.py --nvtx

# Override default options via extra args
ncompass profile --ncu -- --kernel-name="regex:.*gemm.*" -- python main.py --nvtx
```

Output: `.ncu-rep` file and CSV in `.nsys_traces/<timestamp>/` directory

### Nsys Profiling (Timeline Analysis)

Nsys creates timeline traces showing the execution flow across CPU and GPU.

**How Nsys range selection works**: Unlike NCU, Nsys capture ranges require a **manual NVTX marker in the code**. The marker `nvtx.annotate(message="ncompass_nsys_range")` in `main.py` defines the capture region:

```python
with nvtx.annotate(message="ncompass_nsys_range"):
    outputs = llm.generate([test_prompt], sampling_params)
```

```bash
# Run with ncompass CLI
ncompass profile --nsys -- python main.py --nvtx --model Qwen/Qwen2.5-0.5B
```

Output: `.nsys-rep` file that can be viewed in Nsight Systems GUI or converted to Chrome trace

Adding markers using the nCompass VSCode Extension will show up in the trace itself. The `annotate`
command mentined above is only required to enable and disable nsys profiling.

### Torch Profiling

Uses vLLM's built-in PyTorch profiler.

```bash
export VLLM_TORCH_PROFILER_DIR=.torch_traces
python main.py --torch --model Qwen/Qwen2.5-0.5B
```

Output: PyTorch trace files in the specified directory

## Additional Resources

- [nCompass Documentation](https://docs.ncompass.tech)
- [NVIDIA Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)
- [NVIDIA Nsight Systems Documentation](https://docs.nvidia.com/nsight-systems/)
