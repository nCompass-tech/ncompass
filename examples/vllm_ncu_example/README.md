# vLLM Profiling with NCU and Nsys

This example demonstrates how to profile vLLM inference using NVIDIA Nsight Compute (NCU) for kernel-level metrics and Nsight Systems (Nsys) for timeline analysis.

## What This Example Does

- **NCU Profiling**: Generate detailed CSV reports with per-kernel GPU metrics (timing, memory bandwidth, FLOPs)
- **Nsys Profiling**: Create timeline traces for visualizing the execution flow
- **Torch Profiling**: Use vLLM's built-in PyTorch profiler
- **NVTX Markers**: Optionally add markers via the nCompass VSCode extension (no code changes)

## Prerequisites

- **Python 3.10+**
- **NVIDIA GPU** with CUDA support
- **Docker** (recommended) or local installation of:
  - NVIDIA Nsight Compute (`ncu`)
  - NVIDIA Nsight Systems (`nsys`)
- **HuggingFace Token** (for gated models)

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

### Step 3: Check NCU Privileges (Host Machine)

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

### Step 4: Run Profiling (Inside Container)

```bash
# NCU profiling (generates CSV with kernel metrics)
python main.py --ncu --model Qwen/Qwen2.5-0.5B

# Nsys profiling (generates timeline trace)
ncompass profile --no-sudo -- python main.py --nsys --model Qwen/Qwen2.5-0.5B

# Torch profiling
VLLM_TORCH_PROFILER_DIR=.torch_traces python main.py --torch --model Qwen/Qwen2.5-0.5B
```

## Profiling Modes

### NCU Profiling (Kernel-Level Metrics)

NCU provides detailed per-kernel metrics including:
- GPU time duration
- Memory throughput (DRAM, L2, L1)
- Floating-point operations (FP32, FP16, FP64)
- SM utilization

```bash
# Basic NCU profiling
python main.py --ncu --model Qwen/Qwen2.5-0.5B

# Filter specific kernels
python main.py --ncu --model Qwen/Qwen2.5-0.5B --kernel-name "regex:.*gemm.*"
```

Output: CSV file in `.ncu_traces/` directory

### Nsys Profiling (Timeline Analysis)

Nsys creates timeline traces showing the execution flow across CPU and GPU.

```bash
# Run with ncompass CLI (handles nsys invocation)
ncompass profile --no-sudo -- python main.py --nsys --model Qwen/Qwen2.5-0.5B
```

Output: `.nsys-rep` file that can be viewed in Nsight Systems GUI or converted to Chrome trace

### Torch Profiling

Uses vLLM's built-in PyTorch profiler.

```bash
export VLLM_TORCH_PROFILER_DIR=.torch_traces
python main.py --torch --model Qwen/Qwen2.5-0.5B
```

Output: PyTorch trace files in the specified directory

## Adding NVTX Markers (Optional)

For more targeted profiling, add NVTX markers using the nCompass VSCode extension:

1. Open vLLM source files in VSCode with the nCompass extension
2. Highlight code regions to profile
3. Press `Ctrl+.` and select "Add Region to Profile" > "NVTX"
4. Set environment variables:
   ```bash
   export NCOMPASS_CACHE_DIR=<path to directory containing .cache/>
   export NCOMPASS_PROFILER_TYPE=NVTX
   ```
5. Run profiling as usual

See: [Runtime Injection Tutorial](https://www.loom.com/share/1e526ff43fa44bfcaebe18a6f56052ff)

## CLI Options

| Option | Description |
|--------|-------------|
| `--ncu` | Profile with NVIDIA Nsight Compute |
| `--nsys` | Run with NVTX markers (use with `ncompass profile`) |
| `--torch` | Profile with vLLM's Torch profiler |
| `--model` | HuggingFace model name (default: Qwen/Qwen2.5-0.5B) |
| `--output`, `-o` | Base name for output files |
| `--kernel-name`, `-k` | Regex filter for kernel names (NCU only) |
| `--nvtx-include` | NVTX range filter (NCU only) |

## nc_pkg.py Options

| Option | Description |
|--------|-------------|
| `--build` | Build the Docker image |
| `--run` | Run the container (interactive shell) |
| `--down` | Stop and remove the container |
| `--exec <cmd>` | Execute a command in the container |
| `--ncompass-dir` | Path to ncompass directory (required for --run/--exec) |
| `--vllm-src` | Path to vllm source or wheel file |
| `--no-exec` | Don't auto-exec into container after --run |

## File Structure

```
vllm_ncu_example/
├── Dockerfile           # Docker image with NCU, Nsys, and dependencies
├── docker-compose.yaml  # Docker compose configuration
├── nc_pkg.py           # Docker helper script
├── main.py             # Main profiling script
├── env_config.yaml.example  # Environment config template
└── README.md           # This file
```

## Output Directories

- `.ncu_traces/` - NCU CSV reports
- `.nsys_traces/` - Nsys report files (when using ncompass profile)
- `.torch_traces/` - PyTorch profiler traces

## Troubleshooting

### NCU Permission Denied

If NCU fails with permission errors, ensure you've set the kernel module parameter:
```bash
sudo modprobe nvidia NVreg_RestrictProfilingToAdminUsers=0
```

### Container Can't Access GPU

Ensure you have the NVIDIA Container Toolkit installed and Docker is configured to use it.

### HuggingFace Model Access

For gated models, ensure your `HF_TOKEN` is set in `env_config.yaml`.

## Additional Resources

- [NVIDIA Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)
- [NVIDIA Nsight Systems Documentation](https://docs.nvidia.com/nsight-systems/)
- [vLLM Documentation](https://docs.vllm.ai/)
- [nCompass Documentation](https://docs.ncompass.tech)
