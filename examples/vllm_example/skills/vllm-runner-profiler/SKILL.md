---
name: vllm-runner-profiler
description: >
  Docker setup, running, and profiling vLLM inference servers. Load when working inside
  the vllm_example directory, running commands in the vLLM docker container, profiling
  with nsys or ncu, collecting .nsys-rep or .ncu-rep traces, benchmarking inference
  latency, or setting up profiling scenarios for attention/MoE/decode analysis.
  This includes when the example is copied into the runs/ directory.
---

# vLLM Runner & Profiler

## Docker Setup

The example directory has a `nc_pkg.py` file. Two steps are required:

### Step 1: Initial setup (once)
```
python3 nc_pkg.py --setup --docker-dir ../docker --wheel wheels/<wheel_file>
```
Where `<wheel_file>` is whichever `.whl` file exists in the `wheels/` directory.
If multiple files exist in the `wheels/` directory, ask the user which wheel to use.
If running from a `runs/` copy, the `--docker-dir` path should point to the shared
docker infrastructure (e.g. `../../../ncompass/examples/docker/`).

### Step 2: Build and run container
```
python3 nc_pkg.py --build --run --ncompass-dir ../../../ncompass --wheel <wheel_file>
```
Where `--ncompass-dir` points to the ncompass root and `<wheel_file>` matches the
wheel used in setup.

### Nightly builds

To build and run with the latest vLLM nightly, use `--nightly` (no `--wheel` needed):
```
python3 nc_pkg.py --build --nightly --run --ncompass-dir ../../../ncompass
```
This automatically:
1. Builds the Docker image with Python 3.12, cu130 PyTorch index, and system cuBLAS `LD_PRELOAD`
2. Fetches the latest nightly wheel index from `wheels.vllm.ai/nightly/cu130`
3. Clones vLLM source at the matching git commit into `vllm_src/`
4. Downloads the nightly wheel inside the container and installs vLLM editably with precompiled binaries

### Running commands in the container

This will start a container called `vllm_example-vllm_example-1`.

First check if this container is running. If it is, just run commands inside the
container using the `docker exec ...` command. The directory structure inside the
docker container mirrors that outside.

If not, start it first and then run commands inside it.

---

Two profiling approaches are available. Choose based on what you need:

| Use case                                        | Approach                            | Why                                                                                                                        |
|-------------------------------------------------|-------------------------------------|----------------------------------------------------------------------------------------------------------------------------|
| Multiple input/output scenarios at batch size 1 | `vllm_profiler.py` (server mode)    | Loads model once, runs all scenarios via `/start_profile`/`/stop_profile` cycles — much faster than reloading per scenario |
| Batch size > 1 (deterministic)                  | `vllm bench latency` (offline mode) | Controls exact batch size via `--batch-size`. Server mode can't guarantee batch size since the scheduler decides           |
| NCU kernel profiling                            | `vllm bench latency` (offline mode) | Simpler to target specific kernel launches with `--launch-skip`/`--launch-count` in a single-process setup                 |
| Comparing backends at any batch size            | `vllm bench latency`                | Separate invocations with different `--attention-backend` flags                                                            |

---

## Approach 1: `vllm_profiler.py` — Server mode (batch size 1, multiple scenarios)

Best for sweeping input/output token configurations at batch size 1. The model loads once and stays
loaded across all scenarios. Each scenario is a separate cudaProfiler capture range in the same
trace file.

### Script location

`skills/vllm-runner-profiler/vllm_profiler.py` (relative to the example/run directory)

### How it works

1. Starts `vllm serve` under `ncompass profile --nsys` (with `--capture-range-end=repeat`)
2. Model loads once, server starts
3. Warmup requests (outside capture)
4. For each scenario: `/start_profile` → send request(s) → `/stop_profile`
5. All scenarios captured as separate ranges in one `.nsys-rep`
6. Server shuts down once at the end

### Examples

**Multiple scenarios, one model load:**
```
python skills/vllm-runner-profiler/vllm_profiler.py nsys \
    --model Qwen/Qwen3.5-35B-A3B-FP8 \
    --scenarios short_decode:10:256 long_prefill:2048:1 medium:512:128 \
    --output sweep
```
Format: `name:prompt_tokens:max_tokens[:batch_size]`

**Single scenario:**
```
python skills/vllm-runner-profiler/vllm_profiler.py nsys \
    --model Qwen/Qwen3.5-35B-A3B-FP8 \
    --prompt-tokens 50 --max-tokens 256 --output decode_heavy
```

**With custom vLLM args (e.g. different attention backend):**
```
python skills/vllm-runner-profiler/vllm_profiler.py nsys \
    --model Qwen/Qwen3.5-35B-A3B-FP8 \
    --vllm-args "--moe-backend triton --attention-backend FLASHINFER" \
    --scenarios short_decode:10:256 long_prefill:2048:1 \
    --output flashinfer_sweep
```

---

## Approach 2: `vllm bench latency` — Offline mode (batch > 1, NCU, deterministic)

Best for deterministic batch-size control and NCU profiling. Loads the model in-process via vLLM's
`LLM` API and calls `generate()` with an exact number of prompts — no server scheduling. The
`--profile` flag triggers `cudaProfilerStart/Stop` for capture.

The tradeoff is that each invocation reloads the model, so sweeping multiple configs is slower.

### Key flags
| Flag | Default | Purpose |
|------|---------|---------|
| `--batch-size N` | 8 | Exact number of prompts per generate() call |
| `--input-len N` | 32 | Input tokens per prompt |
| `--output-len N` | 128 | Output tokens per prompt |
| `--num-iters N` | 30 | Profiled iterations |
| `--num-iters-warmup N` | 10 | Warmup iterations (not profiled) |
| `--profile` | False | Enable cudaProfilerStart/Stop for nsys/ncu capture |

All model/engine flags work too (`--attention-backend`, `--moe-backend`, etc.).

### nsys examples

**Batch of 32, decode-heavy:**
```
ncompass profile --nsys -o traces/bs32_decode -- \
    vllm bench latency --model Qwen/Qwen3.5-35B-A3B-FP8 \
        --batch-size 32 --input-len 50 --output-len 256 \
        --num-iters 1 --num-iters-warmup 3 --profile
```

**Batch of 256, prefill-heavy:**
```
ncompass profile --nsys -o traces/bs256_prefill -- \
    vllm bench latency --model Qwen/Qwen3.5-35B-A3B-FP8 \
        --batch-size 256 --input-len 4096 --output-len 2 \
        --num-iters 1 --num-iters-warmup 3 --profile
```

**Compare attention backends (batch 32):**
```
ncompass profile --nsys -o traces/bs32_flash_attn -- \
    vllm bench latency --model Qwen/Qwen3.5-35B-A3B-FP8 \
        --batch-size 32 --input-len 4096 --output-len 256 \
        --num-iters 1 --num-iters-warmup 3 --profile

ncompass profile --nsys -o traces/bs32_flashinfer -- \
    vllm bench latency --model Qwen/Qwen3.5-35B-A3B-FP8 \
        --attention-backend FLASHINFER \
        --batch-size 32 --input-len 4096 --output-len 256 \
        --num-iters 1 --num-iters-warmup 3 --profile
```

### NCU examples

**Roofline of a specific kernel:**
```
ncu --profile-from-start off --target-processes all \
    --set roofline --kernel-name "device_kernel" \
    --launch-skip 3 --launch-count 1 \
    -o traces/attn_roofline -f \
    vllm bench latency --model Qwen/Qwen3.5-35B-A3B-FP8 \
        --batch-size 1 --input-len 2048 --output-len 1 \
        --num-iters 1 --num-iters-warmup 3 --profile
```

**Full metrics for a GEMM kernel:**
```
ncu --profile-from-start off --target-processes all \
    --set full --kernel-name "cutlass" \
    --launch-skip 5 --launch-count 1 \
    -o traces/gemm_full -f \
    vllm bench latency --model Qwen/Qwen3.5-35B-A3B-FP8 \
        --batch-size 32 --input-len 1024 --output-len 128 \
        --num-iters 1 --num-iters-warmup 3 --profile
```

---

## Why `ncompass profile --nsys` instead of raw `nsys profile`

`ncompass profile --nsys` wraps `nsys profile` with the right defaults:
`--capture-range=cudaProfilerApi`, `--capture-range-end=repeat`, `--cuda-graph-trace=node`,
`--trace-fork-before-exec=true`, `--trace=cuda,nvtx,osrt,cudnn,cublas`. You can override any of
these by passing extra flags. Use `ncompass profile --nsys` unless you have a specific reason to
call nsys directly.

## Sending workloads to a running server with `vllm bench serve`

If you already have a vLLM server running (standalone or under a profiler) and just need to send
traffic to it, use `vllm bench serve --dataset-name random`:

| Flag | Default | Purpose |
|------|---------|---------|
| `--num-prompts N` | 1000 | Total requests to send |
| `--num-warmups N` | 0 | Warmup requests (not measured) |
| `--random-input-len N` | 1024 | Input tokens per request |
| `--random-output-len N` | 128 | Output tokens per request |
| `--request-rate R` | inf | Requests/sec (`inf` = send all at once) |
| `--max-concurrency N` | None | Max concurrent in-flight requests |
| `--base-url URL` | None | Server URL (use instead of --host/--port) |
| `--ignore-eos` | False | Force exact output length (don't stop at EOS) |

Note: batch size is NOT deterministic in server mode — the scheduler decides how to batch
concurrent requests. Use `vllm bench latency` for deterministic batch profiling.

## Important

If these tools don't cover your exact profiling needs — different profiler flags, custom
workload patterns, or a scenario not anticipated here — decide at runtime how to configure the
profiling. These are starting points, not constraints. You can also create arbitrary scripts that
you want. Create them in `.runtime_scripts/`

## After profiling

Once you have traces, invoke the `performance-optimization-strategizer` skill for analysis guidance.
