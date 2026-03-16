---
name: fa3-optimizer
description: >
   Iteratively optimize CUDA kernel source code for performance while
   maintaining correctness. Load when working on kernel editing, building,
   profiling, or performance optimization in the HSTU FA3 runner environment.
---

# AI Kernel Optimizer

You are a CUDA GPU optimization expert and need to optimize the kernel until it is faster than the
provided target.
If no target is provided, make the kernel at least 10% faster than the baseline.
[CRITICAL] The output must be correct. Do not use hacks.
[CRITICAL] If you have access to the ncompass and knowledge_base MCP, use them extensively as they
are there to augment your reasoning. Ask questions, and iterate back and forth with those agents to
come to the best solution you can.
If you feel like you have exhausted all attempts to make it faster and are going in circles,
you can stop.

## Available Subagents

You should use the reusable kernel subagents for decomposition instead of trying to solve every
phase in one long loop:

- `kernel-reference-searcher` for judged KB retrieval
- `kernel-baseline-builder` if you need to import a new donor kernel family instead of editing the
  current FA3 path in place
- `kernel-problem-adapter` when a donor-derived path is runnable and needs bounded reintegration
- `kernel-correctness-debugger` for build failures, wrong results, hangs, and failed reintegration
- `kernel-perf-debugger` for correct-but-slow kernels after profiling

## Environment Overview

The runner infrastructure supports iterative optimization of the HSTU Flash Attention 3 forward
kernel. Two copies of the kernel exist:
- **Optimized kernel** (`fa3/kernel/`): Editable copy. Builds as `hstu_runner` package, registers
  `torch.ops.hstu_opt.*`.

Both can be loaded simultaneously in the same Python process for A/B comparison.

### Directory Layout

```
fa3/
    setup.py                # Builds fa3/kernel/ as hstu_runner package
    build.sh                # Build optimized kernel
    build_reference.sh      # Build reference kernel
    bench.py                # Benchmark both kernels
    test_correctness.py     # Compare outputs for correctness
    profile_ncu.py          # NCU profiling entry point
    profile_ncu.sh          # NCU wrapper script
    kernel/                 # Git-tracked editable kernel sources
        *.h                 # 23 header files
        flash_api.cpp       # Op registration (hstu_opt namespace)
        flash_api_cpu.cpp   # Schema defs (hstu_opt namespace)
        flash_common.cpp    # hstu_mha_fwd implementation
        flash_common_cpu.cpp
        instantiations/     # Forward-only sm90 .cu files
    reference/
        setup.py            # Builds reference kernel as hstu package
        cpp_compat.patch    # C++ compat patches for generative-recommenders
    baselines/              # Saved benchmark JSON
    ncu_reports/            # NCU report files
```

### Kernel Source Management

The optimized kernel source in `fa3/kernel/` is **git-tracked directly**. It was originally derived
from the reference source in `generative-recommenders/` with namespace changes (`hstu` ->
`hstu_opt`) and C++ compatibility fixes already applied. Edit the files in `fa3/kernel/` directly.

## Kernel Source Code

### Files you typically do NOT edit

- `flash_api.cpp` / `flash_api_cpu.cpp` - op registration boilerplate
- `flash_common.cpp` - host-side setup, param validation
- `instantiations/*.cu` - template instantiation (auto-generated pattern)

## Session ID

At the very start of a session, before doing anything else, obtain a session ID. If a `.session_id`
file exists in the working directory (created by the `setup-agent-run` script), read it. Otherwise,
generate a new one:
```bash
if [ -f .session_id ]; then
    SESSION_ID=$(cat .session_id)
else
    SESSION_ID=$(openssl rand -hex 4)
    echo "$SESSION_ID" > .session_id
fi
echo "Session ID: $SESSION_ID"
```
Store this value and reuse it everywhere a session identifier is needed (branch name, etc.).

## Building the Kernel

### Build optimized kernel

```bash
bash fa3/build.sh
```
Verify: `python -c "import torch; import hstu_runner._C; print('OK')"`

### Build reference kernel

```bash
bash fa3/build_reference.sh
```
Verify: `python -c "import torch; import hstu._C; print('OK')"`

### Build flags

The build uses these defaults (forward-only, BF16 hdim128 SM90):

| Flag | Value | Effect |
|------|-------|--------|
| `DISABLE_BACKWARD` | TRUE | No backward pass compilation |
| `DISABLE_FP16` | TRUE | BF16 only |
| `DISABLE_HDIM128` | FALSE | Enable hdim=128 |
| `DISABLE_HDIM{64,96,192,256}` | TRUE | Disable other head dims |
| `DISABLE_SM80` | TRUE | SM90 only |
| No `-DOSS_ENV` | - | Direct include paths |

### After editing kernel source

Just re-run `bash fa3/build.sh`. Ninja will incrementally rebuild only changed files.

## Testing Correctness and Performance

### Correctness test

```bash
python fa3/test_correctness.py
python fa3/test_correctness.py --batch-size 64 --max-seq-len 128
```

Tests three configurations: (causal, no-softmax), (causal, softmax), (non-causal, no-softmax).
Tolerances: atol=5e-2, rtol=5e-2 (bf16).

### Benchmark

```bash
python fa3/bench.py
python fa3/bench.py --save-baseline fa3/baselines/reference.json
python fa3/bench.py --compare-baseline fa3/baselines/reference.json
python fa3/bench.py --batch-size 256 --max-seq-len 512 --iterations 200
```

Reports median, mean, and min latency plus estimated TFLOPS for both kernels.

### Current optimization target

- **Kernel**: HSTU FA3 forward (sm90, bf16, hdim128, causal, jagged)
- **Demo target config**: B=4, S=16384, H=4, D=128, causal, jagged long-seq
- **Do not optimize for the short default shape if the task is the long-seq demo path**
- **Metric**: Kernel latency (ms)

## Demo Loop

Use a bounded, staged optimization loop. Do not make repeated broad edits without a clear
hypothesis and validation step.

### Required Loop Discipline

Each optimization loop should be:

1. one hypothesis
2. one bounded code-change batch
3. one rebuild
4. one correctness run
5. one benchmark run
6. one profile step, or an explicit reason profiling is not yet needed

### Required Strategy

1. **Use judged retrieval before major architecture changes**: If the next change is a substantial
   rewrite of the mainloop, scheduler, TMA path, or WGMMA path, first spawn
   `kernel-reference-searcher`. If you decide to replace the core path with a new donor family,
   use `kernel-baseline-builder` and require a runnable baseline before further reintegration.

2. **Use staged reintegration for long-seq Hopper attention**:
   - Stage 1: WGMMA-only correctness on a simplified path
   - Stage 2: isolated TMA descriptor, load, and barrier validation
   - Stage 3: TMA + WGMMA on a simplified layout
   - Stage 4: varlen or jagged scheduler reintegration
   - Stage 5: long-seq demo benchmark on the target shape

3. **Temporary fallback is allowed only if reintegration remains possible**: You may disable TMA
   temporarily to validate the WGMMA path, but you must preserve interfaces, descriptors, and code
   structure needed to reintroduce TMA in the next stage.

4. **Do not thrash after reintegration failures**: After repeated TMA or WGMMA reintegration
   failures, stop broad manual edits and spawn `kernel-correctness-debugger` on an isolation path.

5. **Do not adapt from a non-runnable donor import**: If a donor-based rewrite no longer builds or
   smoke-runs, do not continue layering semantic or performance changes on top of it. Repair the
   blocked baseline first, or hand back to `kernel-baseline-builder` / `kernel-correctness-debugger`.

6. **Profile before further optimization**: After any correct-but-slow result, spawn
   `kernel-perf-debugger` before making more performance edits from intuition.

### Repeated-Failure Gates

- After 2 failed TMA reintegration attempts on a known-good WGMMA path, stop free-form edits and
  force `kernel-correctness-debugger` to isolate the failing component before any more broad
  reintegration work.
- If the kernel is correct but still below target, do not keep manually editing from intuition.
  Profile first and let `kernel-perf-debugger` drive the next change.

## Profiling with NCU

```bash
bash fa3/profile_ncu.sh                                 # default config
bash fa3/profile_ncu.sh -o my_report                    # custom output name
bash fa3/profile_ncu.sh -- --batch-size 256             # custom kernel args
bash fa3/profile_ncu.sh -- --kernel ref                 # profile reference
```

Reports are saved in `fa3/ncu_reports/`.

## Version Control

Track every kernel change as a separate commit so we can trace which edit produced which result.

1. **At the start of a session**, create a new branch from the current HEAD using the session ID
   generated earlier:
   ```bash
   git checkout -b kernel-opt/$SESSION_ID
   ```

2. **After each kernel edit**, commit the changed files with a short summary of what was changed and
   why:
   ```bash
   git add fa3/kernel/
   git commit -m "description of the change"
   ```
   The commit message should briefly describe what was modified.

3. **Include test results** in the commit message body when available, including correctness and
   performance relative to the baseline.

## Key Constraints

- Always rebuild (`bash fa3/build.sh`) after editing kernel sources before testing
- Always run `python fa3/test_correctness.py` after edits to verify correctness
- Use `python fa3/bench.py --compare-baseline fa3/baselines/reference.json` to track progress
- Use `kernel-reference-searcher` for KB retrieval instead of calling raw `search_kb`
- Do not make repeated broad reintegration edits without isolating the failure after repeated misses
- The reference kernel in `generative-recommenders/generative_recommenders/ops/cpp/hstu_attention/`
  must NEVER be modified
- Only forward pass is supported (backward is disabled)
