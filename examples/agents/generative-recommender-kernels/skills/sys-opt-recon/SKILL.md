---
name: sys-opt-recon
description: >
  Collect structured facts about a GPU inference pipeline: codebase map,
  baseline measurement, nsys trace, ncompass analysis, and optimization
  barriers. Outputs to .agent/notes/recon/ for downstream planning.
---

# System Optimization — Reconnaissance

Mechanical data collection phase. Your job is to gather facts, not to
form opinions or hypotheses. The planning skill will synthesize later.

## Prerequisites

- The environment is set up (kernel built, dependencies installed).
- `bench.py`, `test_correctness.py`, and `run_model.py` are functional.
- nsys is available on the system.
- `.agent/notes/` directory exists.

## Output

All outputs go to `.agent/notes/recon/`. Create this directory first:
```bash
mkdir -p .agent/notes/recon
```

---

## Step 1: Identify the Forward Pass Entry Point

Read `run_model.py` (or equivalent model runner) to find:
- How the model is created and configured
- How a batch is generated
- What the forward pass entry point is (the function that gets timed)
- What the optimization mode system looks like (how modes are discovered/applied)

Write `.agent/notes/recon/entry_point.md`:
```markdown
# Forward Pass Entry Point

## Model creation
- [how the model is instantiated, key config values]

## Batch structure
- [input types, shapes, any special tensor types like KJT]

## Forward pass signature
- [function(input_a, input_b) -> output]
- [which model method this calls]

## Optimization mode system
- [how bench.py discovers and applies optimization modes]
- [the apply() contract — what it receives, what it must return]
```

## Step 2: Map the Call Graph

Starting from the forward pass entry point, trace the call chain through
the model code. Read each method in sequence. Do not skip any layer of the
call chain — you need to understand the full decomposition.

Write `.agent/notes/recon/call_graph.md`:
```markdown
# Forward Pass Call Graph

## Call chain
forward()
  -> phase_1()  [file:line — brief description]
    -> sub_step_a()  [file:line]
    -> sub_step_b()  [file:line]
  -> phase_2()  [file:line — brief description]
    -> ...
  -> phase_3()  [file:line — brief description]
    -> ...

## Phase boundaries
- [Where does preprocessing end and compute begin?]
- [Where does compute end and postprocessing begin?]
- [Which phases are data-dependent vs fixed-structure?]
```

## Step 3: Baseline Measurement

Run the benchmark to establish the number to beat:
```bash
python model_runner/bench.py --max-seq-len 256 \
  --save-baseline model_runner/baselines/ref.json \
  --warmup-iters 10 --bench-iters 20
```

Read `model_runner/baselines/ref.json` and record the baseline median.

## Step 4: Collect nsys Trace

If ncompass is available, use `generate_ncu_profiling_command` or consult
ncompass for the recommended profiling flags for this workload.

Otherwise, use the benchmark harness's built-in `--profile` mode which
integrates with cudaProfilerApi capture:

```bash
gpu-lock nsys profile \
  -o .agent/notes/recon/baseline_trace \
  --force-overwrite=true \
  python -u model_runner/bench.py --max-seq-len 256 --mode baseline \
    --profile --bench-iters 3
```

Consult `/system-optimizer-reference` for the full set of recommended nsys
flags if ncompass is not available.

## Step 5: Analyze Trace via ncompass

If ncompass is available, run these analyses on the trace:

### 5a: Kernel timing and launch overhead
```
mcp__ncompass__analyze_nsys_sqlite:
  message: "Analyze this baseline inference trace. Report:
    1. Top 20 kernels by total GPU time (name, count, total, avg, % of total)
    2. Overall GPU utilization (kernel time vs wall time)
    3. Total kernel launches per forward pass
    4. CUDA API overhead (cudaLaunchKernel, synchronization calls)
    5. Memory operations (H2D, D2H, D2D) and their overhead
    6. Largest gaps between consecutive kernels (>50us)"
  trace_files: [".agent/notes/recon/baseline_trace.nsys-rep"]
```

### 5b: Iteration structure and variability
```
mcp__ncompass__analyze_nsys_patterns:
  message: "Detect repeating patterns (forward pass iterations).
    Report per-iteration timing, variability, and any outlier iterations."
  trace_files: [".agent/notes/recon/baseline_trace.nsys-rep"]
```

If ncompass is NOT available, extract what you can from nsys stats:
```bash
nsys stats --report cuda_gpu_trace .agent/notes/recon/baseline_trace.nsys-rep \
  --format csv --output . 2>/dev/null | head -50
```

Write `.agent/notes/recon/trace_analysis.md` with the full ncompass output
and a structured summary:
```markdown
# Trace Analysis

## Summary
- Wall time per iteration: X.XX ms
- GPU kernel time per iteration: X.XX ms
- GPU utilization: XX%
- Total kernel launches per iteration: N
- Dominant cost: [GPU compute | CPU overhead | sync stalls | launch overhead]

## Top kernels (by total GPU time)
| Kernel | Count | Total (ms) | Avg (us) | % GPU time |
|--------|-------|------------|----------|------------|
| ...    | ...   | ...        | ...      | ...        |

## Significant gaps (>50us between kernels)
| Gap (us) | After kernel | Before kernel |
|----------|-------------|---------------|
| ...      | ...         | ...           |

## Memory operations
- [H2D, D2H, D2D summary]

## Sync points observed in trace
- [cudaDeviceSynchronize, cudaStreamSynchronize, etc.]
```

## Step 6: Identify Optimization Barriers (Trace-Driven)

This step is driven by the trace findings, not by a hardcoded checklist.

For each dominant cost center identified in Step 5:
1. Map it back to source code using the call graph from Step 2
2. In that source code, identify what constrains optimization

Common barrier categories (use as a reference, not a checklist):
- **CPU-GPU synchronization**: Host readbacks, data-dependent control flow
- **Dynamic shapes**: Sizes derived from tensor values at runtime
- **Non-capturable operations**: Ops that fail under graph capture or tracing
- **Custom/opaque operators**: Ops that compilers can't see through
- **Hot-path memory allocation**: Tensor creation inside the timed region
- **Framework dispatch overhead**: Deep call chains, dict construction, config lookups

For each barrier found, record:
- **Where**: file:line, function name
- **What**: the specific operation
- **Why it matters**: which optimization technique(s) it blocks
- **Severity**: is it on the critical path? How much time does it account for?

Write `.agent/notes/recon/barriers.md`:
```markdown
# Optimization Barriers

## Barrier 1: [descriptive name]
- **Location**: file.py:123, function_name()
- **Operation**: [what it does]
- **Blocks**: [CUDA graphs | torch.compile | operator fusion | ...]
- **Severity**: [critical path, ~X.X ms | minor, <0.1 ms]
- **Context**: [why this operation exists, what it computes]

## Barrier 2: ...
```

## Step 7: Record Metadata

Write `.agent/notes/recon/metadata.json`:
```json
{
  "timestamp": "<ISO>",
  "baseline_median_ms": 0.0,
  "gpu_utilization_pct": 0.0,
  "kernel_launches_per_iter": 0,
  "dominant_cost": "cpu_overhead|gpu_compute|sync_stalls|launch_overhead",
  "barrier_count": 0,
  "ncompass_available": true,
  "trace_path": ".agent/notes/recon/baseline_trace.nsys-rep"
}
```

---

## What NOT to do in this skill

- Do NOT propose optimizations or hypotheses
- Do NOT assess technique feasibility (that's `/sys-opt-plan`)
- Do NOT start implementing anything
- Do NOT editorialize in the findings — state facts, cite evidence

When all 7 steps are complete, tell the user that recon is done and
suggest running `/sys-opt-plan` to synthesize the findings into an
action plan.
