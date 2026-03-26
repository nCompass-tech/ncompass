---
name: system-optimizer
description: >
  Iteratively optimize the DLRM-v3 + HSTU end-to-end inference pipeline for
  system-level performance. Load when working on model-runner profiling,
  torch.compile integration, CUDA graph optimization, operator fusion, or
  launch overhead reduction.
---

# System-Level Optimizer

You are a GPU systems optimization expert. Your goal is to reduce end-to-end
iteration latency of the DLRM-v3 + HSTU inference pipeline by identifying and
fixing system-level bottlenecks — launch overhead, missing fusion, unnecessary
synchronization, poor CPU-GPU overlap, etc.

[CRITICAL] The output must remain correct. Every optimization must pass
`test_correctness.py` before benchmarking.

[CRITICAL] If you have access to **ncompass MCP** (trace analysis) and
**knowledge_bank MCP** (curated docs on torch.compile, CUDA graphs, Triton,
etc.), use them extensively — they are there to augment your reasoning.

[CRITICAL] Do NOT hard-code optimization strategies from prior knowledge.
Profile first, identify the bottleneck, then implement. If the knowledge_bank
MCP is available, search it for techniques that address the observed bottleneck.
Let the data guide you.

## Preflight

Before doing anything else, check which MCPs are configured. Try each one —
if a tool is not available (i.e. the tool doesn't exist), that's fine, skip it.
But if a tool IS available and fails when called, **abort the run** — a
configured-but-broken MCP means the environment is misconfigured.

1. **ncompass**: try calling `check_auth`.
   - Tool not found → no ncompass available, proceed without it.
   - Tool exists but returns an error → **abort the run**.
2. **knowledge_bank**: try calling `search_kb` with a trivial query (e.g. `"cuda graphs"`).
   - Tool not found → no KB available, proceed without it.
   - Tool exists but errors or returns zero results → **abort the run**.

Record which MCPs are available. This determines your workflow:
- **Both available**: full loop (profile → analyze via ncompass → search KB → implement)
- **ncompass only**: profile → analyze via ncompass → implement (no KB search)
- **KB only**: profile manually (read nsys output) → search KB → implement
- **Neither**: profile manually → implement using your own reasoning

## What this task is NOT

[CRITICAL] Read this carefully:

- You are **NOT** writing CUDA kernels. Do not write `.cu`, `.h`, or Triton kernel code.
- You are **NOT** modifying the HSTU attention implementation. The custom kernel is
  pre-built and loaded via `--kernel triton`. It is not your concern.
- You are **NOT** replacing existing kernels with hand-written alternatives.

Your job is **system-level optimization**: torch.compile, CUDA graphs, operator fusion,
launch overhead reduction, async CPU-GPU overlap, synchronization elimination. These are
Python-level changes in `model_runner/optimizations/`.

If you find yourself writing GPU kernel code, you have misunderstood the task. Stop and
re-read this section.

## Session ID

Obtain a session ID at the start. If a `.session_id` file exists (created by
the `setup-agent-run` script), read it. Otherwise, generate one:

```bash
if [ -f .session_id ]; then
    SESSION_ID=$(cat .session_id)
else
    SESSION_ID=$(openssl rand -hex 4)
    echo "$SESSION_ID" > .session_id
fi
echo "Session ID: $SESSION_ID"
```

Create a branch for this session:
```bash
git checkout -b system-opt/$SESSION_ID
```

## Environment Overview

The model runner infrastructure runs the full DLRM-v3 + HSTU inference pipeline
on a single GPU. A baseline forward pass exists alongside manual CUDA graph
capture. Your job is to add system-level optimizations on top.

### Directory Layout

```
model_runner/
    run_model.py              # Baseline model — DO NOT MODIFY
    profile_nsys.py           # Nsys wrapper — DO NOT MODIFY
    bench.py                  # Latency benchmark — DO NOT MODIFY
    test_correctness.py       # Correctness test — DO NOT MODIFY
    optimizations/            # YOUR editable directory
        __init__.py           # Mode discovery (DO NOT MODIFY)
        <your_mode>.py        # You create these
    baselines/                # Saved benchmark results (JSON)
    nsys_traces/              # Nsys trace outputs
```

### What you edit

Only files inside `model_runner/optimizations/`. Each optimization is a Python
module that exposes:

```python
def apply(model, batch, hstu_config, **kwargs) -> callable:
    """Return a callable(uih_features, candidates_features) -> model_output."""
```

The callable must return the same output format as the baseline model:
`(user_emb, item_emb, hidden, mt_target_preds)`.

### What you do NOT edit

- `run_model.py` — baseline model infrastructure
- `bench.py` — benchmark harness
- `test_correctness.py` — correctness harness
- `profile_nsys.py` — nsys wrapper
- `optimizations/__init__.py` — mode discovery

## Running Commands

### Save baseline

```bash
python model_runner/bench.py --save-baseline model_runner/baselines/ref.json
```

### Correctness test

```bash
python model_runner/test_correctness.py --mode <your_mode>
python model_runner/test_correctness.py --all
```

### Benchmark

```bash
python model_runner/bench.py --mode <your_mode>
python model_runner/bench.py --mode <your_mode> --compare-baseline model_runner/baselines/ref.json
```

### Profile with nsys

```bash
nsys profile \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --cuda-memory-usage=true \
  --cuda-graph-trace=node \
  -tcuda,nvtx \
  -o model_runner/nsys_traces/<name> \
  --force-overwrite=true \
  python -u model_runner/bench.py --mode <your_mode> --profile --bench-iters 3
```

For baseline profiling, use `--mode baseline`.

## Optimization Loop

```
0. Preflight — check which MCPs are available (see above)
1. Profile with nsys (baseline first, then optimized)
2. Analyze trace via ncompass MCP
   → Identify: launch overhead %, sync overhead %, idle gaps, kernel count, iteration structure
3. Search knowledge_bank MCP for techniques addressing the observed bottleneck
4. Formulate ONE hypothesis
5. Implement ONE optimization in model_runner/optimizations/
6. Run correctness test
7. Run benchmark (compare against baseline)
8. Profile the optimized path with nsys
9. Diff traces via ncompass MCP (analyze_nsys_diff: before vs after)
   → Quantify: kernel count change, launch overhead reduction, latency change
10. Git commit with results
11. If target not met → go to step 2 with the new trace
```

### Loop Discipline

Each iteration is:
1. One hypothesis
2. One bounded code change
3. One correctness run
4. One benchmark run
5. One profile + analysis, or an explicit reason profiling is not needed yet
6. One git commit

No stacking multiple untested optimizations. No intuition-driven edits without
profiling evidence.

## MCP Tools

| MCP | When to use |
|---|---|
| **ncompass** `check_auth` | Preflight |
| **ncompass** `analyze_nsys_sqlite` | Raw kernel timing, launch overhead, sync analysis |
| **ncompass** `analyze_nsys_perfetto` | Timeline view, GPU/CPU overlap, concurrency |
| **ncompass** `analyze_nsys_patterns` | Iteration structure, timing variability |
| **ncompass** `analyze_nsys_diff` | Before/after trace comparison |
| **ncompass** `consult` | Ambiguous or multi-domain questions |
| **knowledge_bank** `search_kb` | Find techniques for a specific bottleneck class |

Use the specialist ncompass tools when the task clearly maps to one domain.
Use `consult` only for ambiguous or cross-domain questions.

## Failure Gates

### torch.compile graph breaks
If dynamo reports graph breaks, read the log (`TORCH_LOGS="graph_breaks"`),
search the knowledge_bank MCP for the specific pattern, and apply a targeted
fix. After 2 failed attempts on the same graph break, search KB for alternative
approaches.

### CUDA graph capture failure
If capture fails, isolate the offending operation. The existing
`CUDAGraphDlrmHSTU` in `run_model.py` shows the pattern (precomputing `.item()`
results). Search KB troubleshooting docs before attempting fixes.

### Correctness regression
If outputs diverge beyond tolerance (atol=1e-3, rtol=1e-3), **revert
immediately**. Do not stack optimizations on a broken base. Diagnose first.

### Performance plateau
If the last 2 iterations showed <2% improvement, you must profile and analyze
via ncompass before making more changes. Do not guess.

## Version Control

Track every optimization as a separate commit.

```bash
git add model_runner/optimizations/
git commit -m "description of the change

Correctness: PASS/FAIL
Median latency: X.XXX ms (baseline: Y.YYY ms, speedup: Z.ZZx)
"
```

Include test results in the commit message body.

## Key Constraints

- Always run `test_correctness.py` after edits before benchmarking
- Always profile before making performance-driven edits
- Use `knowledge_bank` `search_kb` to find optimization techniques — do not rely
  solely on prior knowledge
- Do not modify files outside `model_runner/optimizations/`
- The custom HSTU kernel is always loaded (`--kernel triton`). System-level
  optimizations are layered around it, not replacing it
- Single-GPU only (multi-GPU is out of scope)
