---
name: system-optimizer-reference
description: >
  Reference material for system-level optimization: MCP tool table, nsys
  commands, note-taking schema, failure gates, and constraints. Load on
  demand when you need to look something up.
---

# System Optimizer — Reference

Load this skill when you need to look up commands, schemas, or tool usage.
Do not keep this loaded during the optimization loop — use `/system-optimizer`
for the loop itself.

## Target Workload

[CRITICAL] All commands MUST use `--max-seq-len 256`. Do NOT use the default
(16384). At short sequence lengths, system overhead dominates — that's what
you're optimizing.

## What This Task Is NOT

- You are **NOT** writing CUDA kernels (.cu, .h, Triton kernel code)
- You are **NOT** modifying the HSTU attention implementation
- You are **NOT** replacing existing kernels with hand-written alternatives
- You **ARE** writing Python-level system optimizations in
  `model_runner/optimizations/`

## Running Commands

### Correctness test
```bash
python model_runner/test_correctness.py --max-seq-len 256 --mode <your_mode>
```

### Benchmark
```bash
python model_runner/bench.py --max-seq-len 256 --mode <your_mode> \
  --compare-baseline model_runner/baselines/ref.json
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
  python -u model_runner/bench.py --max-seq-len 256 --mode <your_mode> \
    --profile --bench-iters 3
```

## MCP Tools

| MCP | Tool | When to use |
|-----|------|-------------|
| ncompass | `check_auth` | Preflight only |
| ncompass | `analyze_nsys_sqlite` | Raw kernel timing, launch overhead, sync analysis |
| ncompass | `analyze_nsys_perfetto` | Timeline view, GPU/CPU overlap, concurrency |
| ncompass | `analyze_nsys_patterns` | Iteration structure, timing variability |
| ncompass | `analyze_nsys_diff` | Before/after trace comparison |
| ncompass | `consult` | Ambiguous or multi-domain questions |
| KB | `search_kb` | Specific lookups: API names, function signatures |
| KB | `search_kb_deep` | Research: optimization techniques, strategy discovery |
| KB | `read_kb_file` | Read full content of a file from search results |
| KB | `list_kb_adjacent` | Explore related files in the same directory |

### KB workflow
1. Start with `search_kb_deep` for technique research (not `search_kb`)
2. Read the top results in full via `read_kb_file`
3. Explore related files via `list_kb_adjacent`
4. Refine your query if initial results are too generic

## Optimization Module Contract

Each optimization is a Python module in `model_runner/optimizations/`:

```python
def apply(model, batch, hstu_config, **kwargs) -> callable:
    """Return a callable(uih_features, candidates_features) -> model_output."""
```

The callable must return the same output format as the baseline. The `batch`
argument is for setup only — the callable will be invoked with different
batches during benchmarking.

## Note-Taking Schema

### `state.json`
```json
{
  "session_id": "<from .session_id>",
  "iteration": 0,
  "baseline_median_ms": 12.450,
  "best_median_ms": 12.450,
  "best_mode": "baseline",
  "best_commit": "<sha>",
  "current_bottleneck": "<free text>",
  "mcps_available": [],
  "working_optimizations": [],
  "start_time": "<ISO>",
  "status": "running|complete",
  "stop_reason": null
}
```

### `iterations.jsonl` (append-only)
```json
{"iter": 1, "hypothesis": "...", "mode": "...", "correctness": "PASS", "median_ms": 10.23, "baseline_ms": 12.45, "speedup": "1.22x", "verdict": "KEEP|REVERT", "commit": "<sha>", "root_cause": "...(if reverted)", "kb_recommendation": "...(if sys-kb-advisor was consulted)", "profiled_after": true, "nsys_trace": "model_runner/nsys_traces/<name>.nsys-rep"}
```

Read `last_bench.json` and `last_correctness.json` for numbers — do not
transcribe from terminal output.

## Failure Gates

### On ANY failure (correctness, performance regression, capture error)

**BEFORE** reverting or pivoting, spawn the `sys-kb-advisor` subagent:

```
"My [technique] optimization failed with: [exact error or symptom].
 The optimization module is model_runner/optimizations/[mode].py.
 What does the KB suggest?"
```

Read the advisor's response. Log it in iterations.jsonl (`kb_recommendation`
field). Only then decide: attempt the KB-suggested fix, or revert.

### Correctness regression
If outputs diverge beyond tolerance (atol=1e-3, rtol=1e-3), consult
sys-kb-advisor first — if the KB has no applicable fix, **revert
immediately**. Do not stack optimizations on a broken base.

### torch.compile graph breaks
Read the log (`TORCH_LOGS="graph_breaks"`). Spawn sys-kb-advisor with the
graph break details. After 2 failed attempts on the same graph break, pivot
to a different technique.

### CUDA graph capture failure
Isolate the offending operation. Spawn sys-kb-advisor with the capture error.
The existing `CUDAGraphDlrmHSTU` in `run_model.py` shows the pattern.

## Version Control

```bash
git add model_runner/optimizations/ .agent/notes/
git commit -m "description

Correctness: PASS/FAIL
Median latency: X.XXX ms (baseline: Y.YYY ms, speedup: Z.ZZx)"
```

## Constraints

- Always run `test_correctness.py` before benchmarking
- Always profile before making performance-driven edits
- Do not modify files outside `model_runner/optimizations/`
- Single-GPU only
- The custom HSTU kernel is pre-built (`--kernel triton`)
