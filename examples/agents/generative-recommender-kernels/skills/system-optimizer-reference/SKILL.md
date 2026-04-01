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

## What You CAN Edit

You may edit any file under `model_runner/`, including `run_model.py`.

Each optimization must still have an entry point module in
`model_runner/optimizations/<mode>.py` with the standard `apply()` signature
— this is how `bench.py` and `test_correctness.py` discover and invoke your
optimization.

**Do NOT modify:**
- `bench.py` — benchmark harness
- `test_correctness.py` — correctness harness
- `optimizations/__init__.py` — mode discovery

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
gpu-lock nsys profile \
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

### GPU lock (automatic)
`bench.py` and `test_correctness.py` automatically acquire an exclusive GPU lock
before running GPU-intensive work. No action needed for those commands.

For `nsys profile` or `ncu` commands invoked directly, prefix with `gpu-lock`:
```bash
gpu-lock nsys profile ...
gpu-lock ncu ...
```
This prevents other agents from running GPU work simultaneously, which would
corrupt measurements. If no lock directory is present, the wrapper is a no-op.

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

## Failure Protocol

### Step 1: Isolate the exact failure

Do NOT describe failures vaguely. Fill in this template:

```markdown
**Error**: <exact error message or symptom>
**Location**: <file:line, function name>
**Tensor/operation**: <specific tensor name, expected vs actual shape>
**Root cause**: <why this specific thing failed — trace it back>
```

"The preprocessor has complex size interactions" is NOT acceptable.
"payload_features['viewer_id'] has shape [16] but _pad_tensor tried to
pad it to [4096] because it wasn't excluded from UIH padding" IS acceptable.

### Step 2: Consult KB before reverting

Spawn sys-kb-advisor with the exact failure + root cause:
```
"My [technique] optimization failed with: [exact error].
 Root cause: [from step 1].
 Module: model_runner/optimizations/[mode].py.
 What does the KB suggest?"
```

### Step 3: Decide

- KB has applicable fix → attempt it
- You have a targeted fix for the root cause → attempt it
- Only revert if targeted fix fails AND root cause is architectural

Log in iterations.jsonl: `"kb_recommendation"`, `"root_cause"`, `"decision"`.

### Specific failure types

**Correctness regression** (atol=1e-3, rtol=1e-3): Follow steps 1-3 above.
If KB has no fix AND root cause is architectural → revert. Do not stack
optimizations on a broken base.

**torch.compile graph breaks**: Read log (`TORCH_LOGS="graph_breaks"`).
Include graph break details in sys-kb-advisor prompt. After 2 failed attempts
on the same graph break, pivot.

**CUDA graph capture failure**: Isolate the offending operation (which
function call, which tensor). Include in sys-kb-advisor prompt.

## Version Control

```bash
git add model_runner/ .agent/notes/
git commit -m "description

Correctness: PASS/FAIL
Median latency: X.XXX ms (baseline: Y.YYY ms, speedup: Z.ZZx)"
```

## Summary Schema

When wrapping up a session, write `.agent/notes/summary.md`:

```markdown
# Session Summary
**Session ID:** <id>
**Stop reason:** <which condition triggered>
**Total iterations:** <N>

## Results
| Metric | Value |
|--------|-------|
| Baseline median | X.XX ms |
| Best median | X.XX ms |
| Best mode | <name> |
| Speedup | X.XXx |

## Optimizations kept
- <mode>: <description> (X.XXx)

## Optimizations reverted
- <mode>: <description> — <root_cause>

## Remaining hypotheses
- <from hypotheses.md>
```

## Constraints

- Always run `test_correctness.py` before benchmarking
- Always profile before making performance-driven edits
- Do not modify `bench.py`, `test_correctness.py`, or `optimizations/__init__.py`
- Do not modify files outside `model_runner/`
- Single-GPU only
- The custom HSTU kernel is pre-built (`--kernel triton`)
