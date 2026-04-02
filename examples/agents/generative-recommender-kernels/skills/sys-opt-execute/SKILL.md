---
name: sys-opt-execute
description: >
  Execute an agreed optimization plan iteration by iteration. Pauses for
  developer input after each iteration and on any failure.
---

# System Optimization — Execute

You have a plan in `.agent/notes/plan.md`. Execute it iteration by iteration.
The developer stays in control of strategic decisions — you handle the grunt
work.

## Rules (ABSOLUTE — survive context compaction)

1. Read `plan.md` before every iteration — it is the source of truth.
2. NEVER skip profiling after an optimization.
3. NEVER silently revert — present failures to the developer and wait.
4. NEVER change strategy without developer approval.
5. One optimization per iteration, measured independently.
6. After context compaction: read `state.json`, `iterations.jsonl`, `plan.md`.

## State

On first run, initialize `.agent/notes/state.json`:
```json
{
  "session_id": "<from .session_id>",
  "iteration": 0,
  "baseline_median_ms": 0.0,
  "best_median_ms": 0.0,
  "best_mode": "baseline",
  "best_commit": null,
  "working_optimizations": [],
  "start_time": "<ISO>",
  "plan_iteration": 1,
  "status": "running"
}
```

On resume: read `state.json`, last 5 lines of `iterations.jsonl`, `plan.md`.

## Iteration Loop

For each iteration in the plan's action sequence:

1. **Announce** — Tell the developer what you're doing and why (from plan).
2. **Check hints** — Read `.agent/notes/hints.md` if it exists.
3. **Implement** — Write `model_runner/optimizations/<mode>.py` per the plan.
4. **Correctness** — `python model_runner/test_correctness.py --max-seq-len 256 --mode <mode>`
5. **Benchmark** — `python model_runner/bench.py --max-seq-len 256 --mode <mode> --compare-baseline model_runner/baselines/ref.json`
6. **Profile** — Collect nsys trace. If ncompass available, use `analyze_nsys_diff` against baseline.
7. **Record** — Append to `iterations.jsonl`, update `state.json`.
8. **Report** — Present results to the developer (see below). **Wait for response.**
9. **Commit** — `git add model_runner/ .agent/notes/ && git commit`

### Iteration Report Format

```
## Iteration N: [name]

Result: X.XXx (Y.YY ms, baseline Z.ZZ ms)
Verdict: KEPT / REVERTED
Cumulative: X.XXx over baseline

What happened: [2-3 sentences]
Profile delta: [key changes from trace diff]

Next: Iteration N+1: [name from plan]
On track for target: [yes/no]

Proceed?
```

## On Failure

If correctness fails or the technique doesn't work:

1. Isolate the exact error (file, line, root cause — not vague descriptions)
2. Check `plan.md` risk section and `recon/barriers.md`
3. If KB available, search for the failure pattern
4. **Present to developer with options and wait** — do not proceed alone

See `/system-optimizer-reference` for the full failure protocol template
and plan deviation handling.

## On Completion

When all plan iterations are done or the developer decides to stop:
run a final 30-iteration benchmark, write `.agent/notes/summary.md`,
update `state.json` with `"status": "complete"`, and commit.

See `/system-optimizer-reference` for the summary schema.
