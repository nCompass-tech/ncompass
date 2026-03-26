#!/usr/bin/env python3
"""End-to-end latency benchmark for model_runner optimizations.

Modes:
  baseline   — raw model forward (default)
  <name>     — loads optimizations/<name>.py and calls its apply()

Examples:
  python model_runner/bench.py
  python model_runner/bench.py --mode baseline --save-baseline model_runner/baselines/ref.json
  python model_runner/bench.py --mode my_opt --compare-baseline model_runner/baselines/ref.json
  python model_runner/bench.py --mode my_opt --profile   # for nsys capture
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

# Allow imports from the GR working directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "generative-recommenders"))

from run_model import (
    build_configs,
    create_model,
    generate_batch,
    load_or_cache_weights,
)

from optimizations import discover_modes, load_mode


# ---------------------------------------------------------------------------
# Benchmark core
# ---------------------------------------------------------------------------

def _run_iters(
    forward_fn,
    uih_features,
    candidates_features,
    num_iters: int,
) -> list[float]:
    """Run *num_iters* forward passes, return per-iteration latencies (seconds)."""
    latencies: list[float] = []
    with torch.no_grad():
        for _ in range(num_iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            forward_fn(uih_features, candidates_features)
            torch.cuda.synchronize()
            latencies.append(time.perf_counter() - t0)
    return latencies


def _stats(latencies: list[float]) -> dict:
    """Compute summary statistics from a list of latencies (seconds)."""
    sorted_lat = sorted(latencies)
    n = len(sorted_lat)
    return {
        "num_iters": n,
        "median_ms": round(statistics.median(sorted_lat) * 1000, 3),
        "mean_ms": round(statistics.mean(sorted_lat) * 1000, 3),
        "min_ms": round(sorted_lat[0] * 1000, 3),
        "max_ms": round(sorted_lat[-1] * 1000, 3),
        "p95_ms": round(sorted_lat[int(n * 0.95)] * 1000, 3) if n >= 20 else None,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    available = discover_modes()
    modes_help = f"baseline (default), or one of: {', '.join(available)}" if available else "baseline (default)"

    parser = argparse.ArgumentParser(
        description="Benchmark model_runner forward pass latency.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--mode", type=str, default="baseline", help=modes_help)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-embeddings", type=int, default=100_000)
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--warmup-iters", type=int, default=3)
    parser.add_argument("--bench-iters", type=int, default=10)
    parser.add_argument("--kernel", choices=["pytorch", "triton"], default="triton")
    parser.add_argument("--cache-dir", type=str,
                        default=str(Path(__file__).resolve().parent / ".model_cache"))
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--save-baseline", type=str, default=None,
                        help="Save results to this JSON path")
    parser.add_argument("--compare-baseline", type=str, default=None,
                        help="Compare against a saved baseline JSON")
    parser.add_argument("--profile", action="store_true",
                        help="Enable cudaProfilerApi markers (for nsys capture)")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda:0")

    # --- Model setup (shared with run_model.py) ---
    hstu_config, table_configs = build_configs(args.num_embeddings, args.max_seq_len)
    print(f"Config: batch_size={args.batch_size}, max_seq_len={hstu_config.max_seq_len}, "
          f"mode={args.mode}, kernel={args.kernel}")

    from generative_recommenders.common import HammerKernel

    model = create_model(hstu_config, table_configs, device)
    if not args.no_cache:
        load_or_cache_weights(model, args.num_embeddings, args.cache_dir, device)
    model.eval()
    kernel = HammerKernel.PYTORCH if args.kernel == "pytorch" else HammerKernel.TRITON
    model.set_hammer_kernel(kernel)

    batch = generate_batch(hstu_config, args.batch_size, device)

    # --- Build forward function ---
    if args.mode == "baseline":
        def forward_fn(uih, cand):
            return model(uih, cand)
    else:
        apply_fn = load_mode(args.mode)
        forward_fn = apply_fn(model, batch, hstu_config)

    # --- Warmup ---
    print(f"Warmup ({args.warmup_iters} iters)...")
    _run_iters(forward_fn, batch.uih_features_kjt, batch.candidates_features_kjt,
               args.warmup_iters)

    # --- Benchmark ---
    if args.profile:
        torch.cuda.cudart().cudaProfilerStart()

    print(f"Benchmarking ({args.bench_iters} iters)...")
    latencies = _run_iters(
        forward_fn, batch.uih_features_kjt, batch.candidates_features_kjt,
        args.bench_iters,
    )

    if args.profile:
        torch.cuda.cudart().cudaProfilerStop()

    stats = _stats(latencies)
    stats["mode"] = args.mode
    stats["batch_size"] = args.batch_size
    stats["max_seq_len"] = hstu_config.max_seq_len
    stats["kernel"] = args.kernel

    # --- Report ---
    print(f"\n{'='*60}")
    print(f"Mode: {args.mode}")
    print(f"  median  {stats['median_ms']:>8.3f} ms")
    print(f"  mean    {stats['mean_ms']:>8.3f} ms")
    print(f"  min     {stats['min_ms']:>8.3f} ms")
    print(f"  max     {stats['max_ms']:>8.3f} ms")
    if stats["p95_ms"] is not None:
        print(f"  p95     {stats['p95_ms']:>8.3f} ms")
    print(f"{'='*60}")

    # --- Save ---
    if args.save_baseline:
        out_path = Path(args.save_baseline)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(stats, indent=2) + "\n")
        print(f"Saved baseline to {out_path}")

    # --- Compare ---
    if args.compare_baseline:
        ref_path = Path(args.compare_baseline)
        if not ref_path.exists():
            print(f"Baseline not found: {ref_path}")
            sys.exit(1)
        ref = json.loads(ref_path.read_text())
        ref_med = ref["median_ms"]
        cur_med = stats["median_ms"]
        speedup = ref_med / cur_med if cur_med > 0 else float("inf")
        delta_pct = (1 - cur_med / ref_med) * 100 if ref_med > 0 else 0
        print(f"\nComparison vs {ref_path.name} (mode={ref.get('mode', '?')}):")
        print(f"  baseline median: {ref_med:.3f} ms")
        print(f"  current  median: {cur_med:.3f} ms")
        print(f"  speedup: {speedup:.2f}x ({delta_pct:+.1f}%)")


if __name__ == "__main__":
    main()
