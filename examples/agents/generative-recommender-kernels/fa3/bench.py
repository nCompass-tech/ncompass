#!/usr/bin/env python3
"""
Benchmark reference (torch.ops.hstu) vs optimized (torch.ops.hstu_opt) kernels.

Usage:
    python runner/bench.py
    python runner/bench.py --save-baseline runner/baselines/reference.json
    python runner/bench.py --compare-baseline runner/baselines/reference.json
    python runner/bench.py --batch-size 256 --max-seq-len 512
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch


def load_kernels():
    """Load reference and optimized kernels, returning availability flags."""
    ref_available = False
    opt_available = False

    try:
        import hstu._C  # noqa: F401
        ref_available = True
    except ImportError:
        pass

    try:
        import hstu_runner._C  # noqa: F401
        opt_available = True
    except ImportError:
        pass

    return ref_available, opt_available


def create_jagged_inputs(batch_size, max_seq_len, num_heads, head_dim, device, dtype,
                         distribution="uniform"):
    """Create jagged (variable-length) inputs with seq_offsets."""
    if distribution == "uniform":
        lengths = torch.randint(
            max_seq_len // 2, max_seq_len + 1, (batch_size,), device="cpu"
        )
    elif distribution == "zipf":
        # Zipf-like: most sequences short, few long
        lengths = torch.clamp(
            (torch.distributions.Pareto(1.0).sample((batch_size,)) * max_seq_len / 4).int(),
            min=1,
            max=max_seq_len,
        )
    else:
        # Fixed length
        lengths = torch.full((batch_size,), max_seq_len, dtype=torch.int32)

    seq_offsets = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    seq_offsets[1:] = torch.cumsum(lengths.to(device), dim=0).int()
    total_tokens = seq_offsets[-1].item()

    q = torch.randn(total_tokens, num_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(total_tokens, num_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(total_tokens, num_heads, head_dim, device=device, dtype=dtype)

    return q, k, v, seq_offsets, total_tokens


def run_kernel(op_fn, q, k, v, seq_offsets, max_seq_len, alpha, causal,
               num_softmax_heads):
    """Run a single kernel call."""
    return op_fn(
        max_seq_len,        # max_seq_len
        alpha,              # alpha
        q, k, v,
        seq_offsets,        # seq_offsets
        causal,             # causal
        None,               # num_targets
        None,               # attn_scale
        0,                  # max_attn_len
        0,                  # min_full_attn_seq_len
        0,                  # contextual_seq_len
        None,               # q_descale
        None,               # k_descale
        None,               # v_descale
        0,                  # sm_margin
        0,                  # max_q_len
        None,               # seq_offsets_q
        num_softmax_heads,  # num_softmax_heads
        False,              # training
    )


def benchmark_kernel(op_fn, q, k, v, seq_offsets, max_seq_len, alpha, causal,
                     num_softmax_heads, warmup=10, iterations=100):
    """Benchmark a kernel, return list of elapsed times in ms."""
    # Warmup
    for _ in range(warmup):
        run_kernel(op_fn, q, k, v, seq_offsets, max_seq_len, alpha, causal,
                   num_softmax_heads)
    torch.cuda.synchronize()

    # Benchmark with CUDA events
    timings = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        run_kernel(op_fn, q, k, v, seq_offsets, max_seq_len, alpha, causal,
                   num_softmax_heads)
        end.record()
        torch.cuda.synchronize()
        timings.append(start.elapsed_time(end))

    return timings


def compute_tflops(batch_size, avg_seq_len, num_heads, head_dim, elapsed_ms, causal):
    """Estimate TFLOPS for attention forward pass."""
    # FLOPs: 2 * N^2 * d * h * b for QK^T + 2 * N^2 * d * h * b for AV
    # With causal mask, roughly half the FLOPs
    flops = 4 * avg_seq_len * avg_seq_len * head_dim * num_heads * batch_size
    if causal:
        flops = flops // 2
    tflops = flops / (elapsed_ms / 1000) / 1e12
    return tflops


def print_results(name, timings, batch_size, avg_seq_len, num_heads, head_dim, causal):
    """Print benchmark results."""
    timings_t = torch.tensor(timings)
    median_ms = timings_t.median().item()
    mean_ms = timings_t.mean().item()
    min_ms = timings_t.min().item()
    std_ms = timings_t.std().item()
    tflops = compute_tflops(batch_size, avg_seq_len, num_heads, head_dim, median_ms, causal)

    print(f"\n  {name}:")
    print(f"    Median: {median_ms:.3f} ms")
    print(f"    Mean:   {mean_ms:.3f} ms (+/- {std_ms:.3f})")
    print(f"    Min:    {min_ms:.3f} ms")
    print(f"    TFLOPS: {tflops:.2f} (estimated)")

    return {
        "median_ms": median_ms,
        "mean_ms": mean_ms,
        "min_ms": min_ms,
        "std_ms": std_ms,
        "tflops": tflops,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark HSTU attention kernels")
    parser.add_argument("--batch-size", "-b", type=int, default=512)
    parser.add_argument("--max-seq-len", "-s", type=int, default=256)
    parser.add_argument("--num-heads", "-n", type=int, default=4)
    parser.add_argument("--head-dim", "-d", type=int, default=128)
    parser.add_argument("--causal", action="store_true", default=True)
    parser.add_argument("--no-causal", dest="causal", action="store_false")
    parser.add_argument("--softmax", action="store_true", default=False,
                        help="Enable softmax heads (num_softmax_heads = num_heads)")
    parser.add_argument("--distribution", choices=["uniform", "zipf", "fixed"],
                        default="uniform")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", "-i", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--save-baseline", type=str, default=None,
                        help="Save results to JSON file")
    parser.add_argument("--compare-baseline", type=str, default=None,
                        help="Compare against saved baseline JSON")
    args = parser.parse_args()

    device = "cuda"
    dtype = torch.bfloat16
    num_softmax_heads = args.num_heads if args.softmax else 0

    print(f"Benchmark config:")
    print(f"  Batch size:    {args.batch_size}")
    print(f"  Max seq len:   {args.max_seq_len}")
    print(f"  Num heads:     {args.num_heads}")
    print(f"  Head dim:      {args.head_dim}")
    print(f"  Causal:        {args.causal}")
    print(f"  Softmax heads: {num_softmax_heads}")
    print(f"  Distribution:  {args.distribution}")
    print(f"  Warmup/Iters:  {args.warmup}/{args.iterations}")

    ref_available, opt_available = load_kernels()
    print(f"\n  Reference kernel (hstu):     {'available' if ref_available else 'NOT available'}")
    print(f"  Optimized kernel (hstu_opt): {'available' if opt_available else 'NOT available'}")

    if not ref_available and not opt_available:
        print("\nError: No kernels available. Build at least one:")
        print("  bash runner/build_reference.sh")
        print("  bash runner/build.sh")
        sys.exit(1)

    # Create inputs
    q, k, v, seq_offsets, total_tokens = create_jagged_inputs(
        args.batch_size, args.max_seq_len, args.num_heads, args.head_dim,
        device, dtype, args.distribution,
    )
    avg_seq_len = total_tokens / args.batch_size
    print(f"\n  Total tokens:  {total_tokens}")
    print(f"  Avg seq len:   {avg_seq_len:.1f}")

    results = {}

    if ref_available:
        timings = benchmark_kernel(
            torch.ops.hstu.hstu_mha_fwd,
            q, k, v, seq_offsets, args.max_seq_len, args.alpha, args.causal,
            num_softmax_heads, args.warmup, args.iterations,
        )
        results["reference"] = print_results(
            "Reference (hstu)", timings, args.batch_size, avg_seq_len,
            args.num_heads, args.head_dim, args.causal,
        )

    if opt_available:
        timings = benchmark_kernel(
            torch.ops.hstu_opt.hstu_mha_fwd,
            q, k, v, seq_offsets, args.max_seq_len, args.alpha, args.causal,
            num_softmax_heads, args.warmup, args.iterations,
        )
        results["optimized"] = print_results(
            "Optimized (hstu_opt)", timings, args.batch_size, avg_seq_len,
            args.num_heads, args.head_dim, args.causal,
        )

    # Speedup comparison
    if ref_available and opt_available:
        ref_median = results["reference"]["median_ms"]
        opt_median = results["optimized"]["median_ms"]
        speedup = ref_median / opt_median
        print(f"\n  Speedup: {speedup:.3f}x ({'faster' if speedup > 1 else 'slower'})")

    # Save baseline
    if args.save_baseline:
        save_data = {
            "config": {
                "batch_size": args.batch_size,
                "max_seq_len": args.max_seq_len,
                "num_heads": args.num_heads,
                "head_dim": args.head_dim,
                "causal": args.causal,
                "softmax": args.softmax,
                "distribution": args.distribution,
                "total_tokens": total_tokens,
                "avg_seq_len": avg_seq_len,
            },
            "results": results,
        }
        os.makedirs(os.path.dirname(args.save_baseline) or ".", exist_ok=True)
        with open(args.save_baseline, "w") as f:
            json.dump(save_data, f, indent=2)
        print(f"\n  Baseline saved to: {args.save_baseline}")

    # Compare against baseline
    if args.compare_baseline and os.path.exists(args.compare_baseline):
        with open(args.compare_baseline) as f:
            baseline = json.load(f)
        print(f"\n  Comparing against baseline: {args.compare_baseline}")
        for kernel_name in ["reference", "optimized"]:
            if kernel_name in results and kernel_name in baseline.get("results", {}):
                cur = results[kernel_name]["median_ms"]
                base = baseline["results"][kernel_name]["median_ms"]
                change = (cur - base) / base * 100
                print(f"    {kernel_name}: {base:.3f} -> {cur:.3f} ms ({change:+.1f}%)")


if __name__ == "__main__":
    main()
