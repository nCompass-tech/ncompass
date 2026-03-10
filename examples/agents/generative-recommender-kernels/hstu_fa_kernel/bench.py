#!/usr/bin/env python3
"""
Benchmark PyTorch reference (pytorch_hstu_mha) vs from-scratch kernel
(torch.ops.hstu_ai_optimized) for HSTU attention.

Usage:
    python hstu_fa_kernel/bench.py
    python hstu_fa_kernel/bench.py --save-baseline hstu_fa_kernel/baselines/reference.json
    python hstu_fa_kernel/bench.py --compare-baseline hstu_fa_kernel/baselines/reference.json
    python hstu_fa_kernel/bench.py --batch-size 256 --max-seq-len 512
"""

import argparse
import json
import os
import sys

# Add hstu_fa_kernel/ to sys.path so the local reference package can be imported.
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import torch
try:
    import fbgemm_gpu  # noqa: F401 — registers torch.ops.fbgemm ops needed by pytorch_hstu_mha
except ImportError:
    pass


def load_kernels():
    """Load reference (PyTorch) and scratch kernel, returning availability flags."""
    ref_available = False
    scratch_available = False

    try:
        from reference.pt_hstu_attention import pytorch_hstu_mha  # noqa: F401
        ref_available = True
    except ImportError:
        pass

    try:
        import hstu_ai_optimized._C  # noqa: F401
        scratch_available = True
    except ImportError:
        pass

    return ref_available, scratch_available


def create_jagged_inputs(batch_size, max_seq_len, num_heads, head_dim, device, dtype,
                         distribution="uniform"):
    """Create jagged (variable-length) inputs with seq_offsets."""
    if distribution == "uniform":
        lengths = torch.randint(
            max_seq_len // 2, max_seq_len + 1, (batch_size,), device="cpu"
        )
    elif distribution == "zipf":
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


def run_ref_kernel(q, k, v, seq_offsets, max_seq_len, alpha, causal, num_softmax_heads):
    """Run the PyTorch reference kernel."""
    from reference.pt_hstu_attention import pytorch_hstu_mha
    return pytorch_hstu_mha(
        max_seq_len=max_seq_len,
        alpha=alpha,
        q=q, k=k, v=v,
        seq_offsets=seq_offsets,
        causal=causal,
        training=False,
    )


def run_scratch_kernel(q, k, v, seq_offsets, max_seq_len, alpha, causal,
                       num_softmax_heads):
    """Run the from-scratch kernel via torch.ops.hstu_ai_optimized."""
    return torch.ops.hstu_ai_optimized.hstu_mha_fwd(
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


def benchmark_kernel(run_fn, q, k, v, seq_offsets, max_seq_len, alpha, causal,
                     num_softmax_heads, warmup=10, iterations=100):
    """Benchmark a kernel, return list of elapsed times in ms."""
    for _ in range(warmup):
        run_fn(q, k, v, seq_offsets, max_seq_len, alpha, causal, num_softmax_heads)
    torch.cuda.synchronize()

    timings = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        run_fn(q, k, v, seq_offsets, max_seq_len, alpha, causal, num_softmax_heads)
        end.record()
        torch.cuda.synchronize()
        timings.append(start.elapsed_time(end))

    return timings


def compute_tflops(batch_size, avg_seq_len, num_heads, head_dim, elapsed_ms, causal):
    """Estimate TFLOPS for attention forward pass."""
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


def parse_args():
    """Parse command-line arguments for the benchmark."""
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
    return parser.parse_args()


def print_config(args, num_softmax_heads):
    """Print benchmark configuration."""
    print(f"Benchmark config:")
    print(f"  Batch size:    {args.batch_size}")
    print(f"  Max seq len:   {args.max_seq_len}")
    print(f"  Num heads:     {args.num_heads}")
    print(f"  Head dim:      {args.head_dim}")
    print(f"  Causal:        {args.causal}")
    print(f"  Softmax heads: {num_softmax_heads}")
    print(f"  Distribution:  {args.distribution}")
    print(f"  Warmup/Iters:  {args.warmup}/{args.iterations}")


def print_kernel_availability(ref_available, scratch_available):
    """Print kernel availability and exit if none are available."""
    print(f"\n  Reference (pytorch_hstu_mha):     {'available' if ref_available else 'NOT available'}")
    print(f"  Scratch kernel (hstu_ai_optimized): {'available' if scratch_available else 'NOT available'}")

    if not ref_available and not scratch_available:
        print("\nError: No kernels available. Build at least one:")
        print("  Ensure hstu_fa_kernel/reference/pt_hstu_attention.py exists (for PyTorch reference)")
        print("  python hstu_fa_kernel/build.py           (for scratch kernel)")
        sys.exit(1)


def run_benchmarks(args, q, k, v, seq_offsets, avg_seq_len, num_softmax_heads,
                   ref_available, scratch_available):
    """Run benchmarks for available kernels and return results dict."""
    results = {}

    if ref_available:
        timings = benchmark_kernel(
            run_ref_kernel,
            q, k, v, seq_offsets, args.max_seq_len, args.alpha, args.causal,
            num_softmax_heads, args.warmup, args.iterations,
        )
        results["reference"] = print_results(
            "Reference (pytorch_hstu_mha)", timings, args.batch_size, avg_seq_len,
            args.num_heads, args.head_dim, args.causal,
        )

    if scratch_available:
        timings = benchmark_kernel(
            run_scratch_kernel,
            q, k, v, seq_offsets, args.max_seq_len, args.alpha, args.causal,
            num_softmax_heads, args.warmup, args.iterations,
        )
        results["scratch"] = print_results(
            "Scratch (hstu_ai_optimized)", timings, args.batch_size, avg_seq_len,
            args.num_heads, args.head_dim, args.causal,
        )

    if ref_available and scratch_available:
        ref_median = results["reference"]["median_ms"]
        scratch_median = results["scratch"]["median_ms"]
        speedup = ref_median / scratch_median
        print(f"\n  Speedup: {speedup:.3f}x ({'faster' if speedup > 1 else 'slower'})")

    return results


def save_baseline(path, args, total_tokens, avg_seq_len, results):
    """Save benchmark results to a JSON baseline file."""
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
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Baseline saved to: {path}")


def compare_baseline(path, results):
    """Compare current results against a saved baseline JSON file."""
    if not os.path.exists(path):
        return
    with open(path) as f:
        baseline = json.load(f)
    print(f"\n  Comparing against baseline: {path}")
    for kernel_name in ["reference", "scratch"]:
        if kernel_name in results and kernel_name in baseline.get("results", {}):
            cur = results[kernel_name]["median_ms"]
            base = baseline["results"][kernel_name]["median_ms"]
            change = (cur - base) / base * 100
            print(f"    {kernel_name}: {base:.3f} -> {cur:.3f} ms ({change:+.1f}%)")


def main():
    args = parse_args()

    device = "cuda"
    dtype = torch.bfloat16
    num_softmax_heads = args.num_heads if args.softmax else 0

    print_config(args, num_softmax_heads)

    ref_available, scratch_available = load_kernels()
    print_kernel_availability(ref_available, scratch_available)

    q, k, v, seq_offsets, total_tokens = create_jagged_inputs(
        args.batch_size, args.max_seq_len, args.num_heads, args.head_dim,
        device, dtype, args.distribution,
    )
    avg_seq_len = total_tokens / args.batch_size
    print(f"\n  Total tokens:  {total_tokens}")
    print(f"  Avg seq len:   {avg_seq_len:.1f}")

    results = run_benchmarks(args, q, k, v, seq_offsets, avg_seq_len,
                             num_softmax_heads, ref_available, scratch_available)

    if args.save_baseline:
        save_baseline(args.save_baseline, args, total_tokens, avg_seq_len, results)

    if args.compare_baseline:
        compare_baseline(args.compare_baseline, results)


if __name__ == "__main__":
    main()
