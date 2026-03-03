#!/usr/bin/env python3
"""
Compare optimized kernel output (torch.ops.hstu_opt) against reference
(torch.ops.hstu) for correctness.

Usage:
    python runner/test_correctness.py
    python runner/test_correctness.py --batch-size 64 --max-seq-len 128
"""

import argparse
import sys

import torch


def load_kernels():
    """Load both kernels, fail if either is missing."""
    try:
        import hstu._C  # noqa: F401
    except ImportError:
        print("Error: Reference kernel not available. Build it first:")
        print("  bash runner/build_reference.sh")
        sys.exit(1)

    try:
        import hstu_runner._C  # noqa: F401
    except ImportError:
        print("Error: Optimized kernel not available. Build it first:")
        print("  bash runner/build.sh")
        sys.exit(1)


def create_jagged_inputs(batch_size, max_seq_len, num_heads, head_dim, device, dtype):
    """Create jagged inputs with uniform random lengths."""
    lengths = torch.randint(
        max(1, max_seq_len // 2), max_seq_len + 1, (batch_size,), device="cpu"
    )
    seq_offsets = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    seq_offsets[1:] = torch.cumsum(lengths.to(device), dim=0).int()
    total_tokens = seq_offsets[-1].item()

    q = torch.randn(total_tokens, num_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(total_tokens, num_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(total_tokens, num_heads, head_dim, device=device, dtype=dtype)

    return q, k, v, seq_offsets


def run_kernel(op_fn, q, k, v, seq_offsets, max_seq_len, causal, num_softmax_heads):
    """Run a kernel and return (output, softmax_lse)."""
    return op_fn(
        max_seq_len,        # max_seq_len
        1.0,                # alpha
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


def test_config(batch_size, max_seq_len, num_heads, head_dim, causal, softmax,
                atol=5e-2, rtol=5e-2):
    """Test a single configuration. Returns (passed, info_dict)."""
    device = "cuda"
    dtype = torch.bfloat16
    num_softmax_heads = num_heads if softmax else 0

    config_name = (
        f"B={batch_size}, S={max_seq_len}, H={num_heads}, D={head_dim}, "
        f"causal={causal}, softmax={softmax}"
    )

    q, k, v, seq_offsets = create_jagged_inputs(
        batch_size, max_seq_len, num_heads, head_dim, device, dtype
    )

    # Run reference
    ref_out, ref_lse = run_kernel(
        torch.ops.hstu.hstu_mha_fwd,
        q.clone(), k.clone(), v.clone(), seq_offsets, max_seq_len, causal,
        num_softmax_heads,
    )

    # Run optimized
    opt_out, opt_lse = run_kernel(
        torch.ops.hstu_opt.hstu_mha_fwd,
        q.clone(), k.clone(), v.clone(), seq_offsets, max_seq_len, causal,
        num_softmax_heads,
    )

    # Compare outputs
    max_abs_diff = (ref_out - opt_out).abs().max().item()
    mean_abs_diff = (ref_out - opt_out).abs().mean().item()
    passed = torch.allclose(ref_out, opt_out, atol=atol, rtol=rtol)

    # Compare softmax_lse if present
    lse_info = ""
    if ref_lse is not None and opt_lse is not None:
        lse_max_diff = (ref_lse - opt_lse).abs().max().item()
        lse_passed = torch.allclose(ref_lse, opt_lse, atol=atol, rtol=rtol)
        passed = passed and lse_passed
        lse_info = f", lse_max_diff={lse_max_diff:.6f}"

    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {config_name}")
    print(f"         max_abs_diff={max_abs_diff:.6f}, mean_abs_diff={mean_abs_diff:.6f}{lse_info}")

    return passed


def main():
    parser = argparse.ArgumentParser(description="Test optimized kernel correctness")
    parser.add_argument("--batch-size", "-b", type=int, default=64)
    parser.add_argument("--max-seq-len", "-s", type=int, default=256)
    parser.add_argument("--num-heads", "-n", type=int, default=4)
    parser.add_argument("--head-dim", "-d", type=int, default=128)
    parser.add_argument("--atol", type=float, default=5e-2)
    parser.add_argument("--rtol", type=float, default=5e-2)
    args = parser.parse_args()

    load_kernels()

    print("Correctness tests (reference vs optimized):")
    print(f"  Tolerances: atol={args.atol}, rtol={args.rtol}\n")

    # Test configurations: (causal, softmax)
    configs = [
        (True, False),   # causal, no softmax (most common HSTU config)
        (True, True),    # causal, with softmax
        (False, False),  # non-causal, no softmax
    ]

    all_passed = True
    for causal, softmax in configs:
        passed = test_config(
            args.batch_size, args.max_seq_len, args.num_heads, args.head_dim,
            causal, softmax, args.atol, args.rtol,
        )
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("All tests PASSED.")
    else:
        print("Some tests FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
