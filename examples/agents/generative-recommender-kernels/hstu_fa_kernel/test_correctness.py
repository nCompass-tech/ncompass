#!/usr/bin/env python3
"""
Compare from-scratch kernel output (torch.ops.hstu_ai_optimized) against
PyTorch reference (pytorch_hstu_mha) for correctness.

Usage:
    python hstu_fa_kernel/test_correctness.py
    python hstu_fa_kernel/test_correctness.py --strict
    python hstu_fa_kernel/test_correctness.py --batch-size 64 --max-seq-len 128
"""

import argparse
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
    """Load both kernels, fail if either is missing."""
    try:
        from reference.pt_hstu_attention import pytorch_hstu_mha  # noqa: F401
    except ImportError:
        print("Error: PyTorch reference not available.")
        print("  Ensure hstu_fa_kernel/reference/pt_hstu_attention.py exists")
        sys.exit(1)

    try:
        import hstu_ai_optimized._C  # noqa: F401
    except ImportError:
        print("Error: Scratch kernel not available. Build it first:")
        print("  python hstu_fa_kernel/build.py")
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


def run_ref_kernel(q, k, v, seq_offsets, max_seq_len, causal, num_softmax_heads):
    """Run PyTorch reference and return output tensor."""
    from reference.pt_hstu_attention import pytorch_hstu_mha
    return pytorch_hstu_mha(
        max_seq_len=max_seq_len,
        alpha=1.0,
        q=q, k=k, v=v,
        seq_offsets=seq_offsets,
        causal=causal,
        training=False,
    )


def run_scratch_kernel(q, k, v, seq_offsets, max_seq_len, causal, num_softmax_heads):
    """Run scratch kernel and return (output, softmax_lse)."""
    return torch.ops.hstu_ai_optimized.hstu_mha_fwd(
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
                atol=1e-1, rtol=5e-2):
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

    # Run PyTorch reference — returns single tensor (total_tokens, H, D)
    ref_out = run_ref_kernel(
        q.clone(), k.clone(), v.clone(), seq_offsets, max_seq_len, causal,
        num_softmax_heads,
    )

    # Run scratch kernel — returns (output, softmax_lse)
    scratch_result = run_scratch_kernel(
        q.clone(), k.clone(), v.clone(), seq_offsets, max_seq_len, causal,
        num_softmax_heads,
    )
    scratch_out = scratch_result[0]

    # Compare outputs
    max_abs_diff = (ref_out - scratch_out).abs().max().item()
    mean_abs_diff = (ref_out - scratch_out).abs().mean().item()
    passed = torch.allclose(ref_out, scratch_out, atol=atol, rtol=rtol)

    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {config_name}")
    print(f"         max_abs_diff={max_abs_diff:.6f}, mean_abs_diff={mean_abs_diff:.6f}")

    return passed


def main():
    parser = argparse.ArgumentParser(description="Test scratch kernel correctness")
    parser.add_argument("--batch-size", "-b", type=int, default=64)
    parser.add_argument("--max-seq-len", "-s", type=int, default=256)
    parser.add_argument("--num-heads", "-n", type=int, default=4)
    parser.add_argument("--head-dim", "-d", type=int, default=128)
    parser.add_argument("--atol", type=float, default=1e-1)
    parser.add_argument("--rtol", type=float, default=5e-2)
    parser.add_argument("--strict", action="store_true", default=False,
                        help="Use stricter tolerances (atol=5e-2)")
    args = parser.parse_args()

    if args.strict:
        args.atol = 5e-2

    load_kernels()

    print("Correctness tests (PyTorch reference vs scratch kernel):")
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
