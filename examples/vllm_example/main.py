#!/usr/bin/env python3
"""
vLLM profiling example with NCU and Nsys support.

This script runs vLLM inference with optional NVTX markers or Torch profiling.
To profile, wrap the command with ncu or nsys externally.

Prerequisites:
    1. Add NVTX markers using the nCompass VSCode extension (optional)
    2. Set environment variables:
       - NCOMPASS_CACHE_DIR=<path to .cache dir> (if using NVTX markers)
       - NCOMPASS_PROFILER_TYPE=NVTX (if using NVTX markers)
       - HF_TOKEN=<your huggingface token> (for gated models)

Usage:
    # Run with NVTX markers (for ncu or nsys profiling)
    ncu <ncu_options> -- python main.py --nvtx --model <model_name>
    ncompass profile -- python main.py --nvtx --model <model_name>

    # Profile with Torch profiler
    VLLM_TORCH_PROFILER_DIR=.torch_traces python main.py --torch --model <model_name>
"""

import argparse
import logging
import os
import sys

import nvtx
import torch
from vllm import LLM, SamplingParams

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def run_vllm_inference(model_name: str, use_nvtx: bool = False) -> None:
    """
    Run vLLM inference with optional NVTX markers.

    Args:
        model_name: HuggingFace model name
        use_nvtx: Whether to wrap inference with NVTX markers
    """
    logger.info(f"Initializing model: {model_name}")

    llm = LLM(
        model=model_name,
        tensor_parallel_size=1,
    )

    test_prompt = (
        "Write a haiku about artificial intelligence. "
        "End with the Haiku, don't say anything else."
    )

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.95,
        max_tokens=50
    )

    outputs = llm.generate([test_prompt], sampling_params)

    try:
        logger.info(f"Test Prompt: {test_prompt}")
        logger.info(f"Response: {outputs[0].outputs[0].text}")
    except Exception as e:
        logger.error(f"Error printing results: {e}")
        print(outputs)


def run_vllm_with_torch_profiler(model_name: str) -> None:
    """
    Run vLLM inference with the built-in Torch profiler.

    Requires VLLM_TORCH_PROFILER_DIR environment variable to be set.

    Args:
        model_name: HuggingFace model name
    """
    profiler_dir = os.environ.get("VLLM_TORCH_PROFILER_DIR")
    if not profiler_dir:
        logger.error("VLLM_TORCH_PROFILER_DIR environment variable not set")
        logger.error("Set it to the directory where traces should be saved")
        sys.exit(1)

    logger.info(f"Initializing model: {model_name}")
    logger.info(f"Torch profiler output: {profiler_dir}")

    llm = LLM(
        model=model_name,
        tensor_parallel_size=1,
    )

    test_prompt = (
        "Write a haiku about artificial intelligence. "
        "End with the Haiku, don't say anything else."
    )

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.95,
        max_tokens=50
    )

    llm.start_profile()
    outputs = llm.generate([test_prompt], sampling_params)
    llm.stop_profile()

    try:
        logger.info(f"Test Prompt: {test_prompt}")
        logger.info(f"Response: {outputs[0].outputs[0].text}")
    except Exception as e:
        logger.error(f"Error printing results: {e}")
        print(outputs)

    logger.info(f"Torch profile saved to: {profiler_dir}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run vLLM inference with profiling support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with NVTX markers for NCU profiling
  ncu --nvtx --nvtx-include "regex:nc_start_capture" -- python main.py --nvtx

  # Run with NVTX markers for Nsys profiling
  ncompass profile -- python main.py --nvtx --model Qwen/Qwen2.5-0.5B

  # Profile with Torch profiler
  VLLM_TORCH_PROFILER_DIR=.torch_traces python main.py --torch --model Qwen/Qwen2.5-0.5B
        """
    )

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        '--nvtx', action='store_true',
        help='Run with NVTX markers (wrap with ncu or nsys for profiling)'
    )
    mode_group.add_argument(
        '--torch', action='store_true',
        help='Profile with vLLM built-in Torch profiler'
    )

    parser.add_argument(
        '--model', type=str, default='Qwen/Qwen2.5-0.5B',
        help='HuggingFace model to use (default: Qwen/Qwen2.5-0.5B)'
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.nvtx:
        run_vllm_inference(model_name=args.model, use_nvtx=True)
        return 0
    elif args.torch:
        run_vllm_with_torch_profiler(model_name=args.model)
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
