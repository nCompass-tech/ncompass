import argparse
import json
import os
from pathlib import Path

from ncompass.trace.infra.utils import logger
import logging
logger.setLevel(logging.DEBUG)

import torch
from vllm import LLM, SamplingParams

def parse_args():
    parser = argparse.ArgumentParser(description='Run vLLM with profiling')
    
    # Mutually exclusive profiling mode
    profiling_group = parser.add_mutually_exclusive_group(required=True)
    profiling_group.add_argument(
        '--torch', action='store_true',
        help='Use torch profiler (llm.start_profile/stop_profile)'
    )
    profiling_group.add_argument(
        '--nsys', action='store_true',
        help='Use NVTX markers for nsys profiling'
    )
    
    parser.add_argument(
        '--model', type=str, default='openai/gpt-oss-20b',
        help='HuggingFace model to use (default: openai/gpt-oss-20b)'
    )
    
    return parser.parse_args()

def load_and_test_model(model_name: str, use_torch: bool):
    """Load a model using vLLM and test it with a simple prompt"""
    
    logger.info(f"Initializing model: {model_name}")

    # Initialize vLLM engine
    llm = LLM(
        model=model_name,
        tensor_parallel_size=1,
    )
    
    # Run a test prompt
    logger.info("\nRunning test prompt...")
    test_prompt = (
        "Write a haiku about artificial intelligence. "
        "End with the Haiku, don't say anything else."
    )
    # Configure sampling parameters
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.95,
        max_tokens=10
    )
    
    # Profile based on mode
    if use_torch:
        llm.start_profile()
    else: 
        torch.cuda.nvtx.range_push("nc_start_capture")
    
    outputs = llm.generate([test_prompt], sampling_params)
    
    if use_torch:
        llm.stop_profile()
    else:
        torch.cuda.nvtx.range_pop()
    
    # Print results
    try:
        logger.info(f"\nTest Prompt: {test_prompt}")
        logger.info(f"Response: {outputs[0].outputs[0].text}")
    except Exception as e:
        logger.error(f"Error printing results. Raw outputs:")
        print(outputs)

if __name__ == "__main__":
    args = parse_args()
    load_and_test_model(
        model_name=args.model,
        use_torch=args.torch
    )
