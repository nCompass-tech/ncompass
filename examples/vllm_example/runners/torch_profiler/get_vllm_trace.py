from ncompass.trace.infra.utils import logger
import logging
logger.setLevel(logging.DEBUG)

import os
import json
from pathlib import Path
from vllm import LLM, SamplingParams

from ncompass.trace.core.rewrite import enable_rewrites
from ncompass.trace.core.pydantic import RewriteConfig

cwd = os.getcwd()
os.environ["VLLM_TORCH_PROFILER_DIR"] = f"{cwd}/.traces"

ncompass_cfg = Path(f"{cwd}/.cache/ncompass/profiles/.default/.default/current/config.json")
if ncompass_cfg.exists():
    with ncompass_cfg.open() as f:
        cfg = json.load(f)
        enable_rewrites(config=RewriteConfig.from_dict(cfg))

def load_and_test_model(model_name: str):
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
        max_tokens=1
    )
    llm.start_profile()
    outputs = llm.generate([test_prompt], sampling_params)
    llm.stop_profile()
    
    # Print results
    try:
        logger.info(f"\nTest Prompt: {test_prompt}")
        logger.info(f"Response: {outputs[0].outputs[0].text}")
        # logger.info("Num Tokens:", len(outputs[0].outputs[0].token_ids))
    except Exception as e:
        logger.error(f"Error printing results. Raw outputs:")
        print(outputs)

if __name__ == "__main__":
    load_and_test_model('meta-llama/Llama-3.1-8B-Instruct')
