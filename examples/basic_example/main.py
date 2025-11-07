"""
Basic example illustrating the use of trace markers to profile a model.

Prerequisites:
    pip install ncompass torch
"""

from dotenv import load_dotenv
load_dotenv()

from ncompass.trace.core.rewrite import enable_rewrites
from ncompass.trace.core.pydantic import RewriteConfig
from ncompass.trace.infra.utils import logger
import logging
import os
from config import TORCH_LOGS_DIR, PROFILING_SESSION_DIR
from model import run_model_inference

logger.setLevel(logging.DEBUG)

PROFILING_TARGETS = {
    "model": {
        "func_line_range_wrappings": [
            {
                "function": "matrix_multiply",
                "start_line": 52,
                "end_line": 54,
                "context_class": "ncompass.trace.profile.torch.TorchRecordContext",
                "context_values": [
                    {
                        "name": "name",
                        "value": "my-custom-marker-name",
                        "type": "literal"
                    },
                ],
            }
        ]
    }
}

def main():
    """Main iterative profiling workflow."""
    config = {"targets": PROFILING_TARGETS}
    enable_rewrites(config=RewriteConfig.from_dict(config))
    run_model_inference(enable_profiler=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", action="store_true", help="Clean up all traces and summaries")
    args = parser.parse_args()
    
    if args.clean:
        import shutil
        if os.path.exists(TORCH_LOGS_DIR):
            shutil.rmtree(TORCH_LOGS_DIR)
        if os.path.exists(PROFILING_SESSION_DIR):
            shutil.rmtree(PROFILING_SESSION_DIR)
        logger.info("Cleaned up all traces and summaries")
    main()
