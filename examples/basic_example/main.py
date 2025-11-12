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
from config import config
from model import run_model_inference

logger.setLevel(logging.DEBUG)

# PROFILING_TARGETS defines which functions should be instrumented with trace markers.
# This configuration tells ncompass to automatically wrap specific code regions with
# profiling contexts that will appear in PyTorch profiler traces.
#
# Structure:
#   - Top-level key ("model") is a target name/namespace
#   - "func_line_range_wrappings" specifies which functions and line ranges to wrap
#
# Each wrapping entry:
#   - "function": Name of the function to instrument (must match the function name in model.py)
#   - "start_line"/"end_line": Line range within the function to wrap with profiling context
#   - "context_class": The profiling context class to use (TorchRecordContext for PyTorch profiling)
#   - "context_values": Metadata to attach to the trace marker (e.g., custom name for identification)
#
# In this example, we're wrapping lines 52-54 of the matrix_multiply function in model.py
# with a custom marker named "my-custom-marker-name". This will create a visible region
# in the PyTorch profiler trace, making it easier to identify and analyze this specific
# computation in the profiling output.
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
    # Build the rewrite configuration from PROFILING_TARGETS
    # This tells ncompass which code to instrument before execution
    config = {"targets": PROFILING_TARGETS}
    
    # Enable code rewriting/instrumentation based on the configuration
    # This will modify the code at runtime to add trace markers
    enable_rewrites(config=RewriteConfig.from_dict(config))
    
    # Run the model inference with PyTorch profiler enabled
    # The profiler will capture execution traces including our custom markers
    run_model_inference(enable_profiler=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", action="store_true", help="Clean up all traces and summaries")
    args = parser.parse_args()
    
    # Optional cleanup: remove previous profiling traces and session data
    if args.clean:
        import shutil
        if os.path.exists(config.torch_logs_dir):
            shutil.rmtree(config.torch_logs_dir)
        if os.path.exists(config.profiling_session_dir):
            shutil.rmtree(config.profiling_session_dir)
        logger.info("Cleaned up all traces and summaries")
    
    # Run the main profiling workflow
    main()
