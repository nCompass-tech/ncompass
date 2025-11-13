"""
ProfilingSession Iterative Workflow Example

This example demonstrates a complete iterative profiling workflow:
1. Initial profiling with minimal instrumentation
2. AI-powered trace analysis
3. Submit feedback for targeted profiling
4. Applying targeted markers
5. Re-running profiling with detailed markers
6. Feedback-driven trace analysis

Prerequisites:
    pip install ncompass torch
"""

from dotenv import load_dotenv
load_dotenv()

from ncompass.trace import ProfilingSession
from ncompass.trace.infra.utils import logger
import logging
import os
from config import config
from model import run_model_inference

logger.setLevel(logging.DEBUG)

def main():
    """Main iterative profiling workflow."""
    # Initialize profiling session with a descriptive name
    # This will be used to name traces, summaries, and configs
    session = ProfilingSession(
        trace_output_dir=config.torch_logs_dir,
        cache_dir=config.profiling_session_dir,
        session_name="pytorch_model"  # Base name for all generated files
    )
    
    # Step 1: Initial profile with minimal markers
    logger.info("="*80)
    logger.info("STEP 1: Initial full trace capture")
    logger.info("="*80)
    
    # Trace will be named: pytorch_model_initial_YYYY_MM_DD_HH_MM_SS.pt.trace.json
    # Using PyTorch's built-in profiler (no injection needed)
    trace_file = session.run_profile(
        user_code=run_model_inference,
        user_code_kwargs={'enable_profiler': True},
        trace_name_suffix="initial",  # Optional: adds descriptive suffix
    )
    
    logger.info(f"Initial trace saved to: {trace_file}")
    logger.info(f"Trace name: {session.latest_trace_name}")
    
    # Step 2: Get AI-powered trace summary
    logger.info("="*80)
    logger.info("STEP 2: Generate trace summary")
    logger.info("="*80)
    
    # Summary will be saved as: summary_pytorch_model_initial_YYYY_MM_DD_HH_MM_SS.json/.md
    summary = session.get_trace_summary(
        trace_path=trace_file
    )
    print(f"Initial summary generated (length = {len(summary['markdown'])})")
    try:
        summary_path = f"{config.profiling_session_dir}/summary_pytorch_model_initial.md"
        with open(summary_path, "w") as f:
            f.write(summary['markdown'])
        logger.info(f"Summary saved to: {summary_path}")
    except Exception as e:
        logger.error(f"Error saving summary: {e}")
    
    logger.info(f"Summary automatically saved to {config.profiling_session_dir} directory")
    
    # Step 3: Submit feedback for targeted profiling
    logger.info("="*80)
    logger.info("STEP 3: Submit feedback for targeted profiling")
    logger.info("="*80)
    
    # Example: You want to understand why matrix multiplication is slow
    # We'll target the matrix_multiply function
    # Note: In practice, use the actual module path (e.g., 'myproject.models.neural_network')
    # For this example, we use __name__ which will be '__main__' when run directly
    feedback_config = session.submit_feedback(
        feedback_text="Why does matrix_multiply take so long? I want to see detailed timing for each matmul operation.",
        target_module="model",  # This module (use actual module path in production)
        start_line=52,  # Line for matrix_multiply function
        end_line=56,   # End of matrix_multiply function
        trace_path=trace_file
    )
    
    logger.info(f"Feedback processed. Config stats: {session.get_config_stats()}")
    
    # Step 4: Apply targeted markers
    logger.info("="*80)
    logger.info("STEP 4: Apply targeted markers")
    logger.info("="*80)
    
    session.apply_targeted_markers()
    
    # Step 5: Re-run profiling with targeted markers
    logger.info("="*80)
    logger.info("STEP 5: Re-run with targeted markers")
    logger.info("="*80)
    
    # Trace will be named: pytorch_model_detailed_YYYY_MM_DD_HH_MM_SS.pt.trace.json
    # Now with AI-generated TorchRecordContext markers injected
    filtered_trace_file = session.run_profile(
        user_code=run_model_inference,
        user_code_kwargs={'enable_profiler': True},
        trace_name_suffix="detailed",  # Optional: adds descriptive suffix
        filter_trace=True  # Optional: filter the trace to only show user annotations
    )
    
    # Get new summary with feedback context
    # This will generate a summary that directly addresses your question
    # Summary will be saved as: summary_pytorch_model_detailed_YYYY_MM_DD_HH_MM_SS.json/.md
    new_summary = session.get_trace_summary(
        trace_path=filtered_trace_file,  # Use filtered trace instead of raw trace
        feedback_context=session.latest_feedback_context  # Use stored feedback context
    )
    print(f"Feedback-driven summary generated (length = {len(new_summary['markdown'])})")
    try:
        summary_path = f"{config.profiling_session_dir}/summary_pytorch_model_detailed.md"
        with open(summary_path, "w") as f:
            f.write(new_summary['markdown'])
        logger.info(f"Summary saved to: {summary_path}")
    except Exception as e:
        logger.error(f"Error saving summary: {e}")
    
    logger.info("Note: This summary directly addresses your question with trace data")
    
    # Save configuration for reproducibility
    # Config will be saved as: profile_config_pytorch_model_detailed_YYYY_MM_DD_HH_MM_SS.json
    # This associates the config with the trace it was used for
    session.save_config()  # Uses automatic naming based on latest trace
    logger.info(f"Configuration automatically saved with associated naming")
    
    # Note: All files (trace, summary, config) share the same base name for easy association
    
    # Optional: Load previous summaries with filtering
    logger.info("="*80)
    logger.info("BONUS: Loading summaries with filtering")
    logger.info("="*80)
    
    # Load the most recent summary for this specific session
    # This prevents mixing up summaries from different profiling sessions
    previous_summary = session.load_trace_summary(
        trace_name_filter="pytorch_model"  # Only load summaries for this session
    )
    
    if previous_summary:
        trace_name = previous_summary.get('trace_name', 'unknown')
        logger.info(f"Successfully loaded previous summary: {trace_name}")
    
    # Step 6 (optional): Submit more feedback and iterate
    logger.info("="*80)
    logger.info("Workflow complete! You can now:")
    logger.info("- Submit more feedback with session.submit_feedback()")
    logger.info("- Apply markers with session.apply_targeted_markers()")
    logger.info("- Re-run with session.run_profile()")
    logger.info("\nNaming features:")
    logger.info("- All traces, summaries, and configs are automatically named")
    logger.info("- Files are associated by trace name for easy organization")
    logger.info("- Use trace_name_filter when loading summaries to avoid mixing sessions")
    logger.info("="*80)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", action="store_true", help="Clean up all traces and summaries")
    args = parser.parse_args()
    
    if args.clean:
        import shutil
        if os.path.exists(config.torch_logs_dir):
            shutil.rmtree(config.torch_logs_dir)
        if os.path.exists(config.profiling_session_dir):
            shutil.rmtree(config.profiling_session_dir)
        logger.info("Cleaned up all traces and summaries")
    main()
