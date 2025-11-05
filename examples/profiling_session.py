"""
ProfilingSession Workflow Example

This example demonstrates a complete profiling workflow:
1. Initial profiling with minimal instrumentation
2. AI-powered trace analysis
3. Configuration management
4. Trace filtering

Prerequisites:
    pip install ncompass
"""

from ncompass.trace import ProfilingSession
import time


def simulate_model_inference():
    """
    Simulated model inference with multiple operations.
    
    Replace with your actual model:
        model = load_model()
        outputs = model(inputs)
    """
    print("  → Loading model...")
    time.sleep(0.05)
    
    print("  → Running forward pass...")
    time.sleep(0.1)
    
    print("  → Post-processing...")
    time.sleep(0.02)
    
    return {"status": "success"}


def main():
    """Complete profiling session workflow."""
    
    print("=" * 70)
    print("nCompass SDK - ProfilingSession Workflow Example")
    print("=" * 70)
    
    # =========================================================================
    # Step 1: Initialize Session
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 1: Initialize ProfilingSession")
    print("=" * 70)
    
    session = ProfilingSession(
        trace_output_dir="./traces",
        cache_dir="./cache",
        session_name="workflow_example"
    )
    print("✓ Session initialized with name: workflow_example")
    
    # =========================================================================
    # Step 2: Initial Profiling Run
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 2: Run Initial Profiling")
    print("=" * 70)
    
    print("\nProfiling inference code...")
    trace_path_1 = session.run_profile(
        user_code=simulate_model_inference,
        trace_name_suffix="initial_run"
    )
    print(f"✓ Initial trace saved: {trace_path_1}")
    
    # =========================================================================
    # Step 3: Get Configuration Statistics
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 3: Configuration Management")
    print("=" * 70)
    
    config_stats = session.get_config_stats()
    print(f"\nConfiguration Statistics:")
    print(f"  - Iteration: {config_stats.get('iteration', 0)}")
    print(f"  - Total targets: {config_stats.get('total_targets', 0)}")
    
    current_config = session.get_current_config()
    print(f"\nCurrent config has {len(current_config.get('targets', {}))} targets")
    
    # Save configuration for later
    session.save_config(name="initial_config")
    print("✓ Configuration saved")
    
    # =========================================================================
    # Step 4: Trace Analysis (Requires Analysis Service)
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 4: AI-Powered Trace Analysis")
    print("=" * 70)
    
    print("\nNOTE: Trace analysis requires the nCompass analysis service")
    print("      to be running. See: https://docs.ncompass.tech/setup")
    
    # Uncomment when analysis service is available:
    # print("\nAnalyzing trace...")
    # summary = session.get_trace_summary(trace_path_1)
    # print("\n" + "-" * 70)
    # print(summary['markdown'])
    # print("-" * 70)
    
    # =========================================================================
    # Step 5: Trace Filtering
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 5: Trace Filtering")
    print("=" * 70)
    
    print("\nFiltering trace to show only user annotations...")
    # Uncomment to filter (requires valid trace format):
    # filtered_trace = session.filter_trace(
    #     trace_path=trace_path_1,
    #     include_cuda_kernels=True,
    #     min_duration_us=50.0
    # )
    # print(f"✓ Filtered trace: {filtered_trace}")
    
    print("NOTE: Filtering shown as example - uncomment to use with real traces")
    
    # =========================================================================
    # Step 6: Iterative Profiling (Advanced)
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 6: Iterative Profiling")
    print("=" * 70)
    
    print("\nFor iterative profiling workflow:")
    print("  1. Use submit_feedback() to add targeted markers")
    print("  2. Call apply_targeted_markers() to enable them")
    print("  3. Run profiling again with new instrumentation")
    print("\nSee advanced_tracing.py for a complete example")
    
    # =========================================================================
    # Step 7: Session Reset
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 7: Session Management")
    print("=" * 70)
    
    print("\nSession can be reset to clean state:")
    # session.reset()  # Uncomment to reset
    print("  - session.reset() clears configuration and state")
    print("  - Trace files are preserved in trace_output_dir")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("WORKFLOW COMPLETE")
    print("=" * 70)
    
    print("\nKey ProfilingSession features demonstrated:")
    print("  ✓ Session initialization and configuration")
    print("  ✓ Profiling execution")
    print("  ✓ Configuration management (save/load)")
    print("  ✓ Trace analysis integration points")
    print("  ✓ Trace filtering capabilities")
    
    print("\nNext steps:")
    print("  1. Replace simulate_model_inference() with your model")
    print("  2. Set up the nCompass analysis service")
    print("  3. Try advanced_tracing.py for iterative profiling")
    print("\nDocumentation: https://docs.ncompass.tech")


if __name__ == "__main__":
    main()

