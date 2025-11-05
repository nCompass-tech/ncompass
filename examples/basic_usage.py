"""
Basic Usage Example for nCompass SDK

This example demonstrates the simplest way to profile your code using nCompass:
1. Initialize a ProfilingSession
2. Run profiling on your inference code
3. Get AI-powered trace analysis

Prerequisites:
    pip install ncompass
"""

from ncompass.trace import ProfilingSession


def example_inference():
    """
    Placeholder for your actual inference code.
    
    Replace this with your model's inference logic:
    - model.forward(inputs)
    - llm.generate(prompts)
    - pipeline.run(data)
    etc.
    """
    # Example: Simulate some work
    import time
    print("Running inference...")
    time.sleep(0.1)  # Simulate computation
    print("Inference complete!")


def main():
    """Main execution flow for basic profiling."""
    
    print("=" * 60)
    print("nCompass SDK - Basic Usage Example")
    print("=" * 60)
    
    # Step 1: Initialize profiling session
    print("\n[1] Initializing profiling session...")
    session = ProfilingSession(
        trace_output_dir="./traces",      # Where trace files will be saved
        cache_dir="./cache",               # Where analysis cache is stored
        session_name="basic_example"       # Name for this profiling session
    )
    print("✓ Session initialized")
    
    # Step 2: Run profiling
    print("\n[2] Running profiling...")
    print("NOTE: Replace 'example_inference' with your actual inference code")
    
    trace_path = session.run_profile(
        user_code=example_inference,
        trace_name_suffix="initial"
    )
    
    print(f"✓ Profiling complete! Trace saved to: {trace_path}")
    
    # Step 3: Get AI-powered trace analysis
    print("\n[3] Analyzing trace with AI...")
    print("NOTE: This requires the nCompass analysis service to be running")
    print("      See https://docs.ncompass.tech for setup instructions")
    
    # Uncomment when analysis service is available:
    # summary = session.get_trace_summary(trace_path)
    # 
    # print("\n" + "=" * 60)
    # print("TRACE ANALYSIS RESULTS")
    # print("=" * 60)
    # print(summary['markdown'])
    
    print("\n[4] Example complete!")
    print("\nNext steps:")
    print("  - Replace example_inference() with your actual model code")
    print("  - Set up the nCompass analysis service")
    print("  - Explore iterative profiling with profiling_session.py")
    print("  - Read the docs at https://docs.ncompass.tech")


if __name__ == "__main__":
    main()

