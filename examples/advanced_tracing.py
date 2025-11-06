"""
Advanced Tracing Example

This example demonstrates advanced nCompass features:
1. Iterative profiling workflow
2. Feedback-driven targeted profiling
3. Configuration iteration
4. Trace filtering and analysis

Prerequisites:
    pip install ncompass
    nCompass analysis service running (see docs)
"""

from ncompass.trace import ProfilingSession, enable_rewrites, RewriteConfig
import time


class MockModel:
    """Mock model for demonstration purposes."""
    
    def __init__(self):
        self.layers = 4
    
    def attention_layer(self, x):
        """Simulated attention computation."""
        time.sleep(0.05)  # Simulate compute
        return x
    
    def ffn_layer(self, x):
        """Simulated feed-forward network."""
        time.sleep(0.02)  # Simulate compute
        return x
    
    def forward(self, inputs):
        """Full forward pass."""
        x = inputs
        for i in range(self.layers):
            print(f"    → Layer {i+1}: Attention")
            x = self.attention_layer(x)
            print(f"    → Layer {i+1}: FFN")
            x = self.ffn_layer(x)
        return x


def run_inference(model):
    """Run model inference."""
    print("  Running model inference...")
    inputs = {"data": "example"}
    outputs = model.forward(inputs)
    return outputs


def main():
    """Advanced tracing workflow demonstration."""
    
    print("=" * 70)
    print("nCompass SDK - Advanced Tracing Example")
    print("=" * 70)
    
    # Initialize model
    model = MockModel()
    
    # =========================================================================
    # ITERATION 1: Initial Full Trace
    # =========================================================================
    print("\n" + "=" * 70)
    print("ITERATION 1: Capture Full Trace")
    print("=" * 70)
    
    session = ProfilingSession(
        trace_output_dir="./traces",
        cache_dir="./cache",
        session_name="advanced_example"
    )
    
    print("\n[1.1] Running initial profiling with minimal instrumentation...")
    trace_1 = session.run_profile(
        user_code=lambda: run_inference(model),
        trace_name_suffix="iter1_full"
    )
    print(f"✓ Trace saved: {trace_1}")
    
    print("\n[1.2] AI Analysis (placeholder - requires service)...")
    # summary_1 = session.get_trace_summary(trace_1)
    # print(summary_1['markdown'])
    
    print("\nAI would identify: 'Attention layers are taking 70% of time'")
    
    # =========================================================================
    # ITERATION 2: Targeted Profiling on Attention
    # =========================================================================
    print("\n" + "=" * 70)
    print("ITERATION 2: Focus on Attention Layers")
    print("=" * 70)
    
    print("\n[2.1] Submitting feedback for targeted profiling...")
    
    # In real usage, submit_feedback would:
    # - Analyze the target code region
    # - Generate profiling markers automatically
    # - Update configuration
    
    # Example (uncomment with real service):
    # config_2 = session.submit_feedback(
    #     feedback_text="Why are attention layers slow? Break down the computation.",
    #     target_module="__main__.MockModel",
    #     start_line=15,  # attention_layer method
    #     end_line=18
    # )
    
    print("✓ Feedback submitted (simulated)")
    print("  → AI would add detailed markers to attention_layer")
    
    print("\n[2.2] Applying targeted markers...")
    # session.apply_targeted_markers()
    print("✓ Markers applied (simulated)")
    
    print("\n[2.3] Re-running with targeted instrumentation...")
    trace_2 = session.run_profile(
        user_code=lambda: run_inference(model),
        trace_name_suffix="iter2_attention"
    )
    print(f"✓ Trace saved: {trace_2}")
    
    print("\n[2.4] Targeted analysis...")
    # summary_2 = session.get_trace_summary(
    #     trace_2,
    #     feedback_context={
    #         'feedback_text': 'attention performance breakdown',
    #         'target_module': '__main__.MockModel'
    #     }
    # )
    
    print("AI would show: 'QKV projection: 40%, Softmax: 30%, Output: 30%'")
    
    # =========================================================================
    # ITERATION 3: Further Refinement
    # =========================================================================
    print("\n" + "=" * 70)
    print("ITERATION 3: Drill Down into QKV Projection")
    print("=" * 70)
    
    print("\n[3.1] Submitting refined feedback...")
    # config_3 = session.submit_feedback(
    #     feedback_text="Why is QKV projection slow? Show matrix operations.",
    #     target_module="__main__.MockModel",
    #     start_line=16,
    #     end_line=17
    # )
    print("✓ Refined feedback submitted (simulated)")
    
    print("\n[3.2] Applying refined markers...")
    # session.apply_targeted_markers()
    print("✓ Refined markers applied (simulated)")
    
    print("\n[3.3] Final profiling run...")
    trace_3 = session.run_profile(
        user_code=lambda: run_inference(model),
        trace_name_suffix="iter3_qkv"
    )
    print(f"✓ Trace saved: {trace_3}")
    
    # =========================================================================
    # Trace Filtering
    # =========================================================================
    print("\n" + "=" * 70)
    print("TRACE FILTERING")
    print("=" * 70)
    
    print("\n[4.1] Filtering trace to only user annotations...")
    # filtered = session.filter_trace(
    #     trace_path=trace_3,
    #     include_cuda_kernels=True,
    #     include_direct_children=False,
    #     min_duration_us=100.0
    # )
    # print(f"✓ Filtered trace: {filtered}")
    print("✓ Filtering example (uncomment with real traces)")
    
    # =========================================================================
    # Configuration Management
    # =========================================================================
    print("\n" + "=" * 70)
    print("CONFIGURATION MANAGEMENT")
    print("=" * 70)
    
    print("\n[5.1] Saving final configuration...")
    session.save_config(name="attention_optimized")
    print("✓ Configuration saved as 'attention_optimized'")
    
    print("\n[5.2] Configuration can be reused later:")
    print("  session.load_config(name='attention_optimized')")
    print("  session.apply_targeted_markers()")
    
    config_stats = session.get_config_stats()
    print(f"\n[5.3] Configuration stats:")
    print(f"  - Iterations: {config_stats.get('iteration', 0)}")
    print(f"  - Total targets: {config_stats.get('total_targets', 0)}")
    
    # =========================================================================
    # Manual Configuration (Advanced)
    # =========================================================================
    print("\n" + "=" * 70)
    print("MANUAL CONFIGURATION (Advanced)")
    print("=" * 70)
    
    print("\nYou can also manually configure profiling markers:")
    print("""
    from ncompass.trace import enable_rewrites, RewriteConfig
    
    config = RewriteConfig(
        targets={
            "my_module.attention": {
                "class_func_context_wrappings": {
                    "Attention": {
                        "forward": {
                            "context_class": "ncompass.trace.profile.torch.TorchRecordContext",
                            "context_values": ["attention.forward"]
                        }
                    }
                }
            }
        }
    )
    
    enable_rewrites(config=config)
    """)
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("ADVANCED TRACING WORKFLOW COMPLETE")
    print("=" * 70)
    
    print("\nIterative profiling workflow:")
    print("  1. ✓ Initial full trace capture")
    print("  2. ✓ AI identifies bottlenecks (attention layers)")
    print("  3. ✓ Targeted profiling on attention")
    print("  4. ✓ Further refinement on specific operations")
    print("  5. ✓ Trace filtering for focused analysis")
    print("  6. ✓ Configuration save/load for reuse")
    
    print("\nKey benefits:")
    print("  • Progressive narrowing of performance issues")
    print("  • AI-guided marker placement")
    print("  • Minimal manual instrumentation")
    print("  • Reusable configurations")
    
    print("\nNext steps:")
    print("  1. Replace MockModel with your actual model")
    print("  2. Set up nCompass analysis service")
    print("  3. Run iterative profiling on real workload")
    print("  4. Use insights to optimize your model")
    
    print("\nDocumentation: https://docs.ncompass.tech")


if __name__ == "__main__":
    main()

