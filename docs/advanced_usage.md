# Advanced Usage

Advanced features and best practices for using the nCompass SDK.

## Iterative Profiling Workflow

The nCompass SDK supports an iterative profiling workflow that helps you progressively narrow down performance bottlenecks.

### Workflow Overview

1. **Full Trace Capture**: Start with minimal instrumentation to capture the complete trace
2. **AI Analysis**: Get AI-powered insights into performance bottlenecks
3. **Targeted Profiling**: Submit feedback to focus on specific areas
4. **Iterative Refinement**: Repeat the cycle to drill down into issues

### Example Workflow

```python
from ncompass.trace import ProfilingSession

session = ProfilingSession(
    trace_output_dir="./traces",
    cache_dir="./cache",
    session_name="my_model_optimization"
)

# Step 1: Initial profiling with minimal markers
def run_inference():
    model.forward(inputs)

trace_path = session.run_profile(
    run_inference,
    trace_name_suffix="initial"
)

# Step 2: Get AI-powered analysis
summary = session.get_trace_summary(trace_path)
print(summary['markdown'])

# Step 3: Submit feedback for targeted profiling
session.submit_feedback(
    feedback_text="Why does the attention layer take 80% of inference time?",
    target_module="vllm.attention.backends.flash_attn",
    start_line=150,
    end_line=300
)

# Step 4: Apply targeted markers and re-run
session.apply_targeted_markers()
trace_path = session.run_profile(
    run_inference,
    trace_name_suffix="targeted_attention"
)

# Step 5: Analyze targeted results
summary = session.get_trace_summary(
    trace_path,
    feedback_context={
        'feedback_text': 'attention performance',
        'target_module': 'vllm.attention.backends.flash_attn'
    }
)
```

## Trace Filtering

Filter traces to focus only on user-annotated regions:

```python
# Filter to only include user annotations
filtered_trace = session.filter_trace(
    trace_path=trace_path,
    include_cuda_kernels=True,
    include_direct_children=False,
    min_duration_us=100.0
)

# Get marker summary before filtering
marker_summary = session.get_marker_summary(trace_path)
print(f"Found {marker_summary['total_annotations']} markers")
```

## Configuration Management

### Saving and Loading Configurations

```python
# Save configuration after feedback
session.save_config(name="attention_focused")

# Load previously saved configuration
session.load_config(name="attention_focused")
session.apply_targeted_markers()
```

### Manual Configuration

```python
from ncompass.trace import enable_rewrites, RewriteConfig

# Define custom profiling targets
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

# Enable with custom config
enable_rewrites(config=config)
```

## Custom Profiling Contexts

### Using PyTorch Profiler

```python
from ncompass.trace.profile.torch import TorchProfilerContext, TorchRecordContext

# TorchProfilerContext: Start a full profiling session
with TorchProfilerContext(name="my_operation", output_dir="./traces"):
    my_expensive_operation()

# TorchRecordContext: Add markers within an existing profiling session
with TorchRecordContext(name="attention_step"):
    attention_forward()
```

### Using NVTX for Nsight Systems

```python
from ncompass.trace.profile.nvtx import NvtxContext

# Add NVTX ranges for Nsight Systems profiling
with NvtxContext(name="model_forward"):
    model.forward(inputs)
```

## Performance Best Practices

### 1. Start Broad, Then Narrow

Begin with full trace capture, then use AI feedback to focus:

```python
# Initial: Capture everything
trace_1 = session.run_profile(inference_fn)

# After analysis: Focus on bottleneck
session.submit_feedback("Why is layer X slow?", "model.layer_x", 100, 200)
session.apply_targeted_markers()
trace_2 = session.run_profile(inference_fn)
```

### 2. Use Trace Filtering

Reduce trace file size by filtering:

```python
# Keep only user annotations and CUDA kernels
filtered = session.filter_trace(
    include_cuda_kernels=True,
    min_duration_us=50.0  # Filter out operations < 50μs
)
```

### 3. Cache AI Analysis Results

Reuse analysis results to avoid recomputation:

```python
# Analysis results are automatically cached
summary_1 = session.get_trace_summary(trace_path)

# Load cached summary (much faster)
summary_2 = session.load_trace_summary(trace_name_filter="my_trace")
```

### 4. Session Management

Reset session state when starting new experiments:

```python
# Reset to clean state
session.reset()

# Or create a new session for each experiment
session_exp1 = ProfilingSession(
    trace_output_dir="./traces",
    session_name="experiment_1"
)
```

## Integration with Existing Profilers

### vLLM Integration

```python
# For vLLM models, use external profiler mode
session = ProfilingSession(trace_output_dir="./traces")

def run_vllm_inference():
    # vLLM manages its own profiling
    outputs = llm.generate(prompts)

# Session captures vLLM-generated traces
trace_path = session.run_profile(run_vllm_inference)
```

### Custom Integration

```python
# For custom profilers, manage profiling lifecycle manually
session = ProfilingSession(trace_output_dir="./traces")

def run_with_custom_profiler():
    profiler.start()
    my_inference()
    profiler.stop()
    profiler.save()  # Save to trace_output_dir

trace_path = session.run_profile(run_with_custom_profiler)
```

## Troubleshooting

### Import Errors After Enabling Rewrites

If you encounter import errors:

```python
from ncompass.trace import disable_rewrites

# Disable rewrites to restore normal imports
disable_rewrites()

# Re-enable with corrected configuration
enable_rewrites(corrected_config)
```

### Trace File Not Found

Ensure trace files are saved to the correct directory:

```python
session = ProfilingSession(
    trace_output_dir="/absolute/path/to/traces"  # Use absolute paths
)
```

### Analysis Service Connection

If the analysis service is unavailable:

```python
# Specify custom service URL
session = ProfilingSession(
    trace_output_dir="./traces",
    base_url="http://your-analysis-service:8000"
)
```

## See Also

- [API Reference](api_reference.md)
- [Getting Started Guide](getting_started.md)
- [Examples](../examples/)

