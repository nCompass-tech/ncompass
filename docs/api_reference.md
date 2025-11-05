# API Reference

Complete API reference for the nCompass Python SDK.

## Core Modules

### `ncompass.trace`

Main module for profiling and tracing functionality.

#### ProfilingSession

```python
class ProfilingSession:
    """Session for iterative profiling workflow."""
    
    def __init__(
        self,
        trace_output_dir: str,
        cache_dir: Optional[str] = None,
        session_name: Optional[str] = None,
        base_url: Optional[str] = None
    )
```

**Parameters:**
- `trace_output_dir` (str): Directory where trace files are saved
- `cache_dir` (Optional[str]): Cache directory for AI analysis
- `session_name` (Optional[str]): Name for this profiling session
- `base_url` (Optional[str]): Base URL for analysis service

**Methods:**

##### `run_profile()`

```python
def run_profile(
    self,
    user_code: Callable,
    trace_name_suffix: Optional[str] = None,
    filter_trace: Optional[bool] = False,
    filter_trace_args: Optional[Dict[str, Any]] = None
) -> str
```

Profile user code and generate trace file.

**Returns:** Path to generated trace file

##### `get_trace_summary()`

```python
def get_trace_summary(
    self,
    trace_path: Optional[str] = None,
    feedback_context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]
```

Get AI-powered summary of trace.

**Returns:** Dictionary with 'markdown' and 'structured' summaries

##### `submit_feedback()`

```python
def submit_feedback(
    self,
    feedback_text: str,
    target_module: str,
    start_line: int,
    end_line: int,
    trace_path: Optional[str] = None
) -> Dict[str, Any]
```

Submit feedback to generate targeted profiling markers.

**Returns:** Updated configuration dictionary

##### `apply_targeted_markers()`

```python
def apply_targeted_markers(self) -> Dict[str, Any]
```

Apply profiling markers based on accumulated feedback.

**Returns:** Current configuration

##### `filter_trace()`

```python
def filter_trace(
    self,
    trace_path: Optional[str] = None,
    output_path: Optional[str] = None,
    include_cuda_kernels: bool = True,
    include_direct_children: bool = False,
    min_duration_us: float = 0.0
) -> str
```

Filter trace to only include user annotations.

**Returns:** Path to filtered trace file

#### Functions

##### `enable_rewrites()`

```python
def enable_rewrites(config: Optional[RewriteConfig] = None) -> None
```

Enable AST rewrites for profiling instrumentation.

**Parameters:**
- `config` (Optional[RewriteConfig]): Configuration for AST rewrites

##### `enable_full_trace_mode()`

```python
def enable_full_trace_mode() -> None
```

Enable minimal profiling for full trace capture.

##### `disable_rewrites()`

```python
def disable_rewrites() -> None
```

Disable AST rewrites.

### `ncompass.trace.core`

Core functionality for AST rewriting and profiling.

#### RewriteConfig

```python
class RewriteConfig:
    """Configuration for AST rewrites."""
    
    targets: Dict[str, Any]
    ai_analysis_targets: List[str]
    full_trace_mode: bool
```

Configuration object for specifying profiling targets and behavior.

## Examples

### Basic Profiling

```python
from ncompass.trace import ProfilingSession

session = ProfilingSession(trace_output_dir="./traces")

def my_inference():
    # Your code here
    pass

trace_path = session.run_profile(my_inference)
summary = session.get_trace_summary(trace_path)
```

### Iterative Profiling

```python
# First run: capture full trace
trace_path = session.run_profile(my_inference)

# Get AI analysis
summary = session.get_trace_summary(trace_path)

# Submit feedback for targeted profiling
session.submit_feedback(
    feedback_text="Why is attention slow?",
    target_module="model.attention",
    start_line=100,
    end_line=150
)

# Apply targeted markers and re-run
session.apply_targeted_markers()
trace_path = session.run_profile(my_inference)
```

## Data Structures

### Trace Summary Format

```python
{
    "markdown": str,  # Human-readable markdown report
    "structured": {   # Structured performance data
        "operations": List[Dict],
        "bottlenecks": List[Dict],
        "recommendations": List[str]
    }
}
```

### Configuration Format

```python
{
    "targets": {
        "module.name": {
            "class_func_context_wrappings": {...},
            "func_line_range_wrappings": {...}
        }
    },
    "ai_analysis_targets": List[str],
    "full_trace_mode": bool
}
```

## See Also

- [Getting Started Guide](getting_started.md)
- [Advanced Usage](advanced_usage.md)
- [Examples](../examples/)

