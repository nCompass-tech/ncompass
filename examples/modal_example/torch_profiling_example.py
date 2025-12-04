"""
Modal example for profiling PyTorch neural network training on GPU with nCompass SDK integration.

This example demonstrates how to:
1. Set up a Modal App with GPU support
2. Integrate nCompass SDK for tracing and instrumentation
3. Train a simple neural network with instrumentation
4. Profile GPU-accelerated PyTorch code
5. Save profiling traces to a Modal Volume

Based on: https://modal.com/docs/examples/torch_profiling
nCompass SDK: https://ncompass.tech
"""

import os
from pathlib import Path
from typing import Optional

import modal

# Create a Modal Volume to persist profiling traces
traces = modal.Volume.from_name("example-traces", create_if_missing=True)
TRACE_DIR = Path("/traces")

# Define the Modal App
app = modal.App("example-torch-profiling")

# nCompass Specific: Define the container image with PyTorch dependencies and nCompass SDK
# We copy the local config.json which is where the nCompass VSCode extension tracks added
# tracepoints, i.e. regions you want to profile. Copying this across allows the SDK to 
# inject trace markers for profiling 
image = modal.Image.debian_slim(python_version="3.11")\
    .uv_pip_install("torch")\
    .uv_pip_install("ncompass>=0.1.9")\
    .uv_pip_install("pathlib")

ncompass_local_tracepoint_config = \
        Path(f"{os.getcwd()}/.cache/ncompass/profiles/.default/.default/current/config.json")
ncompass_remote_tracepoint_config = "/config/ncompass_config.json"
if ncompass_local_tracepoint_config.exists(): 
    image = image.add_local_file(ncompass_local_tracepoint_config, 
                                 ncompass_remote_tracepoint_config)

ncompass_trace_dir = Path(f"{os.getcwd()}/.traces")
ncompass_trace_dir.mkdir(exist_ok=True)

# Import torch and nCompass SDK within the image context
with image.imports():
    import torch
    import torch.nn as nn
    import torch.optim as optim
    
    # nCompass Specifc: Imports to print logs as well as run linking
    import json
    from pathlib import Path
    from ncompass.trace.core.rewrite import enable_rewrites
    from ncompass.trace.core.pydantic import RewriteConfig
    from ncompass.trace.infra.utils import logger
    from ncompass.trace.converters import link_user_annotation_to_kernels
    from ncompass.trace.converters.utils import write_chrome_trace
    import logging

# Reusable config for Modal functions
config = {"gpu": "a10g", "image": image}

@app.function(**config)
def train_simple_network(epochs=10, hidden_size=512):
    """
    Train a simple feedforward neural network on dummy data.
    
    This function demonstrates typical PyTorch training patterns that
    can be profiled to identify performance bottlenecks.
    """
    # Define a simple neural network
    class SimpleNet(nn.Module):
        def __init__(self, input_size=784, hidden_size=512, output_size=10):
            super(SimpleNet, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.fc3 = nn.Linear(hidden_size, output_size)
        
        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            x = self.relu(x)
            x = self.fc3(x)
            return x
    
    # Create model and move to GPU
    model = SimpleNet(hidden_size=hidden_size).cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Generate dummy training data
    batch_size = 128
    X = torch.randn(batch_size, 784, device="cuda")
    y = torch.randint(0, 10, (batch_size,), device="cuda")
    
    # Training loop
    for epoch in range(epochs):
        # Forward pass
        outputs = model(X)
        loss = criterion(outputs, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Force synchronization for accurate timing
    torch.cuda.synchronize()
    
    return {"final_loss": loss.item(), "epochs": epochs}

@app.function(volumes={TRACE_DIR: traces}, **config)
def profile(
    function,
    label: Optional[str] = None,
    steps: int = 3,
    schedule=None,
    record_shapes: bool = False,
    profile_memory: bool = False,
    with_stack: bool = False,
    print_rows: int = 0,
    link_annotations: bool = True,
    verbose: bool = False,
    **kwargs,
):
    """
    Profile a Modal function using PyTorch's built-in profiler.
    
    Args:
        function: Name or reference to the Modal function to profile
        label: Optional label for the profiling run
        steps: Number of profiling steps (default: 3)
        schedule: Custom profiling schedule (default: wait=1, warmup=1, active=steps-2)
        record_shapes: Record tensor shapes during profiling
        profile_memory: Profile memory usage
        with_stack: Include Python stack traces
        print_rows: Number of rows to print in summary table
        link_annotations: If True, link user_annotation events to kernels (default: True)
        verbose: If True, print detailed statistics when linking annotations
        **kwargs: Arguments to pass to the target function
    """
    from uuid import uuid4
    
    # nCompass Specific: Initialize nCompass SDK
    # Here we load the config.json that's generated by the extension and passed to the SDK to
    # perform the profiler statement injections. It's important that this is placed in a function
    # that is run BEFORE the function(s) you want to profile and where you will place tracepoints
    # via the SDK 
    if Path(ncompass_remote_tracepoint_config).exists():
        logger.setLevel(logging.DEBUG)
        with open(ncompass_remote_tracepoint_config) as f:
            cfg = json.load(f)
            enable_rewrites(config=RewriteConfig.from_dict(cfg))

    # Resolve function name to actual function
    if isinstance(function, str):
        try:
            function = app.registered_functions[function]
        except KeyError:
            raise ValueError(f"Function {function} not found")
    function_name = function.tag

    # Create output directory for this profiling run
    output_dir = (
        TRACE_DIR / (function_name + (f"_{label}" if label else "")) / str(uuid4())
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up profiling schedule
    if schedule is None:
        if steps < 3:
            raise ValueError("Steps must be at least 3 when using default schedule")
        schedule = {"wait": 1, "warmup": 1, "active": steps - 2, "repeat": 0}
    schedule = torch.profiler.schedule(**schedule)

    # Run profiling
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=schedule,
        record_shapes=record_shapes,
        profile_memory=profile_memory,
        with_stack=with_stack,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(output_dir),
    ) as prof:
        for _ in range(steps):
            function.local(**kwargs)  # Execute the target function
            prof.step()

    # Print summary table if requested
    if print_rows:
        print(
            prof.key_averages().table(sort_by="cuda_time_total", row_limit=print_rows)
        )

    # Get the trace file path
    trace_path = sorted(
        output_dir.glob("**/*.pt.trace.json"),
        key=lambda pth: pth.stat().st_mtime,
        reverse=True,
    )[0]

    # Run post processing of trace to ensure annotations coincide with GPU kernels
    write_chrome_trace(trace_path, link_user_annotation_to_kernels(trace_path))
    print(f"Trace saved to {trace_path.relative_to(TRACE_DIR)}")

    return trace_path.read_text(), trace_path.relative_to(TRACE_DIR)

@app.local_entrypoint()
def main(
    function: str = "train_simple_network",
    label: Optional[str] = None,
    steps: int = 3,
    schedule=None,
    record_shapes: bool = False,
    profile_memory: bool = False,
    with_stack: bool = False,
    print_rows: int = 10,
    local_trace_dir: str = ".traces",
    kwargs_json_path: Optional[str] = None,
    no_link: bool = False,
    verbose: bool = False,
):
    """
    Run profiling from the command line.
    
    Example usage:
        modal run torch_profiling_example.py --function train_simple_network --print-rows 10
        modal run torch_profiling_example.py --function train_simple_network --label "baseline" --steps 5
        modal run torch_profiling_example.py --local-trace-dir ./traces
        modal run torch_profiling_example.py --no-link  # Disable linking annotations
        modal run torch_profiling_example.py --verbose  # Show detailed linking statistics
    """
    if kwargs_json_path is not None:
        import json
        kwargs = json.loads(Path(kwargs_json_path).read_text())
    else:
        kwargs = {}

    results, remote_path = profile.remote(
        function,
        label=label,
        steps=steps,
        schedule=schedule,
        record_shapes=record_shapes,
        profile_memory=profile_memory,
        with_stack=with_stack,
        print_rows=print_rows,
        link_annotations=not no_link,
        verbose=verbose,
        **kwargs,
    )

    # Save trace locally
    local_dir = Path(local_trace_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    output_path = local_dir / remote_path.name
    output_path.write_text(results)
    print(f"Trace saved locally at {output_path} - view using the nComapss VSCode extension")
