"""
ProfilingSession Iterative Workflow Example

This example demonstrates a complete iterative profiling workflow:
1. Initial profiling with minimal instrumentation
2. AI-powered trace analysis
3. User feedback for targeted profiling
4. Applying targeted markers
5. Re-running profiling with detailed markers
6. Feedback-driven trace analysis

Prerequisites:
    pip install ncompass torch
"""

from dotenv import load_dotenv
load_dotenv()

from ncompass.trace.infra.utils import logger
import logging
import os
import torch
from config import config
logger.setLevel(logging.DEBUG)


class SimpleNeuralNetwork:
    """A simple neural network with multiple layers for profiling demonstration."""
    
    def __init__(self, input_size=512, hidden_size=1024, output_size=256, device=None):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Determine device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        # Initialize weights
        self.weight1 = torch.randn(input_size, hidden_size, device=self.device, requires_grad=False)
        self.weight2 = torch.randn(hidden_size, hidden_size, device=self.device, requires_grad=False)
        self.weight3 = torch.randn(hidden_size, output_size, device=self.device, requires_grad=False)
        self.bias1 = torch.randn(hidden_size, device=self.device, requires_grad=False)
        self.bias2 = torch.randn(hidden_size, device=self.device, requires_grad=False)
        self.bias3 = torch.randn(output_size, device=self.device, requires_grad=False)
    
    def matrix_multiply(self, x, weight, bias):
        """Perform matrix multiplication: x @ weight + bias."""
        result = torch.matmul(x, weight)
        result = result + bias
        return result
    
    def relu_activation(self, x):
        """Apply ReLU activation function."""
        return torch.clamp(x, min=0.0)
    
    def layer_norm(self, x):
        """Apply simple layer normalization."""
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return (x - mean) / (std + 1e-5)
    
    def forward_layer1(self, x):
        """First layer: input -> hidden."""
        x = self.matrix_multiply(x, self.weight1, self.bias1)
        x = self.relu_activation(x)
        return x
    
    def forward_layer2(self, x):
        """Second layer: hidden -> hidden."""
        x = self.matrix_multiply(x, self.weight2, self.bias2)
        x = self.layer_norm(x)
        x = self.relu_activation(x)
        return x
    
    def forward_layer3(self, x):
        """Third layer: hidden -> output."""
        x = self.matrix_multiply(x, self.weight3, self.bias3)
        return x
    
    def forward(self, x):
        """Full forward pass through all layers."""
        logger.info("Starting forward pass...")
        x = self.forward_layer1(x)
        logger.info("Completed layer 1")
        x = self.forward_layer2(x)
        logger.info("Completed layer 2")
        x = self.forward_layer3(x)
        logger.info("Completed layer 3")
        return x


def prepare_input_data(batch_size=32, input_size=512, device=None):
    """Prepare input data for the model."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Device not specified, using default: {device}")
    logger.info(f"Preparing input data: batch_size={batch_size}, input_size={input_size}, device={device}")
    return torch.randn(batch_size, input_size, device=device)


def run_model_inference(enable_profiler: bool = False):
    """Run model inference with optional PyTorch profiler.
    
    Args:
        enable_profiler: If True, enable PyTorch's built-in profiler
    """
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    logger.info("Initializing model...")
    model = SimpleNeuralNetwork(input_size=512, hidden_size=1024, output_size=256, device=device)
    
    logger.info("Preparing input data...")
    inputs = prepare_input_data(batch_size=32, input_size=512, device=device)
    
    # Enable PyTorch profiler if requested
    if enable_profiler:
        logger.info("Enabling PyTorch profiler...")
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            with_stack=True,
        ) as prof:
            outputs = model.forward(inputs)
        
        # Export trace
        trace_path = os.path.join(config.torch_logs_dir, "trace.json")
        prof.export_chrome_trace(trace_path)
        logger.info(f"Trace exported to: {trace_path}")
    else:
        logger.info("Running inference without profiler...")
        outputs = model.forward(inputs)
    
    logger.info(f"Inference complete. Output shape: {outputs.shape}")
    return outputs