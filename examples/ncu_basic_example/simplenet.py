"""
Simple neural network model for NCU profiling.

This file contains the model architecture and inference logic.
The nCompass rewriter will instrument the functions in this file
to add NVTX annotations for NVIDIA Nsight Compute profiling.
"""

import argparse
import logging

import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleNet(nn.Module):
    """Simple feedforward neural network for profiling demonstration."""
    
    def __init__(self, input_size=1024, hidden_size=2048, output_size=512):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run SimpleNet for NCU profiling.")
    parser.add_argument("--batch", type=int, default=512, help="Batch size")
    parser.add_argument("--iters", type=int, default=20, help="Profiling iterations")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--input-dim", type=int, default=1024, help="Input dimension")
    parser.add_argument("--hidden-dim", type=int, default=2048, help="Hidden dimension")
    parser.add_argument("--output-dim", type=int, default=512, help="Output dimension")
    parser.add_argument(
        "--precision", 
        type=str, 
        default="fp32", 
        choices=["fp32", "fp16"], 
        help="Precision (fp32 or fp16)"
    )
    return parser.parse_args()


def main() -> None:
    """Run inference."""
    args = parse_args()
    
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available. Install a CUDA build of PyTorch.")
    
    device = torch.device("cuda")
    torch.manual_seed(0)
    
    # Map precision string to torch dtype
    dtype = torch.float32
    if args.precision == "fp16":
        dtype = torch.float16
    
    logger.info(f"Using device: {device}, precision: {args.precision}")
    logger.info(f"Model: input={args.input_dim}, hidden={args.hidden_dim}, output={args.output_dim}")
    logger.info(f"Batch size: {args.batch}, warmup: {args.warmup}, iters: {args.iters}")
    
    # Create model and input with specified precision
    model = SimpleNet(
        input_size=args.input_dim,
        hidden_size=args.hidden_dim,
        output_size=args.output_dim
    ).to(device).to(dtype)
    model.eval()
    
    x = torch.randn(args.batch, args.input_dim, device=device, dtype=dtype)
    
    with torch.no_grad():
        # Warmup iterations
        logger.info("Running warmup iterations...")
        for _ in range(args.warmup):
            _ = model(x)
        torch.cuda.synchronize()
        
        # Profiling iterations
        logger.info("Running profiling iterations...")
        for _ in range(args.iters):
            _ = model(x)
        torch.cuda.synchronize()
    
    logger.info("Profiling complete.")


if __name__ == "__main__":
    main()
