"""
Simple neural network model and training function.

This file contains the model architecture and training logic separate from the 
profiling/instrumentation code to avoid conflicts with nCompass rewriting.

The nCompass rewriter in modal_replica.py will instrument the functions in this file.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from ncompass.trace.infra.utils import logger


class SimpleNet(nn.Module):
    """Simple feedforward neural network for profiling demonstration."""
    
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

def train_simple_network(epochs=10, hidden_size=512, X=None, y=None, batch_size=128):
    """
    Train a simple feedforward neural network on dummy data.
    
    This function demonstrates typical PyTorch training patterns that
    can be profiled to identify performance bottlenecks.
    
    Args:
        epochs: Number of training epochs
        hidden_size: Hidden layer size
        X: Optional pre-created input tensor. If None, creates dummy data.
        y: Optional pre-created target tensor. If None, creates dummy targets.
        batch_size: Batch size (only used if X and y are None)
    
    Returns:
        Dictionary with final loss and epoch count
    """
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Generate dummy training data if not provided
    if X is None:
        X = torch.randn(batch_size, 784, device=device)
    if y is None:
        # Match batch size to X if X was provided
        actual_batch_size = X.shape[0] if X is not None else batch_size
        y = torch.randint(0, 10, (actual_batch_size,), device=device)
    
    # Ensure data is on the correct device
    X = X.to(device)
    y = y.to(device)
    
    # Create model and move to device
    model = SimpleNet(hidden_size=hidden_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
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
    if device == "cuda":
        torch.cuda.synchronize()
    
    return {"final_loss": loss.item(), "epochs": epochs, "batch_size": X.shape[0]}

