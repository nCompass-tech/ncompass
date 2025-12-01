"""
Simple neural network model and training function.

This file contains the model architecture and training logic separate from the 
profiling/instrumentation code to avoid conflicts with nCompass rewriting.

The nCompass rewriter in main.py will instrument the functions in this file.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from ncompass.trace.infra.utils import logger

import logging
logger.setLevel(logging.DEBUG)


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

def train_simple_network(epochs=10, hidden_size=512):
    """
    Train a simple feedforward neural network on dummy data.
    
    This function demonstrates typical PyTorch training patterns that
    can be profiled to identify performance bottlenecks.
    
    Args:
        epochs: Number of training epochs
        hidden_size: Hidden layer size
    
    Returns:
        Dictionary with final loss and epoch count
    """
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Create model and move to device
    model = SimpleNet(hidden_size=hidden_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Generate dummy training data
    batch_size = 128
    X = torch.randn(batch_size, 784, device=device)
    y = torch.randint(0, 10, (batch_size,), device=device)
    
    logger.info(f"Training for {epochs} epochs with hidden_size={hidden_size}")
    
    # Training loop
    for epoch in range(epochs):
        # Forward pass
        outputs = model(X)
        loss = criterion(outputs, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % max(1, epochs // 10) == 0 or epoch == 0:
            logger.info(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
    
    # Force synchronization for accurate timing
    if device == "cuda":
        torch.cuda.synchronize()
    
    logger.info(f"Training complete. Final loss: {loss.item():.4f}")
    
    return {"final_loss": loss.item(), "epochs": epochs}

