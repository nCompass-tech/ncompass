"""
Description: Model for testing.
"""


class Model:
    """Model for testing."""
    
    def __init__(self):
        self.x = 0
    
    def forward(self):
        print("Forward pass")
        self.x += 1
        return self.x