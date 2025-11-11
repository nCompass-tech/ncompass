"""
Description: Run script
"""


from model import Model

model = Model()
fwd = model.forward()
print(f"Forward pass: {fwd}")