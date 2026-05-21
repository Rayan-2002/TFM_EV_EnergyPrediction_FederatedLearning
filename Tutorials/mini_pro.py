import torch

import numpy as np
# numpy contains a lot of functions for working with arrays, including mathematical operations, random number generation, and more.

# Creating Synthetic Data : Generating y = 3x + 2 with some noise
# hint : use torch.randn() to add some noise to the data

x = torch.randn(100, 1) * 10 # # 100 random points roughly centered around 0

y = 3 * x + 2 + torch.randn(100, 1) # Adding some noise to the data


# Create a linear regression model using PyTorch's nn.Module
# hint : use nn.Linear() to create a linear layer

class LinearRegressionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1) # 1 input feature, 1 output feature

    def forward(self, x):
        return self.linear(x)
    
# Instantiate the model
model = LinearRegressionModel()

print(model)

# Train the model for several epochs using Forward pass, loss calculation, 
# zeroing gradients, back propagation and optimizer step
# hint : use loss.backward() to perform backpropagation 
# and optimizer.step() to update the model parameters and 
# optimizer.zero_grad() to zero the gradients before the next iteration

# Define loss function and optimizer
loss_function = torch.nn.MSELoss() # Mean Squared Error Loss
optimizer = torch.optim.SGD(model.parameters(), lr=0.01) # Stochastic Gradient

# Training loop
for epoch in range(100):

    model.train()

    # Forward pass
    predictions = model(x)

    # Compute loss
    loss = loss_function(predictions, y)

    # Reset gradients
    optimizer.zero_grad()

    # Backpropagation
    loss.backward()

    # Update weights
    optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")