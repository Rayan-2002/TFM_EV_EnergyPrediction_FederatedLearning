import torch
import numpy as np


a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

# Multiplication of tensors

product_tensor = a * b
print(f"Product of tensors: {product_tensor} \n")

# Dot product of tensors
dot_product = torch.dot(a, b)
print(f"Dot product of tensors: {dot_product} \n")


# Generate random input values for x for Mini Linear Regression

x = torch.rand(100, 1) * 10

# Generate noisy targets y 

y = 2 * x + 1 + torch.randn(100, 1) * 2

# print shape of x and y
print(f"Shape of x: {x.shape} \n")
print(f"Shape of y: {y.shape} \n")

print(x[:5])
print(y[:5])

