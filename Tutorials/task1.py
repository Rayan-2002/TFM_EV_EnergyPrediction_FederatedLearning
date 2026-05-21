import torch

import numpy as np


data = [[1,2],[3,4]]

x_data = torch.tensor(data) 
# This will create a tensor of shape 
#(2, 2) that contains the values from the list of lists `data` 
# which are [[1, 2], [3, 4]]. 
# The resulting tensor will have the same values as the original list 
# of lists but in a format that can be used for computations in PyTorch.



print(x_data)


x_ones = torch.ones_like(x_data) # retains the properties of x_data
print(f"Ones Tensor: \n {x_ones} \n")


x_zeros = torch.zeros_like(x_data) # retains the properties of x_data
print(f"Zeros Tensor: \n {x_zeros} \n")

x_rand = torch.randint_like(x_data, low=0, high=10) # retains the properties of x_data
print(f"Random Tensor: \n {x_rand} \n")

x_rand_float = torch.rand_like(x_data, dtype= torch.float)
print(f"Random Float Tensor: \n {x_rand_float} \n")


shape = (3,4) # shape takes only a tuple of integers since it represents the dimensions of the tensor.
rand_tensor = torch.rand(shape)
print(f"Random Tensor with shape {shape}: \n {rand_tensor} \n")

x_rand_int = torch.randint(low=5, high = 18, size = shape)
print(f"Random Integer Tensor with shape {shape}: \n {x_rand_int} \n")



tensor = torch.ones(4,4)
print(f"Original tensor: \n {tensor} \n")
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")
tensor[3, :] = 0
print(tensor)

