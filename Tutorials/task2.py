import torch

x = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])

print(f"Original tensor: \n {x} \n")

# Print shape of the tensor
print(f"Shape of the tensor: {x.shape} \n")

# Print number of dimensions
print(f"Number of dimensions: {x.ndim} \n")

# Print total number of elements
print(f"Total number of elements is : {x.numel()} \n")


y = torch.arange(12)
print(f"Original tensor: \n {y} \n")

# Reshape the tensor to (3, 4)
y_reshaped = y.reshape(3,4)
print(f"Reshaped tensor (3, 4): \n {y_reshaped} \n")

# Reshape the tensor to (2, 6)
y_reshaped_2 = y.reshape(2, 6)
print(f"Reshaped tensor (2, 6): \n {y_reshaped_2} \n")

# Using the .view() method to reshape the tensor to (4, 3)
y_view = y.view(4, 3)
print(f"Reshaped tensor using .view() (4, 3): \n {y_view} \n")

# The difference between .reshape() and .view() is that .reshape() can return a view or a copy of the original tensor, while .view() always returns a view. 
# If the original tensor is contiguous in memory, both .reshape() and .view() will return a view.