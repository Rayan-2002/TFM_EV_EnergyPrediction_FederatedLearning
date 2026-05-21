import torch 

a = torch.tensor([1,2,3])
b = torch.tensor([4,5,6])

sum_tensor = a + b
print(f"Sum of tensors: {sum_tensor} \n")

product_tensor = a * b
print(f"Product of tensors: {product_tensor} \n ")

# Operations on the elements of tensor a 
sum_a = torch.sum(a)
print(f"Sum of elements in tensor a: {sum_a} \n")

mult_a = torch.prod(a)
print(f"Product of elements in tensor a: {mult_a} \n")

sub_a = torch.sub(a, 1)
print(f"Subtraction of 1 from each element in tensor a: {sub_a} \n")



A = torch.randint(low=0, high=10, size=(2, 3))
B = torch.randint(low=0, high=10, size=(3, 4))

print(f"Matrix A: \n {A} \n")
print(f"Matrix B: \n {B} \n")


# Perform matrix multiplication
matrix_product = torch.matmul(A, B)
print(f"Matrix product of A and B: \n {matrix_product} \n")

# Using the @ operator for matrix multiplication
matrix_product_at = A @ B
print(f"Matrix product of A and B using @ operator: \n {matrix_product_at} \n")


# Indexing and slicing tensors

x = torch.tensor([[10,20,30],
                  [40,50,60]])

# Access the value 20
value_20 = x[0,1]
print(f"Value 20: {value_20} \n")

# Access the second row
second_row = x[1]

# Access the third column
third_column = x[:, 2]
print(f"Third column: {third_column} \n")

# Access the last element
last_element = x[1,2]
print(f"Last element: {last_element} \n")


# Modifying tensor values
x[0,0] = 99
print(f"Modified tensor x: \n {x} \n")

# GPU Check using torch.cuda.is_available() and .to(device)
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available. Using GPU.")
else:    device = torch.device("cpu")
print(f"Using device: {device} \n")


tensor = torch.tensor([1, 2, 3])

# We move our tensor to the current accelerator if available
if torch.accelerator.is_available():
    tensor = tensor.to(torch.accelerator.current_accelerator())
print(f"Tensor on device: {tensor} \n")
