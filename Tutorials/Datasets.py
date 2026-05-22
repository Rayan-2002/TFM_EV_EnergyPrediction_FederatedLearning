import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


# Requirements
# Create:
# training dataset
# test dataset
# Print:
# number of samples in training set
# number of samples in test set
# Access: training_data[0]
# Print: features and labels of training_data[0] and image shape

training_data = datasets.FashionMNIST(
    root = "data",
    train = True, # or False for test data
    download = True, # download data from the internet if not available
    transform = ToTensor() # convert data to torch tensors
)

test_data = datasets.FashionMNIST(
    root = "data",
    train = False,
    download = True,
    transform = ToTensor()
)

print(f"Number of samples in training set: {len(training_data)}")
print(f"Number of samples in test set: {len(test_data)}")

training_data_sample = training_data[0]
features, label = training_data_sample
print(f"Features shape: {features.shape}")
print(f"Label: {label}")

# Print tensor properties
print(f"Tensor shape: {features.shape}")
print(f"Tensor datatype: {features.dtype}")
print(f"Tensor device: {features.device}")

print(f"min pixel value: {features.min()}")
print(f"max pixel value: {features.max()}")
print(f"mean pixel value: {features.mean()}")

# Confirm label type 
print(f"Label datatype: {type(label)}")



# DataLoader and batching

train_loader = DataLoader(training_data, batch_size = 64, shuffle= True)

# Extract one batch 
images, labels = next(iter(train_loader))

print(f"Batch of images shape: {images.shape}")
print(f"Batch of labels shape: {labels.shape}")


# Now we inspect indexing inside batches
print(f"First image in batch shape: {images[0].shape}")
print(f"First label in batch: {labels[0]}")

print(labels[:5])

print(images[0].shape)

print(labels[0], labels[1], labels[2])