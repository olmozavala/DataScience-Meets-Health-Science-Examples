import torch
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
print(F"Using {device} device")
#%% 
#%% Simple tensor
vec_tensor = torch.tensor([1,3,6,4,2,3,3])
vec_tensor_rand = torch.rand(10)
mat_tensor = torch.Tensor(
    [
     [0, 1, 2],
     [3, 4, 5]
    ]
)

#%% Operations
print(F"Operations with constants (vec_tensor*2 + 1): {vec_tensor*2 + 1}")
print(F"Doc product (vec_tensor_rand.dot(vec_tensor_rand)): {vec_tensor_rand.dot(vec_tensor_rand)}")

#%% Properties
print(F"Tensor: {mat_tensor}")
print(F"Shape: {mat_tensor.shape}")
print(F"Size = {mat_tensor.size()}")
print(F"Number of elements = {mat_tensor.numel()}")

#%% GPU
print(F"Initial Device: {mat_tensor.device}")
gpu_tensor = mat_tensor.to(device)
print(F"Initial Device: {gpu_tensor.device}")

#%% Indexing
print(mat_tensor)
print(F"mat_tensor[1]: {mat_tensor[1]}")
print(F"mat_tensor[1, 1]: {mat_tensor[1, 1]}")
print(F"mat_tensor[1, 1].item(): {mat_tensor[1, 1].item()}")  # Scalar value
print(F"mat_tensor[:, 0]: {mat_tensor[:, 0]}")

#%% Initialization
print(F"Init ones: {torch.ones_like(mat_tensor)}")
print(F"Init zeros: {torch.zeros_like(mat_tensor)}")
print(F"Init random: {torch.randn_like(mat_tensor)}")
print(F"Init random gpu: {torch.randn(2, 2, device='cuda')}")  # Alternatively 'cuda' or 'cpu'

#%% Basic Functions
(mat_tensor - 5) * 2
print(F"Sum of tensor: {mat_tensor.sum()}")
print(F"Flatten of tensor: {mat_tensor.flatten()}")
print(F"Mean of tensor: {mat_tensor.mean()}")
print(F"Std of tensor: {mat_tensor.std()}")
print(F"Tensor Tranpose: {mat_tensor.t()}")
mat_tensor.mean(0)
# Equivalently, you could also write:
# mat_tensor.mean(dim=0)
# mat_tensor.mean(axis=0)
# torch.mean(mat_tensor, 0)
# torch.mean(mat_tensor, dim=0)
# torch.mean(mat_tensor, axis=0)

#%%
print(f"Vector: {mat_tensor}")
print(f"Shape: {mat_tensor.shape}")
print(f"Reshape as row vector: {mat_tensor.view(-1)}")  # Reshape as row vector
print(f"Reshape as column vector: {mat_tensor.view(-1,1)}")  # Reshape as column vector
print(f"Reshape to custom size: {mat_tensor.view(3,2)}")  # Reshape as column vector
print(f"Reshape to custom size: {mat_tensor.view(3,2)}")  # Reshape as column vector
