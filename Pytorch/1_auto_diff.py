import torch

# Points p1(1,2) and p2(2,4)
x = torch.tensor([1.0, 2.0], dtype=torch.float32)
y = torch.tensor([2.0, 4.0], dtype=torch.float32)

# Initial weights and bias with gradient tracking
w = torch.tensor(1.8, dtype=torch.float32, requires_grad=True)
b = torch.tensor(0.1, dtype=torch.float32, requires_grad=True)

print("--- Initial Values ---")
print(f"p1: (1, 2), p2: (2, 4)")
print(f"w: {w.item()}, b: {b.item()}\n")

# Linear model: y_hat = w * x + b
y_hat = w * x + b

# Calculate Loss (Mean Squared Error)
# L = 1/n * Σ (y_i - y_hat_i)^2
loss = torch.mean((y - y_hat)**2)

# Compute gradients using backpropagation
loss.backward()

print("--- Results ---")
print(f"Predictions: {y_hat.detach().numpy()}")
print(f"Loss (MSE): {loss.item():.4f}")
print(f"dL/dw (gradient with respect to w): {w.grad.item():.4f}")
print(f"dL/db (gradient with respect to b): {b.grad.item():.4f}")

# Verification of manually derived gradients:
# dL/dw = -2/n * Σ x_i * (y_i - y_hat_i)
# dL/db = -2/n * Σ (y_i - y_hat_i)
