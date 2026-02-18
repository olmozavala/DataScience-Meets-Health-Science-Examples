import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

def plot_current_fit(X, y, model, epoch, loss):
    plt.cla()  # Clear current axes
    plt.scatter(X, y, label="Data points", alpha=0.5)
    plt.plot(X, model(X).detach(), color="red", label=f"Learned Line (Epoch {epoch})")
    plt.title(f"Training Epoch: {epoch} | Loss: {loss:.4f}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.pause(0.01)  # Brief pause to allow the plot to update

# 1. Generate synthetic data: y = 2x + 5 + noise
X = torch.randn(100, 1)
y = 2 * X + 5 + 0.2 * torch.randn(100, 1)

# 2. Simplest Model: A single neuron (equivalent to y = wx + b)
model = nn.Linear(1, 1)

# 3. Setup: MSE Loss and SGD Optimizer
optimizer = optim.SGD(model.parameters(), lr=0.1)
criterion = nn.MSELoss()

# 4. Training Loop
plt.ion()  # Turn on interactive mode for real-time plotting
fig = plt.figure(figsize=(10, 6))

print("Training started...")
for epoch in range(100):
    # Forward pass
    y_pred = model(X)
    loss = criterion(y_pred, y)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Call plot function every epoch
    plot_current_fit(X, y, model, epoch + 1, loss.item())
    
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}/100 - Loss: {loss.item():.4f}")

# Print final learned parameters
plt.ioff()  # Turn off interactive mode
w, b = model.weight.item(), model.bias.item()
print(f"\nLearned: weight={w:.4f}, bias={b:.4f}")
print(f"True:    weight=2.0000, bias=5.0000")

plt.show()  # Keep the final plot open
