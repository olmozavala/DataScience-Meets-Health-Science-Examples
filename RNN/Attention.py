#%%
import torch
import torch.nn as nn
import torch.optim as optim

# Synthetic dataset
def generate_data(num_samples, seq_length):
    inputs = torch.randint(1, 10, (num_samples, seq_length, 1)).float()  # Random integers between 1 and 10
    targets = (inputs[:, 0] + inputs[:, -1]).view(-1, 1)
    return inputs, targets

num_train_samples = 1000
num_test_samples = 100
seq_length = 5

train_inputs, train_targets = generate_data(num_train_samples, seq_length)
test_inputs, test_targets = generate_data(num_test_samples, seq_length)

# Printing the first 5 input and output pairs
for i in range(5):
    print(f"Input: {train_inputs[i].flatten().data.numpy()}, Output (first + last): {train_targets[i].data.numpy()}")

#%%
# Attention mechanism
class Attention(nn.Module):
    def __init__(self, input_dim):
        super(Attention, self).__init__()
        self.Wq = nn.Linear(input_dim, input_dim)
        self.Wk = nn.Linear(input_dim, input_dim)
        self.Wv = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        queries = self.Wq(x)
        keys = self.Wk(x)
        values = self.Wv(x)
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / (x.size(-1) ** 0.5)
        attention = torch.softmax(scores, dim=-1)
        context = torch.matmul(attention, values)
        return context

# Model
class SumFirstLast(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SumFirstLast, self).__init__()
        self.attention = Attention(input_dim)
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.attention(x)
        x = torch.relu(self.fc(x))
        x = self.output(x)
        return x.sum(dim=1)

# Hyperparameters
#%%
input_dim = 1
hidden_dim = 1
learning_rate = 0.001
epochs = 500

# Model, Loss, and Optimizer
model = SumFirstLast(input_dim, hidden_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training
for epoch in range(epochs):
    optimizer.zero_grad()
    output = model(train_inputs)
    loss = criterion(output, train_targets)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch: {epoch+1}, Loss: {loss.item()}")

# Test
with torch.no_grad():
    test_output = model(test_inputs)
    test_loss = criterion(test_output, test_targets)
    print(f"Test Loss: {test_loss.item()}")
#%%
# Test the model
for i in range(5):
    print(f"Input: {test_inputs[i].flatten().data.numpy()}, Output (first + last): {test_targets[i].data.numpy()}")
    output = model(test_inputs[i].unsqueeze(0))
    print(f"Prediction (first + last): {output.data.numpy()}")