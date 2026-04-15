# %%
import torch
import torch.nn as nn

# This is a simple model that learns to predict the 'next' character in a sequence of characters.
# The model is a simple RNN with one hidden layer.
# The input is a sequence of characters, and the output is a probability distribution over the possible characters.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the RNN model
class CharRNN(nn.Module):
    def __init__(self, input_size, emb_size, hidden_size, output_size, n_layers=1):
        super(CharRNN, self).__init__()
        self.hidden_size = hidden_size
        self.emb_size = emb_size
        self.n_layers = n_layers

        self.embed = nn.Embedding(input_size, emb_size)
        # self.rnn = nn.RNN(emb_size, hidden_size, n_layers)
        # self.rnn = nn.GRU(emb_size, hidden_size, n_layers)
        self.rnn = nn.LSTM(emb_size, hidden_size, n_layers)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        embedded = self.embed(input)
        output, hidden = self.rnn(embedded.view(1, 1, -1), hidden)
        output = self.fc(output.view(1, -1))
        return output, hidden

    def init_hidden(self, device="cpu"):
        if isinstance(self.rnn, nn.LSTM):
            return (torch.zeros(self.n_layers, 1, self.hidden_size).to(device),
                    torch.zeros(self.n_layers, 1, self.hidden_size).to(device))
        else:
            return torch.zeros(self.n_layers, 1, self.hidden_size).to(device)

#%%
# Dataset preparation
text = "We will learn about RNNs today. RNNs are powerful models that can be used for many tasks. I hope you will enjoy this example."
chars = list(set(text))
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

print(char_to_idx)

# %%

input_size = len(chars)
emb_size = 8
hidden_size = 10 
output_size = len(chars)
n_layers = 1

model = CharRNN(input_size, emb_size, hidden_size, output_size, n_layers).to(device)

# Training the model
learning_rate = 0.005
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


#%%
n_epochs = 500
for epoch in range(1, n_epochs + 1):
    input_seq = torch.tensor([char_to_idx[ch] for ch in text[:-1]], dtype=torch.long).to(device)
    target_seq = torch.tensor([char_to_idx[ch] for ch in text[1:]], dtype=torch.long).to(device)

    hidden = model.init_hidden(device)
    model.zero_grad()
    loss = 0

    for i in range(input_seq.size(0)):
        output, hidden = model(input_seq[i], hidden)
        loss += criterion(output, target_seq[i].view(1))

    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(f"Epoch: {epoch}, Loss: {loss.item() / input_seq.size(0)}")

#%%
# Test the model
text = "We will learn about RNNs today. RNNs are powerful models that can be used for many tasks. I hope you will enjoy this example."
start_str = "We will learn about RNNs today. RNNs"

hidden = model.init_hidden(device)
input_seq = torch.tensor([char_to_idx[ch] for ch in start_str], dtype=torch.long)

full_pred = ""
for i in range(len(start_str) - 1):
    output, hidden = model(input_seq[i].to(device), hidden)
    full_pred += idx_to_char[output.argmax().item()]
    print("Input: ", idx_to_char[input_seq[i].item()])
    print("Output:" , idx_to_char[output.argmax().item()])

print(f"Input: {start_str}")
print(f"Full prediction: {full_pred}")

# %% Generate the rest of the sequence from previous output
for i in range(10):
    # Convert the previous output's argmax to a tensor for next input
    next_input = torch.tensor([output.argmax().item()], dtype=torch.long).to(device)
    output, hidden = model(next_input, hidden)
    full_pred += idx_to_char[output.argmax().item()]

print(f"Full prediction: {full_pred}")
# %%
