# %%
import torch
import torch.nn as nn
import numpy as np

input_size = 1 # embedding size
hidden_size = 1 # hidden size
num_layers = 1 # number of layers
batch_size = 1 # batch size

rnn = nn.RNN(input_size, hidden_size, num_layers)
# rnn = nn.GRU(input_size, hidden_size, num_layers)
# rnn = nn.LSTM(input_size, hidden_size, num_layers)

print(f"Input size: {input_size}, Hidden size: {hidden_size}, Number of layers: {num_layers}")
total_weights = sum(p.numel() for p in rnn.parameters() if p.requires_grad)
if isinstance(rnn, nn.RNN):
    print(f"Number of weights in the model (Wx, Wh, bh, bi): {total_weights}")
    # Print tha values of the weights
    print(f"Wx: {rnn.weight_ih_l0}")
    print(f"Wh: {rnn.weight_hh_l0}")
    print(f"bh: {rnn.bias_ih_l0}")
    print(f"bi: {rnn.bias_hh_l0}")
elif isinstance(rnn, nn.GRU):
    print(f"Number of weights in the model: {total_weights}")
   
#%%
x = torch.ones(batch_size, input_size)*1
x2 = torch.ones(batch_size, input_size)*1
h = torch.ones(num_layers, hidden_size)*0

# Showing that it is the same to call the rnn twice than to call it once with two inputs
print("------------- Calling the RNN first time -------------")
output, hn = rnn(x, h)
print(f"Input x_t: {x.data}, value h_t-1: {h.data}")
# The output size is (1,5) because it matches the hidden_size parameter (5), not the input_size (3)
# The RNN maps the input to the hidden dimension at each step
print(f"Output size: {output.size()}, value O_t: {output.data}  # Output dim matches hidden_size=5")
print(f"Hidden state size: {hn.size()}, value h_t: {hn.data}")
print("------------- Calling the RNN second time -------------")
output, hn = rnn(x2, hn)
print(f"Input x_t: {x2.data}, value h_t-1: {hn.data}")
print(f"Output size: {output.size()}, value O_t: {output.data}")
print(f"Hidden state size: {hn.size()}, value h_t: {hn.data}")
print("------------- Calling the RNN with two inputs --------------")
x3 = torch.cat((x, x2), dim=0)
output, hn = rnn(x3, h)
print(f"Input x_t: {x3.data}, value h_t-1: {h.data}")
print(f"Output size: {output.size()}, value O_t: {output.data}")
print(f"Hidden state size: {hn.size()}, value h_t: {hn.data}")

#%% BiRNNs
input_size = 16  # embedding size
hidden_size = 8 # hidden size
num_layers = 2 # number of layers
batch_size = 1 # batch size
sequence_size = 4  # number of words in a sentence
bidirectional = False

rnn = nn.RNN(input_size, hidden_size, num_layers, bidirectional=bidirectional)

print(f"Input size: {input_size}, Hidden size: {hidden_size}, Number of layers: {num_layers}")
total_weights = sum(p.numel() for p in rnn.parameters() if p.requires_grad)
print(f"Number of weights in the model: {total_weights}")

if bidirectional:
    sequence_size = sequence_size * 2

x = torch.ones(batch_size, sequence_size, input_size)
h = torch.ones(num_layers, sequence_size, hidden_size)

output, hn = rnn(x, h)
print(f"Input x_t: {x.data.size()}, value h_t-1: {h.data.size()}")
print(f"Output size: {output.size()}")
print(f"Hidden state size: {hn.size()}")