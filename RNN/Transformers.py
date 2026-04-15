# %%
# Import the necessary libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the MultiHeadAttention module
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()

        # Initialize the number of heads and the model dimension
        self.num_heads = num_heads
        self.d_model = d_model

        # Ensure that the model dimension is divisible by the number of heads
        assert d_model % self.num_heads == 0

        # Compute the depth of each head
        self.depth = d_model // self.num_heads

        # Define the linear layers for the query, key, and value projections
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)

        # Define the output linear layer
        self.dense = nn.Linear(d_model, d_model)

    # Define the split_heads function to reshape the input tensor to split the heads
    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)

    # Define the forward function to compute the attention scores and output tensor
    def forward(self, query, key, value):
        batch_size = query.shape[0]

        # Project the query, key, and value tensors
        query = self.split_heads(self.wq(query), batch_size)
        key = self.split_heads(self.wk(key), batch_size)
        value = self.split_heads(self.wv(value), batch_size)

        # Compute the attention scores
        matmul_qk = torch.matmul(query, key.permute(0, 1, 3, 2))
        d_k = query.size(-1)
        scaled_attention_logits = matmul_qk / torch.sqrt(d_k)

        # Apply the softmax function to get the attention weights
        attention_weights = torch.nn.functional.softmax(scaled_attention_logits, dim=-1)

        # Compute the output tensor
        output = torch.matmul(attention_weights, value)

        # Define other modules and functions as neede
        matmul_qk = torch.matmul(query, key.permute(0, 1, 3, 2))

        d_k = query.size(-1)
        scaled_attention_logits = matmul_qk / torch.sqrt(d_k)

        attention_weights = torch.nn.functional.softmax(scaled_attention_logits, dim=-1)

        output = torch.matmul(attention_weights, value)

        output = output.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.d_model)

        output = self.dense(output)

        return output, attention_weights

# %%
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model)
        )

        self.layernorm1 = nn.LayerNorm((d_model))
        self.layernorm2 = nn.LayerNorm((d_model))

    def forward(self, x):
        attn_output, _ = self.mha(x, x, x)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)
        out2 = self.layernorm2(out1 + ffn_output)
        return out2

# %%
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads):
        super(DecoderLayer, self).__init__()
        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model)
        )

        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.layernorm3 = nn.LayerNorm(d_model)

    def forward(self, x, enc_output):
        attn1, _ = self.mha1(x, x, x)
        out1 = self.layernorm1(attn1 + x)
        
        attn2, _ = self.mha2(out1, enc_output, enc_output)
        out2 = self.layernorm2(attn2 + out1)

        ffn_output = self.ffn(out2)
        out3 = self.layernorm3(ffn_output + out2)
        return out3

# %%
class Transformer(nn.Module):
    def __init__(self, d_model, num_heads, num_encoder_layers, num_decoder_layers):
        super(Transformer, self).__init__()

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads) for _ in range(num_encoder_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads) for _ in range(num_decoder_layers)])

    def forward(self, inp, tar):
        enc_output = inp
        for layer in self.encoder_layers:
            enc_output = layer(enc_output)

        dec_output = tar
        for layer in self.decoder_layers:
            dec_output = layer(dec_output, enc_output)

        return dec_output

# %%
# Example usage
d_model = 512
num_heads = 8
num_encoder_layers = 3
num_decoder_layers = 3

transformer = Transformer(d_model, num_heads, num_encoder_layers, num_decoder_layers)

# Dummy data
inp = torch.randn((32, 10, d_model))  # Batch size: 32, Sequence length: 10, Embedding dimension: 512
tar = torch.randn((32, 10, d_model))

output = transformer(inp, tar)
print(output.shape)  # Should print torch.Size([32, 10, 512])