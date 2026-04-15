# %%
import torch
from torchtext.vocab import GloVe

# Load the pre-trained GloVe embeddings (with 300-dimensional vectors)
glove = GloVe(name='6B', dim=50)

# Get the embeddings for some example words
word1 = "apple"
word2 = "pizza"
word3 = "orange"
word4 = "car"
word5 = "vehicle"

word1_embedding = glove[word1]
word2_embedding = glove[word2]
word3_embedding = glove[word3]
word4_embedding = glove[word4]
word5_embedding = glove[word5]

print(f"Embedding for '{word1}':\n{word1_embedding}")
print(f"Embedding for '{word2}':\n{word2_embedding}")
print(f"Embedding for '{word3}':\n{word3_embedding}")
print(f"Similarity between '{word1}' and '{word2}': {torch.norm(word1_embedding - word2_embedding):0.3f}")
print(f"Similarity between '{word1}' and '{word3}': {torch.norm(word1_embedding - word3_embedding):0.3f}")
print(f"Similarity between '{word1}' and '{word4}': {torch.norm(word1_embedding - word4_embedding):0.3f}")
print(f"Similarity between '{word4}' and '{word5}': {torch.norm(word4_embedding - word5_embedding):0.3f}")

#%%
relationship = glove["men"] - glove["king"]
comparison = glove["women"] - glove.vectors
distances = torch.norm(comparison - relationship, dim=1)

# Get the indices of the closest words
_, indices = torch.sort(distances)
# Print the top 10 closest word
for i in range(10):
    print(f"Closest word to women in the same direction as men-king: {glove.itos[indices[i]]}")
#%%
