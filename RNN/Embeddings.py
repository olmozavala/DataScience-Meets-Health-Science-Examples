# %%
%matplotlib notebook
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
from mpl_toolkits.mplot3d import Axes3D
import matplotlib

# Define the vocabulary size and the embedding dimension
vocab_size = 10
embedding_dim = 3

# Create the embedding layer
class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Word2Vec, self).__init__()
        self.embedding_layer = nn.Embedding(vocab_size, embedding_dim)
        # Add a linear layer to project the embeddings to a lower dimension
        self.linear_layer = nn.Linear(embedding_dim, embedding_dim)

        
    def forward(self, x):
        return self.embedding_layer(x)

embedding_layer = Word2Vec(vocab_size, embedding_dim).embedding_layer
with torch.no_grad():
    embedding_layer.weight.data *= 0.1
# Example indices of words in the vocabulary
word_indices = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.long)
# Get the embeddings for the word indices
word_embeddings = embedding_layer(word_indices)
# Print the embeddings
print(word_embeddings)

#%% A function to plot the embeddings in 3D (showing relationships)
def plot_embeddings(embeddings, words):
    # Plot the 3D embeddings
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    scatter_points = []
    
    for i, word in enumerate(words):
        x, y, z = embeddings[i].detach().numpy()
        scatter = ax.scatter(x, y, z, label=word, picker=True)
        ax.text(x, y, z, word)
        scatter_points.append(scatter)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.legend()

    def on_pick(event):
        ind = event.ind[0]
        print(f'Selected word: {words[ind]}')
        print(f'Coordinates: {embeddings[ind].detach().numpy()}')

    fig.canvas.mpl_connect('pick_event', on_pick)
    
    # Enable rotation and zooming
    ax.mouse_init()
    
    # Show the plot
    plt.show()

words = ['dog', 'cat', 'horse', 'house', 'car', 'wolf', 'tiger', 'zebra', 'building', 'vehicle']
plot_embeddings(word_embeddings, words)

# %%
# Define the loss function (Mean Squared Error in this case)
optimizer = optim.SGD(embedding_layer.parameters(), lr=0.01)

# Define some simple context pairs (word, context_word)
context_pairs = [
    # Similar animals
    ('dog', 'cat'), ('cat', 'dog'),
    ('dog', 'horse'), ('horse', 'dog'),
    ('cat', 'horse'), ('horse', 'cat'),
    ('dog', 'wolf'), ('wolf', 'dog'),
    ('cat', 'tiger'), ('tiger', 'cat'),
    ('horse', 'zebra'), ('zebra', 'horse'),
    # Add more animal relationships
    ('wolf', 'tiger'), ('tiger', 'wolf'),
    ('zebra', 'horse'), ('horse', 'zebra'),
    
    # Buildings/vehicles 
    ('house', 'car'), ('car', 'house'),
    ('house', 'building'), ('building', 'house'),
    ('car', 'vehicle'), ('vehicle', 'car'),
    # Add more building/vehicle relationships
    ('building', 'vehicle'), ('vehicle', 'building'),
    ('car', 'building'), ('building', 'car'),
    
    # Negative examples (dissimilar pairs)
    ('dog', 'car'), ('car', 'dog'),
    ('house', 'tiger'), ('tiger', 'house'),
    ('zebra', 'vehicle'), ('vehicle', 'zebra')
]

# Create word to index mapping
word_to_idx = {word: idx for idx, word in enumerate(words)}
idx_to_word = {idx: word for idx, word in enumerate(words)}

# Convert pairs to tensor indices
input_indices = torch.tensor([word_to_idx[pair[0]] for pair in context_pairs])
target_indices = torch.tensor([word_to_idx[pair[1]] for pair in context_pairs])

for i in range(len(context_pairs)):
    print(f"Related words: {idx_to_word[input_indices[i].item()]} and {idx_to_word[target_indices[i].item() ]}")

# %%

# Define the loss function (using cross entropy with a simple dot product similarity)
def get_loss(input_indices, target_indices):
    # Get embeddings for input words
    input_embeds = embedding_layer(input_indices)
    
    # Compute similarity scores between input and all possible targets
    # We need to compute similarity with ALL possible targets for CrossEntropyLoss
    all_targets = embedding_layer.weight  # Shape: [vocab_size, embedding_dim]
    
    # Compute similarities between each input and ALL possible targets
    # Using matrix multiplication for efficient computation
    similarities = torch.matmul(input_embeds, all_targets.T)  # Shape: [batch_size, vocab_size]
    
    # CrossEntropyLoss includes softmax internally, so we don't need to apply it explicitly
    return nn.CrossEntropyLoss()(similarities, target_indices)

# Train the embedding layer
epochs = 10000
losses = []
for i in range(epochs):
    optimizer.zero_grad()
    loss = get_loss(input_indices, target_indices)
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    if i % 100 == 0:
        print(f"Epoch: {i}, Loss: {loss.item()}")

# Plot loss curve
plt.figure(figsize=(10, 5))
plt.plot(losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()


# %% Plot all the distances between the words
# Create list to store word pairs and distances
distances = []

for i in range(len(words)):
    for j in range(i+1, len(words)):
        # Get embeddings and squeeze to remove extra dimension
        word1_embed = embedding_layer(torch.tensor([i])).squeeze()
        word2_embed = embedding_layer(torch.tensor([j])).squeeze()
        # Calculate cosine similarity since this better represents semantic relationships
        cos_sim = torch.nn.functional.cosine_similarity(word1_embed.unsqueeze(0), word2_embed.unsqueeze(0))
        distance = 1 - cos_sim.item()  # Convert to distance (0 = similar, 2 = dissimilar)
        distances.append((words[i], words[j], distance))

# Sort distances
distances.sort(key=lambda x: x[2])

# Print sorted distances
print("Word pair distances (0 = most similar, 2 = most different):")
for word1, word2, dist in distances:
    print(f"Distance between {word1} and {word2}: {dist:.2f}")

# Plot final embeddings using the same embeddings
embeddings = embedding_layer(word_indices[0:3])
plot_embeddings(embeddings, words[0:3])
# Print the embeddings for the first 3 words
print(embedding_layer(torch.tensor([0, 1, 2])).squeeze())
# %%
