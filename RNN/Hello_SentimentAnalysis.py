import torch
import torch.nn as nn
import torch.optim as optim
from torchtext import  datasets
from torch.utils.data import Dataset, DataLoader

# Load the IMDb dataset
# train_data, test_data = datasets.IMDB(root = "/home/olmozavala/Dropbox/MyCourses/2023/ISC_4933_5935_DataScience_meets_HealthSciences/DataSets/IMDB/",
#                                       split = ('train', 'test'))
# train_data = datasets.IMDB(root = "/home/olmozavala/Dropbox/MyCourses/2023/ISC_4933_5935_DataScience_meets_HealthSciences/DataSets/IMDB/",
#                                       split = 'train')
train_data = datasets.IMDB(split = 'train')


train_loader = DataLoader(train_data, batch_size=50, shuffle=True)
# test_data = DataLoader(test_data, batch_size=5, shuffle=True)

#%%
for i, c_batch in enumerate(train_loader):
    print(f"---------------- Batch {i}----------------")
    x, y = c_batch
    for j in range(len(x)):
        print(x[j].item())
        # print(y[j])

    # if i > 30:
    #     break

#%%
#
# class SentimentClassifier(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size, num_layers, bidirectional):
#         super(SentimentClassifier, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.bidirectional = bidirectional
#         self.num_directions = 2 if bidirectional else 1
#
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)
#         self.fc = nn.Linear(hidden_size * self.num_directions, output_size)
#
#     def forward(self, x):
#         h0 = torch.zeros(self.num_layers * self.num_directions, x.size(0), self.hidden_size)
#         c0 = torch.zeros(self.num_layers * self.num_directions, x.size(0), self.hidden_size)
#
#         out, _ = self.lstm(x, (h0, c0))
#         out = self.fc(out[:, -1, :])
#         return out
#
# # Create iterators
# BATCH_SIZE = 64
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
# train_iterator, test_iterator = BucketIterator.splits(
#     (train_data, test_data),
#     batch_size=BATCH_SIZE,
#     device=device,
#     sort_within_batch=True,
#     sort_key=lambda x: len(x.text)
# )
#
# model = SentimentClassifier(input_size=300, hidden_size=128, output_size=1, num_layers=1, bidirectional=True).to(device)
#
# # Define loss function and optimizer
# criterion = nn.BCEWithLogitsLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
#
# # Training loop
# NUM_EPOCHS = 10
#
# for epoch in range(NUM_EPOCHS):
#     epoch_loss = 0
#     epoch_acc = 0
#
#     for batch in train_iterator:
#         optimizer.zero_grad()
#
#         # Get input sequence and lengths
#         text, text_lengths = batch.text
#
#         # Forward pass
#         predictions = model(text).squeeze(1)
#         # Calculate loss
#         loss = criterion(predictions, batch.label)
#
#         # Calculate accuracy
#         acc = ((torch.sigmoid(predictions) > 0.5).float() == batch.label).float().mean()
#
#         # Backward pass
#         loss.backward()
#         optimizer.step()
#
#         epoch_loss += loss.item()
#         epoch_acc += acc.item()
#
#     print(f"Epoch: {epoch+1}, Loss: {epoch_loss / len(train_iterator):.3f}, Acc: {epoch_acc / len(train_iterator):.3f}")
