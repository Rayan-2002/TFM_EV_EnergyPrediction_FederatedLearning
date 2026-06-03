import pandas as pd
import torch 
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader


# load CSV file
df = pd.read_csv("/home/rayan/Internship/SUMO_Barcelona/dataset/client_1.csv")

# Extracting data 

data = df["consumption_power_watts"].values

# Normalize the data
data = (data - np.min(data)) / (np.max(data) - np.min(data))

# Convert to Tensor
data = torch.tensor(data, dtype=torch.float32)

# Create sequences and targets
window_size = 10  # Using past 10 time steps to predict the next time step
# changed from 5 to 10 and saw a little improvement

def create_sequences(data, window_size):
    sequences = []
    targets = []
    
    for i in range(len(data) - window_size):
        seq = data[i:i+window_size]
        target = data[i+window_size]
        
        sequences.append(seq)
        targets.append(target)
    
    return torch.stack(sequences), torch.stack(targets)


# Create Dataset class
class EnergyDataset(Dataset):
    def __init__(self, data, window_size):
        self.sequences, self.targets = create_sequences(data, window_size)

    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        x = self.sequences[idx].unsqueeze(-1)  # Add feature dimension
        y = self.targets[idx].float()
        return x, y
    


# Train / Split 
split = int(0.8 * len(data)) # This variable is used to split the dataset into training and testing sets. It calculates the index at which to split the data, using 80% of the data for training and 20% for testing.
train_data = data[:split]
test_data = data[split - window_size:]

train_dataset = EnergyDataset(train_data, window_size)
test_dataset = EnergyDataset(test_data, window_size)



train_loader = DataLoader(train_dataset, batch_size=32, shuffle= False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle= False)


import torch.nn as nn

class EnergyLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out.squeeze(-1)

model = EnergyLSTM(input_size=1, hidden_size=64)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# Changed from 0.001 to 0.01 and saw a significant improvement in convergence speed

epochs = 30

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for x_batch, y_batch in train_loader:
        pred = model(x_batch)
        loss = criterion(pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")


# Evaluation
model.eval()

preds, targets = [], []

with torch.no_grad():
    for x_batch, y_batch in test_loader:
        pred = model(x_batch)

        preds.extend(pred.numpy())
        targets.extend(y_batch.numpy())


# Visualization

plt.plot(preds, label="Predicted")
plt.plot(targets, label="Actual")
plt.legend()
plt.title("Energy Consumption Prediction")
plt.xlabel("Time Step")
plt.ylabel("Normalized Consumption")
plt.show()

