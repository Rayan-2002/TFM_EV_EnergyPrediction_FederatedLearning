import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# Generate sin wave dataset 

wave = torch.linspace(0, 20, steps = 200)
data = torch.sin(wave)

# print(data.shape)
# print(data)

# Create Pytorch Dataset
class SineWaveDataset(Dataset):
    def __init__(self, data, sequence_length):
        self.data = data
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.data) - self.sequence_length
    
    def __getitem__(self, index):
        x = self.data[index:index + self.sequence_length]
        y = self.data[index + self.sequence_length]

        # LSTM expects: (sequence_length, num_features)
        # Here num_features = 1 because we only have sine value
        x = x.float().unsqueeze(-1)
        y = y.float()

        return x, y
    

# Instantiate the dataset
sequence_length = 10
dataset = SineWaveDataset(data, sequence_length)


print(f"Number of samples in dataset: {len(dataset)}")
sample_X, sample_y = dataset[0]
print(f"Sample X shape: {sample_X.shape}")
print(f"Sample y shape: {sample_y.shape}")

# Create DataLoader
loader = DataLoader(dataset, batch_size=16, shuffle=False)

# Inspect batch shapes
batch_x, batch_y = next(iter(loader))

print(f"Batch X shape: {batch_x.shape}")
print(f"Batch y shape: {batch_y.shape}")

# Create the LSTM model
# inside : nn.LSTM and nn.Linear

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, output_size=1):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size = input_size,
            hidden_size = hidden_size,
            batch_first = True
        )

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):

        output, _ = self.lstm(x)

        last_timestep_output = output[:, -1, :]

        prediction = self.fc(last_timestep_output)

        return prediction.squeeze(-1)
    

def main():
    sequence_length = 30
    batch_size = 32
    epochs = 20
    learning_rate = 0.001

    # Generate sine wave
    x_values = np.linspace(0, 100, 1000)
    data = torch.tensor(np.sin(x_values).astype(np.float32))

    # Split into train and test
    train_size = int(len(data) * 0.8)
    train_data = data[:train_size]
    test_data = data[train_size - sequence_length:]

    # Create datasets
    train_dataset = SineWaveDataset(train_data, sequence_length)
    test_dataset = SineWaveDataset(test_data, sequence_length)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    # Create model
    model = LSTMModel(
        input_size=1,
        hidden_size=32,
        output_size=1
    )

    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print("Training started")
    print("-" * 40)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for x_batch, y_batch in train_loader:
            # x_batch shape: (batch_size, sequence_length, num_features)
            # y_batch shape: (batch_size,)

            predictions = model(x_batch)

            loss = loss_function(predictions, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(train_loader)

        print(f"Epoch {epoch + 1}/{epochs} - Loss: {average_loss:.6f}")

    # Evaluation
    model.eval()

    predictions_list = []
    targets_list = []

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            predictions = model(x_batch)

            predictions_list.extend(predictions.numpy())
            targets_list.extend(y_batch.numpy())

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(targets_list, label="Real sine wave")
    plt.plot(predictions_list, label="Predicted sine wave")
    plt.title("Sine Wave Forecasting with LSTM")
    plt.xlabel("Time step")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()


    


    


