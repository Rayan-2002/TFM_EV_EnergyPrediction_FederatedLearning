import os
import glob

import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


# ============================================================
# 1. Configuration
# ============================================================

DATASET_PATH = "/home/rayan/Internship/SUMO_Barcelona/dataset"

NUM_CLIENTS = 10

T_PAST_SECONDS = 10
T_FUTURE_SECONDS = 5
TIME_STEP_SECONDS = 1.0

BATCH_SIZE = 32


# ============================================================
# 2. Load and aggregate SUMO CSV files
# ============================================================

def load_sumo_csvs(dataset_path):
    """
    Loads a subset of SUMO CSV files and aggregates them into one dataframe.

    This represents the centralized Gold Standard setting:
    all selected client data is collected in one place.
    """

    csv_files = sorted(glob.glob(os.path.join(dataset_path, "client_*.csv")))

    if len(csv_files) == 0:
        raise ValueError(f"No CSV files found in {dataset_path}")

    selected_files = csv_files[:NUM_CLIENTS]

    dataframes = []

    for file_path in selected_files:
        df = pd.read_csv(file_path)

        client_id = os.path.basename(file_path).replace(".csv", "")
        df["client_id"] = client_id

        dataframes.append(df)

    aggregated_df = pd.concat(dataframes, ignore_index=True)

    return aggregated_df


df = load_sumo_csvs(DATASET_PATH)

print("Raw dataframe:")
print(df.head())
print(df.columns)


# ============================================================
# 3. Select and rename useful columns
# ============================================================

column_mapping = {
    "speed": "v",
    "acceleration": "a",
    "road_grade_angle": "theta",
    "consumption_power_watts": "P"
}

df = df.rename(columns=column_mapping)

required_columns = ["v", "a", "theta", "P"]

missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

df = df[required_columns + ["client_id"]]

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()

print("Cleaned dataframe shape:", df.shape)
print(df.head())


# ============================================================
# 4. Train/test split
# ============================================================

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    shuffle=False
)

feature_columns = ["v", "a", "theta", "P"]

scaler = MinMaxScaler()

train_scaled = scaler.fit_transform(train_df[feature_columns])
test_scaled = scaler.transform(test_df[feature_columns])

train_scaled_df = pd.DataFrame(train_scaled, columns=feature_columns)
test_scaled_df = pd.DataFrame(test_scaled, columns=feature_columns)

print("Train features after scaling:")
print(train_scaled_df.head())

print("Test features after scaling:")
print(test_scaled_df.head())


# ============================================================
# 5. Custom PyTorch Dataset
# ============================================================

class SUMOSlidingWindowDataset(Dataset):
    def __init__(
        self,
        dataframe,
        feature_columns,
        target_column="P",
        t_past_seconds=10,
        t_future_seconds=5,
        time_step_seconds=1.0
    ):
        """
        Creates sliding windows for time-series forecasting.

        Input X:
            past T_past seconds of features [v, a, theta, P]

        Target Y:
            average power P over the next T_future seconds
        """

        self.data = dataframe[feature_columns].values.astype(np.float32)

        self.feature_columns = feature_columns
        self.target_column = target_column
        self.target_index = feature_columns.index(target_column)

        self.past_steps = int(t_past_seconds / time_step_seconds)
        self.future_steps = int(t_future_seconds / time_step_seconds)

        if self.past_steps <= 0:
            raise ValueError("past_steps must be greater than 0")

        if self.future_steps <= 0:
            raise ValueError("future_steps must be greater than 0")

        self.total_window_size = self.past_steps + self.future_steps

        if len(self.data) < self.total_window_size:
            raise ValueError(
                f"Dataset too small. Need at least {self.total_window_size} rows, "
                f"but got {len(self.data)} rows."
            )

    def __len__(self):
        return len(self.data) - self.total_window_size + 1

    def __getitem__(self, index):
        past_start = index
        past_end = index + self.past_steps

        future_start = past_end
        future_end = future_start + self.future_steps

        X = self.data[past_start:past_end, :]

        future_power = self.data[
            future_start:future_end,
            self.target_index
        ]

        Y = np.mean(future_power)

        X = torch.tensor(X, dtype=torch.float32)
        Y = torch.tensor(Y, dtype=torch.float32)

        return X, Y


# ============================================================
# 6. Create Dataset and DataLoader
# ============================================================

train_dataset = SUMOSlidingWindowDataset(
    dataframe=train_scaled_df,
    feature_columns=feature_columns,
    target_column="P",
    t_past_seconds=T_PAST_SECONDS,
    t_future_seconds=T_FUTURE_SECONDS,
    time_step_seconds=TIME_STEP_SECONDS
)

test_dataset = SUMOSlidingWindowDataset(
    dataframe=test_scaled_df,
    feature_columns=feature_columns,
    target_column="P",
    t_past_seconds=T_PAST_SECONDS,
    t_future_seconds=T_FUTURE_SECONDS,
    time_step_seconds=TIME_STEP_SECONDS
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("Number of training samples:", len(train_dataset))
print("Number of testing samples:", len(test_dataset))


# ============================================================
# 7. Data sanity check
# ============================================================

for X_batch, Y_batch in train_loader:
    print("X batch shape:", X_batch.shape)
    print("Y batch shape:", Y_batch.shape)

    print("Example X[0]:")
    print(X_batch[0])

    print("Example Y[0]:")
    print(Y_batch[0])

    break


# ============================================================
# 8. LSTM Model Architecture
# ============================================================

class GoldStandardLSTM(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        output_size=1,
        dropout=0.0
    ):
        super(GoldStandardLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.fc = nn.Linear(
            in_features=hidden_size,
            out_features=output_size
        )

    def forward(self, x):
        lstm_out, (hidden_state, cell_state) = self.lstm(x)

        last_hidden_state = lstm_out[:, -1, :]

        prediction = self.fc(last_hidden_state)

        prediction = prediction.squeeze(-1)

        return prediction


# ============================================================
# 9. Model initialization
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_size = len(feature_columns)
hidden_size = 64
num_layers = 1
output_size = 1
dropout = 0.0

model = GoldStandardLSTM(
    input_size=input_size,
    hidden_size=hidden_size,
    num_layers=num_layers,
    output_size=output_size,
    dropout=dropout
).to(device)

print(model)
print("Using device:", device)


# ============================================================
# 10. Model sanity check
# ============================================================

for X_batch, Y_batch in train_loader:
    X_batch = X_batch.to(device)
    Y_batch = Y_batch.to(device)

    Y_pred = model(X_batch)

    print("X batch shape:", X_batch.shape)
    print("Y batch shape:", Y_batch.shape)
    print("Y pred shape:", Y_pred.shape)

    break



# ============================================================
# 11. Training configuration
# ============================================================

LEARNING_RATE = 0.001
NUM_EPOCHS = 50

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# 12. Centralized training loop
# ============================================================

train_losses = []
test_losses = []

for epoch in range(NUM_EPOCHS):
    # ----------------------------
    # Training phase
    # ----------------------------
    model.train()

    running_train_loss = 0.0

    for X_batch, Y_batch in train_loader:
        X_batch = X_batch.to(device)
        Y_batch = Y_batch.to(device)

        # Reset gradients
        optimizer.zero_grad()

        # Forward pass
        Y_pred = model(X_batch)

        # Compute loss
        loss = criterion(Y_pred, Y_batch)

        # Backpropagation
        loss.backward()

        # Update model parameters
        optimizer.step()

        running_train_loss += loss.item()

    average_train_loss = running_train_loss / len(train_loader)
    train_losses.append(average_train_loss)

    # ----------------------------
    # Test phase
    # ----------------------------
    model.eval()

    running_test_loss = 0.0

    with torch.no_grad():
        for X_batch, Y_batch in test_loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            Y_pred = model(X_batch)

            loss = criterion(Y_pred, Y_batch)

            running_test_loss += loss.item()

    average_test_loss = running_test_loss / len(test_loader)
    test_losses.append(average_test_loss)

    train_rmse = np.sqrt(average_train_loss)
    test_rmse = np.sqrt(average_test_loss)

    print(
        f"Epoch [{epoch + 1}/{NUM_EPOCHS}] "
        f"Train MSE: {average_train_loss:.6f} | "
        f"Train RMSE: {train_rmse:.6f} | "
        f"Test MSE: {average_test_loss:.6f} | "
        f"Test RMSE: {test_rmse:.6f}"
    )


# ============================================================
# 13. Final evaluation
# ============================================================

final_train_mse = train_losses[-1]
final_test_mse = test_losses[-1]

final_train_rmse = np.sqrt(final_train_mse)
final_test_rmse = np.sqrt(final_test_mse)

print("\nFinal results:")
print(f"Final Train RMSE: {final_train_rmse:.6f}")
print(f"Final Test RMSE: {final_test_rmse:.6f}")

import matplotlib.pyplot as plt

# ============================================================
# 14. Plot training and test loss
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(train_losses, label="Train MSE")
plt.plot(test_losses, label="Test MSE")

plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training and Test Loss During Centralized Training")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()