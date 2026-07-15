from pathlib import Path
from collections import OrderedDict
from typing import List, Tuple

import flwr as fl
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "dataset"

FEATURE_COLUMNS = [
    "speed",
    "acceleration",
    "road_grade_angle",
    "consumption_power_watts",
]


TARGET_COLUMN = "consumption_power_watts"

T_PAST = 30 # Number of past time steps to consider for prediction
T_FUTURE = 5 # Number of future time steps to predict

BATCH_SIZE = 64
HIDDEN_SIZE = 32
LEARNING_RATE = 0.001
TRAIN_SPLIT = 0.8

class PowerLSTMForecaster(nn.Module):
    """
    LSTM model for energy prediction.

    Input shape:
        [batch_size, 30, 4]

    Output shape:
        [batch_size, 1]
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int = 1):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
        )

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)

        # Keep the last output of the sequence.
        # It summarizes the 30 past seconds.
        last_output = lstm_out[:, -1, :]

        prediction = self.fc(last_output)

        return prediction


class SUMOPowerWindowDataset(Dataset):
    """
    Converts one client CSV into temporal windows.

    Example:
        X = 30 past seconds
        y = average consumption over the next 5 seconds
    """

    def __init__(self, csv_path: Path, t_past: int = 30, t_future: int = 5):
        self.csv_path = csv_path
        self.t_past = int(t_past)
        self.t_future = int(t_future)

        if not self.csv_path.exists():
            raise FileNotFoundError(f"Client data file not found: {self.csv_path}")

        df = pd.read_csv(self.csv_path)

        required_columns = list(dict.fromkeys(FEATURE_COLUMNS + [TARGET_COLUMN]))

        missing_columns = [
            column for column in required_columns
            if column not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing columns in {self.csv_path}: {missing_columns}\n"
                f"Available columns: {list(df.columns)}"
            )

        df = df.dropna(subset=required_columns).reset_index(drop=True)

        if len(df) < self.t_past + self.t_future:
            raise ValueError(
                f"Not enough rows in {self.csv_path}. "
                f"Need at least {self.t_past + self.t_future} rows, "
                f"but found {len(df)}."
            )

        features = df[FEATURE_COLUMNS].values.astype(np.float32)
        target = df[TARGET_COLUMN].values.astype(np.float32)

        X_windows = []
        y_values = []

        max_start = len(df) - self.t_past - self.t_future + 1

        for start_idx in range(max_start):
            past_start = start_idx
            past_end = start_idx + self.t_past

            future_start = past_end
            future_end = past_end + self.t_future

            X = features[past_start:past_end]
            y = target[future_start:future_end].mean()

            X_windows.append(X)
            y_values.append([y])

        self.X_windows = torch.tensor(np.array(X_windows), dtype=torch.float32)
        self.y_values = torch.tensor(np.array(y_values), dtype=torch.float32)

    def __len__(self):
        return len(self.X_windows)

    def __getitem__(self, idx):
        return self.X_windows[idx], self.y_values[idx]


def load_client_data(client_id: int) -> Tuple[DataLoader, DataLoader, int, int]:
    """
    Load one client's local CSV and create LSTM temporal windows.

    client_id = 0 loads dataset/client_0.csv
    client_id = 1 loads dataset/client_1.csv
    etc.
    """

    data_path = DATASET_DIR / f"client_{client_id}.csv"

    dataset = SUMOPowerWindowDataset(
        csv_path=data_path,
        t_past=T_PAST,
        t_future=T_FUTURE,
    )

    train_size = int(TRAIN_SPLIT * len(dataset))
    val_size = len(dataset) - train_size

    if train_size == 0 or val_size == 0:
        raise ValueError(
            f"Client {client_id} does not have enough temporal windows. "
            f"Total windows: {len(dataset)}"
        )

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    trainloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    valloader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    input_dim = len(FEATURE_COLUMNS)
    num_examples = len(dataset)

    return trainloader, valloader, input_dim, num_examples


def get_parameters(model: nn.Module) -> List[np.ndarray]:
    """
    Convert PyTorch model weights to NumPy arrays for Flower.
    """

    return [
        val.detach().cpu().numpy()
        for _, val in model.state_dict().items()
    ]


def set_parameters(model: nn.Module, parameters: List[np.ndarray]) -> None:
    """
    Load Flower parameters into the PyTorch model.
    """

    params_dict = zip(model.state_dict().keys(), parameters)

    state_dict = OrderedDict(
        {
            key: torch.tensor(value, device=DEVICE)
            for key, value in params_dict
        }
    )

    model.load_state_dict(state_dict, strict=True)


def train(model: nn.Module, trainloader: DataLoader, epochs: int = 1) -> float:
    """
    Train locally on one client's temporal windows.
    """

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model.train()
    model.to(DEVICE)

    total_loss = 0.0

    for _ in range(epochs):
        for X_batch, y_batch in trainloader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            optimizer.zero_grad()

            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(X_batch)

    return total_loss / len(trainloader.dataset)


def evaluate(model: nn.Module, valloader: DataLoader) -> float:
    """
    Evaluate locally on one client's validation windows.
    """

    criterion = nn.MSELoss()

    model.eval()
    model.to(DEVICE)

    total_loss = 0.0

    with torch.no_grad():
        for X_batch, y_batch in valloader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)

            total_loss += loss.item() * len(X_batch)

    return total_loss / len(valloader.dataset)


class FlowerClient(fl.client.NumPyClient):
    """
    Flower client for one local Barcelona route.
    """

    def __init__(self, client_id: int):
        self.client_id = int(client_id)

        (
            self.trainloader,
            self.valloader,
            self.input_dim,
            self.num_examples,
        ) = load_client_data(self.client_id)

        self.model = PowerLSTMForecaster(
            input_size=self.input_dim,
            hidden_size=HIDDEN_SIZE,
            output_size=1,
        ).to(DEVICE)

        print(
            f"[Client {self.client_id}] Loaded "
            f"{DATASET_DIR / f'client_{self.client_id}.csv'} "
            f"with {self.num_examples} temporal windows"
        )

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        set_parameters(self.model, parameters)

        local_epochs = int(config.get("local_epochs", 1))

        train_loss = train(
            model=self.model,
            trainloader=self.trainloader,
            epochs=local_epochs,
        )

        train_rmse = train_loss ** 0.5

        return (
            get_parameters(self.model),
            len(self.trainloader.dataset),
            {
                "client_id": self.client_id,
                "train_loss": float(train_loss),
                "train_rmse": float(train_rmse),
            },
        )

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)

        val_loss = evaluate(
            model=self.model,
            valloader=self.valloader,
        )

        val_rmse = val_loss ** 0.5

        return (
            float(val_loss),
            len(self.valloader.dataset),
            {
                "client_id": self.client_id,
                "val_loss": float(val_loss),
                "val_rmse": float(val_rmse),
            },
        )