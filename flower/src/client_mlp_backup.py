from pathlib import Path
from collections import OrderedDict
from typing import List, Tuple

import flwr as fl
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TARGET_COLUMN = "consumption_power_watts"

BATCH_SIZE = 64
TRAIN_SPLIT = 0.8


class EnergyModel(nn.Module):
    """
    Temporary MLP model.

    This is useful to test Flower mechanics, but it is not the final model.
    The final federated model should reuse the centralized LSTM architecture.
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)


def load_client_data(client_id: int) -> Tuple[DataLoader, DataLoader, int, int]:
    """
    Load only the local CSV of one client.

    Example:
        client_id = 3
        file loaded = dataset/client_3.csv
    """

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    data_path = PROJECT_ROOT / "dataset" / f"client_{client_id}.csv"

    if not data_path.exists():
        raise FileNotFoundError(f"Client data file not found: {data_path}")

    df = pd.read_csv(data_path)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found in {data_path}. "
            f"Available columns: {list(df.columns)}"
        )

    numeric_df = df.select_dtypes(include=[np.number]).copy()

    if TARGET_COLUMN not in numeric_df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' is not numeric in {data_path}."
        )

    X = numeric_df.drop(columns=[TARGET_COLUMN]).values.astype(np.float32)
    y = numeric_df[TARGET_COLUMN].values.astype(np.float32).reshape(-1, 1)

    dataset = TensorDataset(torch.tensor(X), torch.tensor(y))

    train_size = int(TRAIN_SPLIT * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(
        dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    input_dim = X.shape[1]
    num_examples = len(dataset)

    return train_loader, valloader, input_dim, num_examples


def get_parameters(model: nn.Module) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model: nn.Module, parameters: List[np.ndarray]) -> None:
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict(
        {key: torch.tensor(value) for key, value in params_dict}
    )
    model.load_state_dict(state_dict, strict=True)


def train(model: nn.Module, train_loader: DataLoader, epochs: int = 1) -> float:
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    model.to(DEVICE)

    total_loss = 0.0

    for _ in range(epochs):
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(X_batch)

    return total_loss / len(train_loader.dataset)


def evaluate(model: nn.Module, valloader: DataLoader) -> float:
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
    One Flower client = one local Barcelona route.
    """

    def __init__(self, client_id: int):
        self.client_id = int(client_id)

        (
            self.trainloader,
            self.valloader,
            self.input_dim,
            self.num_examples,
        ) = load_client_data(self.client_id)

        self.model = EnergyModel(input_dim=self.input_dim).to(DEVICE)

        print(
            f"[Client {self.client_id}] Loaded dataset/client_{self.client_id}.csv "
            f"with {self.num_examples} local examples"
        )

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        set_parameters(self.model, parameters)

        local_epochs = int(config.get("local_epochs", 1))

        train_loss = train(
            model=self.model,
            train_loader=self.trainloader,
            epochs=local_epochs,
        )

        return (
            get_parameters(self.model),
            len(self.trainloader.dataset),
            {
                "client_id": self.client_id,
                "train_loss": float(train_loss),
            },
        )

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)

        val_loss = evaluate(
            model=self.model,
            valloader=self.valloader,
        )

        return (
            float(val_loss),
            len(self.valloader.dataset),
            {
                "client_id": self.client_id,
                "val_loss": float(val_loss),
            },
        )