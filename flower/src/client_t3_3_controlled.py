from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import List

import flwr as fl
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PowerLSTMForecaster(nn.Module):
    def __init__(self, input_size: int = 4, hidden_size: int = 32):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        return self.fc(output[:, -1, :])


def get_parameters(model: nn.Module) -> List[np.ndarray]:
    return [value.detach().cpu().numpy() for value in model.state_dict().values()]


def set_parameters(model: nn.Module, parameters: List[np.ndarray]) -> None:
    keys = model.state_dict().keys()
    state = OrderedDict(
        (key, torch.tensor(value, device=DEVICE))
        for key, value in zip(keys, parameters)
    )
    model.load_state_dict(state, strict=True)


def flatten_update(
    before: List[np.ndarray], after: List[np.ndarray]
) -> np.ndarray:
    return np.concatenate(
        [
            (new.astype(np.float64) - old.astype(np.float64)).reshape(-1)
            for old, new in zip(before, after)
        ]
    ).astype(np.float32)


class ControlledT33Client(fl.client.NumPyClient):
    """
    T3.3 client using a pre-generated normalized partition.

    Every fit writes its model-update vector to a unique per-round file. The
    update is the locally trained parameter vector minus the received global
    parameter vector, which is an empirical proxy for the local optimization
    direction.
    """

    def __init__(
        self,
        partition_path: Path,
        update_dir: Path,
        learning_rate: float,
        batch_size: int,
        base_seed: int,
    ):
        values = np.load(partition_path, allow_pickle=False)
        self.x_train = values["x_train"].astype(np.float32)
        self.y_train = values["y_train"].astype(np.float32)
        self.partition_id = int(values["partition_id"])
        self.mode = str(values["mode"].item())
        self.group_code = int(values["slot_profile_code"])
        self.source_client_id = int(values["source_client_id"])
        self.num_calm_examples = int(values["num_calm_examples"])
        self.num_aggressive_examples = int(values["num_aggressive_examples"])

        if len(self.x_train) == 0:
            raise ValueError(f"Empty partition: {partition_path}")

        self.update_dir = update_dir
        self.learning_rate = float(learning_rate)
        self.batch_size = int(batch_size)
        self.base_seed = int(base_seed)
        self.model = PowerLSTMForecaster().to(DEVICE)

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        server_round = int(config.get("server_round", 0))
        local_epochs = int(config.get("local_epochs", 1))
        round_seed = self.base_seed + server_round * 100_003 + self.partition_id

        torch.manual_seed(round_seed)
        np.random.seed(round_seed % (2**32 - 1))
        set_parameters(self.model, parameters)
        before = [np.array(value, copy=True) for value in parameters]

        generator = torch.Generator().manual_seed(round_seed)
        loader = DataLoader(
            TensorDataset(
                torch.from_numpy(self.x_train),
                torch.from_numpy(self.y_train),
            ),
            batch_size=self.batch_size,
            shuffle=True,
            generator=generator,
        )

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate
        )
        self.model.train()

        total_loss = 0.0
        total_seen = 0
        for _ in range(local_epochs):
            for x_batch, y_batch in loader:
                x_batch = x_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)
                optimizer.zero_grad()
                predictions = self.model(x_batch)
                loss = criterion(predictions, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item()) * len(x_batch)
                total_seen += len(x_batch)

        after = get_parameters(self.model)
        update = flatten_update(before, after)
        update_norm = float(np.linalg.norm(update.astype(np.float64)))

        round_dir = self.update_dir / f"round_{server_round:03d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        update_path = round_dir / f"partition_{self.partition_id:03d}.npz"
        np.savez_compressed(
            update_path,
            update=update,
            num_examples=np.int64(len(self.x_train)),
            partition_id=np.int64(self.partition_id),
            group_code=np.int64(self.group_code),
            source_client_id=np.int64(self.source_client_id),
            mode=np.asarray(self.mode),
            update_l2_norm=np.float64(update_norm),
        )

        mean_train_mse = total_loss / max(total_seen, 1)
        return (
            after,
            len(self.x_train),
            {
                "partition_id": self.partition_id,
                "source_client_id": self.source_client_id,
                "group_code": self.group_code,
                "train_mse_normalized": float(mean_train_mse),
                "update_l2_norm": update_norm,
                "num_calm_examples": self.num_calm_examples,
                "num_aggressive_examples": self.num_aggressive_examples,
            },
        )

    def evaluate(self, parameters, config):
        # Controlled T3.3 uses one fixed server-side evaluation set, so this
        # method is only a safe fallback and is not sampled by the strategy.
        set_parameters(self.model, parameters)
        return 0.0, len(self.x_train), {"partition_id": self.partition_id}
