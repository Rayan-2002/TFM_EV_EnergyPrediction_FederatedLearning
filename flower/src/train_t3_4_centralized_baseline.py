from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


FLOWER_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREPARED_DIR = FLOWER_ROOT / "data" / "t3_3_controlled"
DEFAULT_OUTPUT_DIR = FLOWER_ROOT / "outputs"


class PowerLSTMForecaster(nn.Module):
    """Architecture used by the controlled T3.3 federated experiment."""

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


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(requested)


def load_pooled_training_data(
    prepared_dir: Path,
    source_mode: str,
    manifest: dict,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """
    Pool every controlled local partition exactly once.

    The IID and non-IID directories contain the same normalized raw training
    windows. The non-IID directory is the default because each file preserves
    one original route partition, which makes the no-duplication audit simple.
    """

    mode_dir = prepared_dir / source_mode
    num_partitions = int(manifest["num_partitions"])
    x_parts = []
    y_parts = []
    seen_partition_ids = set()

    for partition_id in range(num_partitions):
        path = mode_dir / f"partition_{partition_id:03d}.npz"
        if not path.exists():
            raise FileNotFoundError(path)

        with np.load(path, allow_pickle=False) as values:
            x = values["x_train"].astype(np.float32)
            y = values["y_train"].astype(np.float32)
            stored_partition_id = int(values["partition_id"])

        if stored_partition_id != partition_id:
            raise ValueError(
                f"Partition ID mismatch in {path}: expected {partition_id}, "
                f"found {stored_partition_id}."
            )
        if stored_partition_id in seen_partition_ids:
            raise ValueError(f"Duplicate partition ID: {stored_partition_id}")
        if len(x) == 0 or len(y) == 0:
            raise ValueError(f"Empty training partition: {path}")
        if len(x) != len(y):
            raise ValueError(f"X/Y length mismatch in {path}")

        seen_partition_ids.add(stored_partition_id)
        x_parts.append(x)
        y_parts.append(y)

    x_train = np.concatenate(x_parts, axis=0)
    y_train = np.concatenate(y_parts, axis=0)

    expected_key = f"total_training_examples_{source_mode}"
    expected_examples = int(manifest[expected_key])
    if len(x_train) != expected_examples:
        raise ValueError(
            f"Training example count mismatch: expected {expected_examples}, "
            f"found {len(x_train)}."
        )

    expected_t_past = int(manifest["t_past"])
    expected_features = len(manifest["feature_columns"])
    if x_train.ndim != 3:
        raise ValueError(f"Expected 3-D training X, got shape {x_train.shape}")
    if x_train.shape[1:] != (expected_t_past, expected_features):
        raise ValueError(
            "Training tensor shape is incompatible with the controlled "
            f"manifest: got {x_train.shape[1:]}, expected "
            f"({expected_t_past}, {expected_features})."
        )
    if y_train.ndim != 2 or y_train.shape[1] != 1:
        raise ValueError(f"Expected Y shape (N, 1), got {y_train.shape}")

    audit = {
        "num_partitions": num_partitions,
        "num_examples": int(len(x_train)),
        "t_past": int(x_train.shape[1]),
        "num_features": int(x_train.shape[2]),
    }
    return x_train, y_train, audit


def load_evaluation_arrays(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as values:
        x = values["x"].astype(np.float32)
        y = values["y"].astype(np.float32)
    if len(x) == 0 or len(x) != len(y):
        raise ValueError(f"Invalid evaluation set: {path}")
    return x, y


def make_loader(
    x: np.ndarray,
    y: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    generator: torch.Generator | None = None,
    num_workers: int = 0,
) -> DataLoader:
    return DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def evaluate_watts(
    model: nn.Module,
    loader: DataLoader,
    *,
    target_std: float,
    device: torch.device,
) -> Tuple[float, float, int]:
    model.eval()
    squared_error_sum_w = 0.0
    squared_error_sum_normalized = 0.0
    count = 0

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            prediction = model(x_batch)
            normalized_error = prediction - y_batch
            error_watts = normalized_error * target_std
            squared_error_sum_normalized += float(
                (normalized_error**2).sum().item()
            )
            squared_error_sum_w += float((error_watts**2).sum().item())
            count += int(y_batch.numel())

    if count == 0:
        raise ValueError("Evaluation loader is empty.")

    mse_normalized = squared_error_sum_normalized / count
    mse_watts = squared_error_sum_w / count
    return mse_watts, mse_normalized, count


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    loss_sum = 0.0
    count = 0

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        prediction = model(x_batch)
        loss = criterion(prediction, y_batch)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss: {loss.item()}")
        loss.backward()
        optimizer.step()

        loss_sum += float(loss.item()) * len(x_batch)
        count += len(x_batch)

    if count == 0:
        raise ValueError("Training loader is empty.")
    return loss_sum / count


def evaluate_all_profiles(
    model: nn.Module,
    loaders: Dict[str, DataLoader],
    *,
    target_std: float,
    device: torch.device,
) -> dict:
    output = {}
    for profile, loader in loaders.items():
        mse_w, mse_normalized, count = evaluate_watts(
            model,
            loader,
            target_std=target_std,
            device=device,
        )
        output[f"{profile}_mse_w"] = float(mse_w)
        output[f"{profile}_rmse_w"] = float(math.sqrt(mse_w))
        output[f"{profile}_mse_normalized"] = float(mse_normalized)
        output[f"{profile}_examples"] = int(count)

    output["profile_gap_w"] = float(
        abs(output["calm_rmse_w"] - output["aggressive_rmse_w"])
    )
    return output


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    manifest: dict,
    scaler: dict,
    config: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": int(epoch),
            "metrics": metrics,
            "manifest": manifest,
            "scaler": scaler,
            "config": config,
        },
        path,
    )


def run(args: argparse.Namespace) -> dict:
    manifest = load_json(args.prepared_dir / "manifest.json")
    scaler = load_json(args.prepared_dir / "scaler.json")

    seed_everything(args.seed, args.deterministic)
    device = resolve_device(args.device)

    x_train, y_train, training_audit = load_pooled_training_data(
        args.prepared_dir,
        args.source_mode,
        manifest,
    )

    evaluation_dir = args.prepared_dir / "evaluation"
    evaluation_arrays = {
        name: load_evaluation_arrays(evaluation_dir / f"{name}.npz")
        for name in ("global", "calm", "aggressive")
    }

    for profile, (x_eval, _) in evaluation_arrays.items():
        expected_key = f"{profile}_validation_examples"
        expected_count = int(manifest[expected_key])
        if len(x_eval) != expected_count:
            raise ValueError(
                f"{profile} evaluation count mismatch: expected "
                f"{expected_count}, found {len(x_eval)}."
            )

    training_generator = torch.Generator().manual_seed(args.seed)
    train_loader = make_loader(
        x_train,
        y_train,
        batch_size=args.batch_size,
        shuffle=True,
        generator=training_generator,
        num_workers=args.num_workers,
    )
    evaluation_loaders = {
        name: make_loader(
            x,
            y,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        for name, (x, y) in evaluation_arrays.items()
    }

    input_size = len(manifest["feature_columns"])
    model = PowerLSTMForecaster(
        input_size=input_size,
        hidden_size=args.hidden_size,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()
    target_std = float(scaler["target_std"])

    config = {
        "task": "T3.4 controlled centralized baseline",
        "source_mode": args.source_mode,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "hidden_size": args.hidden_size,
        "seed": args.seed,
        "device": str(device),
        "deterministic": args.deterministic,
        "model_architecture": "LSTM(input_size=4, hidden_size=32, num_layers=1) + Linear(32, 1)",
        "normalization": manifest["normalization"],
        "comparison_exposure": (
            "60 centralized epochs matches 20 federated rounds times "
            "3 local epochs when all clients participate"
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint = args.output_dir / "t3_4_controlled_centralized_best_model.pt"
    final_checkpoint = args.output_dir / "t3_4_controlled_centralized_final_model.pt"
    history_path = args.output_dir / "t3_4_controlled_centralized_history.json"

    print("=" * 68)
    print("T3.4 controlled centralized baseline")
    print("=" * 68)
    print(f"Prepared data:              {args.prepared_dir}")
    print(f"Training source:            {args.source_mode}")
    print(f"Training partitions:        {training_audit['num_partitions']}")
    print(f"Training examples:          {training_audit['num_examples']}")
    print(f"Global validation examples: {len(evaluation_arrays['global'][0])}")
    print(f"Epochs:                     {args.epochs}")
    print(f"Learning rate:              {args.learning_rate}")
    print(f"Batch size:                 {args.batch_size}")
    print(f"Device:                     {device}")
    print("=" * 68)

    records = []
    initial_metrics = evaluate_all_profiles(
        model,
        evaluation_loaders,
        target_std=target_std,
        device=device,
    )
    initial_record = {
        "epoch": 0,
        "train_mse_normalized": None,
        **initial_metrics,
    }
    records.append(initial_record)

    best_epoch = 0
    best_metrics = dict(initial_metrics)
    save_checkpoint(
        best_checkpoint,
        model=model,
        optimizer=optimizer,
        epoch=0,
        metrics=initial_metrics,
        manifest=manifest,
        scaler=scaler,
        config=config,
    )

    print(
        f"Epoch 000/{args.epochs:03d} | "
        f"global RMSE: {initial_metrics['global_rmse_w']:.3f} W | "
        f"calm: {initial_metrics['calm_rmse_w']:.3f} W | "
        f"aggressive: {initial_metrics['aggressive_rmse_w']:.3f} W"
    )

    for epoch in range(1, args.epochs + 1):
        train_mse = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
        )
        metrics = evaluate_all_profiles(
            model,
            evaluation_loaders,
            target_std=target_std,
            device=device,
        )
        record = {
            "epoch": epoch,
            "train_mse_normalized": float(train_mse),
            **metrics,
        }
        records.append(record)

        if metrics["global_rmse_w"] < best_metrics["global_rmse_w"]:
            best_epoch = epoch
            best_metrics = dict(metrics)
            save_checkpoint(
                best_checkpoint,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=metrics,
                manifest=manifest,
                scaler=scaler,
                config=config,
            )

        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} | "
            f"train MSE(norm): {train_mse:.6f} | "
            f"global RMSE: {metrics['global_rmse_w']:.3f} W | "
            f"best: {best_metrics['global_rmse_w']:.3f} W "
            f"(epoch {best_epoch})"
        )

    final_metrics = records[-1].copy()
    final_metrics.pop("epoch", None)
    final_metrics.pop("train_mse_normalized", None)
    save_checkpoint(
        final_checkpoint,
        model=model,
        optimizer=optimizer,
        epoch=args.epochs,
        metrics=final_metrics,
        manifest=manifest,
        scaler=scaler,
        config=config,
    )

    history = {
        "experiment": config,
        "fairness_checks": {
            "same_controlled_raw_training_windows_as_federated": True,
            "same_global_zscore_scaler": True,
            "same_lstm_architecture": True,
            "same_initialization_seed": True,
            "same_fixed_evaluation_sets": True,
            "same_batch_size": args.batch_size == 64,
            "same_learning_rate": math.isclose(args.learning_rate, 0.001),
            "matched_example_exposure": args.epochs == 60,
        },
        "training_data": training_audit,
        "evaluation_data": {
            "global_examples": len(evaluation_arrays["global"][0]),
            "calm_examples": len(evaluation_arrays["calm"][0]),
            "aggressive_examples": len(evaluation_arrays["aggressive"][0]),
        },
        "records": records,
        "initial": {
            "epoch": 0,
            **initial_metrics,
        },
        "best": {
            "epoch": int(best_epoch),
            **best_metrics,
        },
        "final": {
            "epoch": int(args.epochs),
            **final_metrics,
        },
        "checkpoints": {
            "best": str(best_checkpoint),
            "final": str(final_checkpoint),
        },
    }

    with history_path.open("w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)

    print()
    print("Centralized baseline complete:")
    print(f"  Best RMSE:   {best_metrics['global_rmse_w']:.3f} W")
    print(f"  Best epoch:  {best_epoch}")
    print(f"  Best model:  {best_checkpoint}")
    print(f"  Final model: {final_checkpoint}")
    print(f"  History:     {history_path}")
    return history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a controlled centralized LSTM baseline on the exact "
            "normalized windows used by the T3.3 federated experiment."
        )
    )
    parser.add_argument(
        "--prepared-dir",
        type=Path,
        default=DEFAULT_PREPARED_DIR,
    )
    parser.add_argument(
        "--source-mode",
        choices=["iid", "non_iid"],
        default="non_iid",
        help=(
            "Directory from which to pool windows. Both modes contain the "
            "same examples; non_iid preserves original route partitions."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.hidden_size != 32:
        parser.error(
            "T3.4 requires hidden_size=32 to match the controlled federated model."
        )
    return args


if __name__ == "__main__":
    run(parse_args())
