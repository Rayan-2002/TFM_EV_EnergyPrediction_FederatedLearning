from __future__ import annotations

import argparse
import json
import math
import pickle
import random
import shutil
from pathlib import Path
from typing import List, Tuple

import flwr as fl
import numpy as np
import torch
from flwr.common import Context, Metrics, ndarrays_to_parameters
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from client_t3_3_controlled import (
    ControlledT33Client,
    PowerLSTMForecaster,
    get_parameters,
    set_parameters,
)


FLOWER_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREPARED_DIR = FLOWER_ROOT / "data" / "t3_3_controlled"
OUTPUT_DIR = FLOWER_ROOT / "outputs"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def make_loader(path: Path, batch_size: int) -> DataLoader:
    values = np.load(path, allow_pickle=False)
    x = torch.from_numpy(values["x"].astype(np.float32))
    y = torch.from_numpy(values["y"].astype(np.float32))
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)


def evaluate_watts(
    model: nn.Module,
    loader: DataLoader,
    target_std: float,
    device: torch.device,
) -> Tuple[float, int]:
    model.eval()
    model.to(device)
    squared_error_sum = 0.0
    count = 0

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            prediction = model(x_batch)
            # Both prediction and target are normalized with the same scaler.
            error_watts = (prediction - y_batch) * target_std
            squared_error_sum += float((error_watts**2).sum().item())
            count += int(y_batch.numel())

    if count == 0:
        raise ValueError("The fixed evaluation loader is empty.")
    return squared_error_sum / count, count


def aggregate_fit_metrics(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    if not metrics:
        return {}

    total_examples = sum(num_examples for num_examples, _ in metrics)
    if total_examples <= 0:
        return {}

    def weighted(key: str, group_code: int | None = None) -> float | None:
        selected = []
        for num_examples, values in metrics:
            if key not in values:
                continue
            if group_code is not None and int(values.get("group_code", -1)) != group_code:
                continue
            selected.append((num_examples, float(values[key])))
        denominator = sum(num_examples for num_examples, _ in selected)
        if denominator == 0:
            return None
        return sum(n * value for n, value in selected) / denominator

    result: Metrics = {}
    for key in ["train_mse_normalized", "update_l2_norm"]:
        global_value = weighted(key)
        group_0 = weighted(key, 0)
        group_1 = weighted(key, 1)
        if global_value is not None:
            result[key] = float(global_value)
        if group_0 is not None:
            result[f"{key}_group_0"] = float(group_0)
        if group_1 is not None:
            result[f"{key}_group_1"] = float(group_1)
    return result


def metric_series_to_json(series: dict) -> dict:
    return {
        name: [[int(round_number), float(value)] for round_number, value in values]
        for name, values in series.items()
    }


def run(args: argparse.Namespace):
    manifest = load_json(args.prepared_dir / "manifest.json")
    scaler = load_json(args.prepared_dir / "scaler.json")
    num_clients = int(manifest["num_partitions"])
    mode_dir = args.prepared_dir / args.mode

    for partition_id in range(num_clients):
        path = mode_dir / f"partition_{partition_id:03d}.npz"
        if not path.exists():
            raise FileNotFoundError(path)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluation_dir = args.prepared_dir / "evaluation"
    global_loader = make_loader(evaluation_dir / "global.npz", args.batch_size)
    calm_loader = make_loader(evaluation_dir / "calm.npz", args.batch_size)
    aggressive_loader = make_loader(
        evaluation_dir / "aggressive.npz", args.batch_size
    )

    initial_model = PowerLSTMForecaster().to(device)
    initial_ndarrays = get_parameters(initial_model)
    initial_parameters = ndarrays_to_parameters(initial_ndarrays)

    update_dir = OUTPUT_DIR / "t3_3_controlled_updates" / args.mode
    if update_dir.exists() and not args.keep_existing_updates:
        shutil.rmtree(update_dir)
    update_dir.mkdir(parents=True, exist_ok=True)

    final_model_path = OUTPUT_DIR / f"t3_3_controlled_{args.mode}_global_model.pt"
    target_std = float(scaler["target_std"])

    evaluation_records = []

    def evaluate_fn(server_round: int, parameters, config):
        model = PowerLSTMForecaster().to(device)
        set_parameters(model, parameters)

        global_mse, global_count = evaluate_watts(
            model, global_loader, target_std, device
        )
        calm_mse, calm_count = evaluate_watts(
            model, calm_loader, target_std, device
        )
        aggressive_mse, aggressive_count = evaluate_watts(
            model, aggressive_loader, target_std, device
        )

        global_rmse = math.sqrt(global_mse)
        calm_rmse = math.sqrt(calm_mse)
        aggressive_rmse = math.sqrt(aggressive_mse)
        metrics = {
            "global_rmse_w": float(global_rmse),
            "calm_rmse_w": float(calm_rmse),
            "aggressive_rmse_w": float(aggressive_rmse),
            "profile_gap_w": float(abs(calm_rmse - aggressive_rmse)),
            "global_examples": int(global_count),
            "calm_examples": int(calm_count),
            "aggressive_examples": int(aggressive_count),
        }
        evaluation_records.append(
            {"round": int(server_round), "loss": float(global_mse), **metrics}
        )

        if server_round == args.num_rounds:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "mode": args.mode,
                    "server_round": int(server_round),
                    "scaler": scaler,
                    "manifest": manifest,
                    "seed": args.seed,
                },
                final_model_path,
            )

        return float(global_mse), metrics

    def fit_config(server_round: int) -> dict:
        return {
            "server_round": int(server_round),
            "local_epochs": int(args.local_epochs),
            "seed": int(args.seed),
        }

    def client_fn(context: Context):
        partition_id = int(context.node_config["partition-id"])
        partition_path = mode_dir / f"partition_{partition_id:03d}.npz"
        client = ControlledT33Client(
            partition_path=partition_path,
            update_dir=update_dir,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            base_seed=args.seed,
        )
        return client.to_client()

    if args.fraction_fit == 1.0:
        min_fit_clients = num_clients
    else:
        min_fit_clients = max(2, int(num_clients * args.fraction_fit))

    print("============================================================")
    print(f"T3.3 controlled {args.mode.upper()} simulation")
    print("============================================================")
    print(f"Clients:                    {num_clients}")
    print(f"Rounds:                     {args.num_rounds}")
    print(f"Training participation:     {args.fraction_fit:.0%}")
    print(f"Local epochs:               {args.local_epochs}")
    print(f"Learning rate:              {args.learning_rate}")
    print("Evaluation:                 fixed server-side holdout")
    print("Initialization:             deterministic and shared")
    print(f"Update vectors:             {update_dir}")
    print("============================================================")

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=args.fraction_fit,
        fraction_evaluate=0.0,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=1,
        min_available_clients=num_clients,
        initial_parameters=initial_parameters,
        on_fit_config_fn=fit_config,
        evaluate_fn=evaluate_fn,
        fit_metrics_aggregation_fn=aggregate_fit_metrics,
    )

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
        client_resources={
            "num_cpus": args.num_cpus,
            "num_gpus": args.num_gpus,
        },
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    base_name = (
        f"t3_3_controlled_{args.mode}_{num_clients}_clients_"
        f"{args.num_rounds}_rounds"
    )
    pickle_path = OUTPUT_DIR / f"{base_name}.pkl"
    json_path = OUTPUT_DIR / f"{base_name}.json"

    with pickle_path.open("wb") as file:
        pickle.dump(history, file)

    history_data = {
        "experiment": {
            "task": "T3.3 controlled IID versus non-IID",
            "mode": args.mode,
            "num_clients": num_clients,
            "num_rounds": args.num_rounds,
            "fraction_fit": args.fraction_fit,
            "local_epochs": args.local_epochs,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "normalization": manifest["normalization"],
            "fixed_evaluation": True,
            "same_initialization_seed": args.seed,
            "prepared_manifest": str(args.prepared_dir / "manifest.json"),
        },
        "losses_centralized": [
            [int(round_number), float(loss)]
            for round_number, loss in history.losses_centralized
        ],
        "metrics_centralized": metric_series_to_json(
            history.metrics_centralized
        ),
        "metrics_distributed_fit": metric_series_to_json(
            history.metrics_distributed_fit
        ),
        "evaluation_records": evaluation_records,
        "update_directory": str(update_dir),
        "final_model": str(final_model_path),
    }
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(history_data, file, indent=2)

    print()
    print("Controlled history saved:")
    print(f"  Pickle: {pickle_path}")
    print(f"  JSON:   {json_path}")
    print(f"  Model:  {final_model_path}")
    print(f"  Updates:{update_dir}")
    return history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one normalized controlled T3.3 Flower experiment."
    )
    parser.add_argument("--mode", choices=["iid", "non_iid"], required=True)
    parser.add_argument("--prepared-dir", type=Path, default=DEFAULT_PREPARED_DIR)
    parser.add_argument("--num-rounds", type=int, default=20)
    parser.add_argument(
        "--fraction-fit",
        type=float,
        default=1.0,
        help="Use 1.0 for a fully controlled comparison with identical participants.",
    )
    parser.add_argument("--local-epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num-cpus", type=float, default=1.0)
    parser.add_argument("--num-gpus", type=float, default=0.0)
    parser.add_argument("--keep-existing-updates", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
