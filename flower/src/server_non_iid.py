from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import flwr as fl
import pandas as pd
from flwr.common import Context, Metrics

from client import FlowerClient
from server import discover_client_ids, metric_series_to_json


FLOWER_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "dataset"
OUTPUT_DIR = FLOWER_ROOT / "outputs"

ACCELERATION_COLUMN = "acceleration"
CALM_CODE = 0
AGGRESSIVE_CODE = 1


class ProfiledFlowerClient(FlowerClient):
    """FlowerClient that reports its non-IID profile with every metric."""

    def __init__(self, client_id: int, profile_code: int):
        super().__init__(client_id)
        self.profile_code = int(profile_code)

    def fit(self, parameters, config):
        updated_parameters, num_examples, metrics = super().fit(parameters, config)
        metrics = dict(metrics)
        metrics["profile_code"] = self.profile_code
        return updated_parameters, num_examples, metrics

    def evaluate(self, parameters, config):
        loss, num_examples, metrics = super().evaluate(parameters, config)
        metrics = dict(metrics)
        metrics["profile_code"] = self.profile_code
        return loss, num_examples, metrics


def build_non_iid_profiles(
    valid_client_ids: List[int],
    tail_fraction: float,
) -> Tuple[pd.DataFrame, List[int], List[int]]:
    """
    Rank valid clients by signed mean acceleration.

    The lowest tail becomes Calm, the highest tail becomes Aggressive,
    and middle clients are excluded to strengthen non-IID heterogeneity.
    """

    if not 0.0 < tail_fraction <= 0.5:
        raise ValueError("tail_fraction must be greater than 0 and at most 0.5.")

    records = []

    for client_id in valid_client_ids:
        csv_path = DATASET_DIR / f"client_{client_id}.csv"
        df = pd.read_csv(csv_path, usecols=[ACCELERATION_COLUMN])

        acceleration = pd.to_numeric(
            df[ACCELERATION_COLUMN], errors="coerce"
        ).dropna()

        if acceleration.empty:
            continue

        records.append(
            {
                "client_id": int(client_id),
                "num_acceleration_rows": int(len(acceleration)),
                "mean_acceleration": float(acceleration.mean()),
                "mean_abs_acceleration": float(acceleration.abs().mean()),
                "acceleration_std": float(acceleration.std(ddof=0)),
            }
        )

    profiles = pd.DataFrame(records)

    if profiles.empty:
        raise ValueError("No valid acceleration values were found.")

    profiles = profiles.sort_values(
        by=["mean_acceleration", "client_id"],
        ascending=[True, True],
    ).reset_index(drop=True)

    tail_count = max(1, int(len(profiles) * tail_fraction))

    if 2 * tail_count > len(profiles):
        raise ValueError("The selected profile tails overlap.")

    profiles["profile"] = "Middle"
    profiles["profile_code"] = -1

    calm_index = profiles.index[:tail_count]
    aggressive_index = profiles.index[-tail_count:]

    profiles.loc[calm_index, "profile"] = "Calm"
    profiles.loc[calm_index, "profile_code"] = CALM_CODE

    profiles.loc[aggressive_index, "profile"] = "Aggressive"
    profiles.loc[aggressive_index, "profile_code"] = AGGRESSIVE_CODE

    calm_ids = (
        profiles.loc[profiles["profile"] == "Calm", "client_id"]
        .astype(int)
        .tolist()
    )
    aggressive_ids = (
        profiles.loc[profiles["profile"] == "Aggressive", "client_id"]
        .astype(int)
        .tolist()
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    profiles_path = OUTPUT_DIR / "t3_3_client_acceleration_profiles.csv"
    profiles.to_csv(profiles_path, index=False)

    print("============================================================")
    print("T3.3 non-IID acceleration profiles")
    print("============================================================")
    print(f"Valid clients analysed:       {len(profiles)}")
    print(f"Tail fraction per profile:    {tail_fraction:.0%}")
    print(f"Calm clients:                 {len(calm_ids)}")
    print(f"Aggressive clients:           {len(aggressive_ids)}")
    print(f"Middle clients excluded:      {len(profiles) - 2 * tail_count}")
    print(
        "Calm mean acceleration:      "
        f"{profiles.loc[calm_index, 'mean_acceleration'].min():.6f} to "
        f"{profiles.loc[calm_index, 'mean_acceleration'].max():.6f}"
    )
    print(
        "Aggressive mean acceleration:"
        f" {profiles.loc[aggressive_index, 'mean_acceleration'].min():.6f} to "
        f"{profiles.loc[aggressive_index, 'mean_acceleration'].max():.6f}"
    )
    print(f"Profiles saved:               {profiles_path}")
    print("============================================================")

    return profiles, calm_ids, aggressive_ids


def interleave_profiles(calm_ids: List[int], aggressive_ids: List[int]) -> List[int]:
    """Create a deterministic virtual-client list alternating both profiles."""

    if len(calm_ids) != len(aggressive_ids):
        raise ValueError("Calm and Aggressive groups must contain equal counts.")

    selected_client_ids: List[int] = []

    for calm_id, aggressive_id in zip(calm_ids, aggressive_ids):
        selected_client_ids.extend([calm_id, aggressive_id])

    return selected_client_ids


def weighted_mse(
    metrics: List[Tuple[int, Metrics]],
    mse_key: str,
    profile_code: int | None = None,
) -> float | None:
    selected = []

    for num_examples, client_metrics in metrics:
        if mse_key not in client_metrics:
            continue

        if profile_code is not None:
            current_code = int(client_metrics.get("profile_code", -1))
            if current_code != profile_code:
                continue

        selected.append((num_examples, client_metrics))

    total_examples = sum(num_examples for num_examples, _ in selected)

    if total_examples == 0:
        return None

    return sum(
        num_examples * float(client_metrics[mse_key])
        for num_examples, client_metrics in selected
    ) / total_examples


def aggregate_profile_metrics(
    metrics: List[Tuple[int, Metrics]],
    mse_key: str,
    metric_prefix: str,
) -> Metrics:
    """Aggregate global and per-profile MSE/RMSE values."""

    result: Metrics = {}

    global_mse = weighted_mse(metrics, mse_key)
    calm_mse = weighted_mse(metrics, mse_key, CALM_CODE)
    aggressive_mse = weighted_mse(metrics, mse_key, AGGRESSIVE_CODE)

    if global_mse is not None:
        result[mse_key] = float(global_mse)
        result[f"{metric_prefix}_rmse"] = float(math.sqrt(global_mse))

    if calm_mse is not None:
        result[f"{metric_prefix}_loss_calm"] = float(calm_mse)
        result[f"{metric_prefix}_rmse_calm"] = float(math.sqrt(calm_mse))

    if aggressive_mse is not None:
        result[f"{metric_prefix}_loss_aggressive"] = float(aggressive_mse)
        result[f"{metric_prefix}_rmse_aggressive"] = float(
            math.sqrt(aggressive_mse)
        )

    if calm_mse is not None and aggressive_mse is not None:
        calm_rmse = math.sqrt(calm_mse)
        aggressive_rmse = math.sqrt(aggressive_mse)
        result[f"{metric_prefix}_rmse_profile_gap"] = float(
            abs(calm_rmse - aggressive_rmse)
        )

    return result


def aggregate_fit_metrics(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    return aggregate_profile_metrics(
        metrics=metrics,
        mse_key="train_loss",
        metric_prefix="train",
    )


def aggregate_evaluate_metrics(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    return aggregate_profile_metrics(
        metrics=metrics,
        mse_key="val_loss",
        metric_prefix="val",
    )


def save_non_iid_history(
    history,
    profiles: pd.DataFrame,
    calm_ids: List[int],
    aggressive_ids: List[int],
    args: argparse.Namespace,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    num_clients = len(calm_ids) + len(aggressive_ids)
    base_name = (
        f"t3_3_non_iid_extremes_{num_clients}_clients_"
        f"{args.num_rounds}_rounds"
    )

    pickle_path = OUTPUT_DIR / f"{base_name}.pkl"
    json_path = OUTPUT_DIR / f"{base_name}.json"

    with pickle_path.open("wb") as file:
        pickle.dump(history, file)

    calm_rows = profiles[profiles["profile"] == "Calm"]
    aggressive_rows = profiles[profiles["profile"] == "Aggressive"]

    history_data = {
        "experiment": {
            "task": "T3.3 Non-IID Analysis",
            "profile_statistic": "signed mean acceleration",
            "tail_fraction": float(args.tail_fraction),
            "num_profiled_clients": int(len(profiles)),
            "num_non_iid_clients": int(num_clients),
            "num_calm_clients": int(len(calm_ids)),
            "num_aggressive_clients": int(len(aggressive_ids)),
            "num_rounds": int(args.num_rounds),
            "fraction_fit": float(args.fraction_fit),
            "fraction_evaluate": float(args.fraction_evaluate),
            "local_epochs": int(args.local_epochs),
            "calm_mean_acceleration_range": [
                float(calm_rows["mean_acceleration"].min()),
                float(calm_rows["mean_acceleration"].max()),
            ],
            "aggressive_mean_acceleration_range": [
                float(aggressive_rows["mean_acceleration"].min()),
                float(aggressive_rows["mean_acceleration"].max()),
            ],
            "calm_client_ids": [int(value) for value in calm_ids],
            "aggressive_client_ids": [int(value) for value in aggressive_ids],
        },
        "losses_distributed": [
            [int(server_round), float(loss)]
            for server_round, loss in history.losses_distributed
        ],
        "losses_centralized": [
            [int(server_round), float(loss)]
            for server_round, loss in history.losses_centralized
        ],
        "metrics_distributed": metric_series_to_json(
            history.metrics_distributed
        ),
        "metrics_centralized": metric_series_to_json(
            history.metrics_centralized
        ),
        "metrics_distributed_fit": metric_series_to_json(
            history.metrics_distributed_fit
        ),
    }

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(history_data, file, indent=2)

    print()
    print("T3.3 history saved:")
    print(f"  Pickle: {pickle_path}")
    print(f"  JSON:   {json_path}")


def run_non_iid_simulation(args: argparse.Namespace):
    valid_client_ids = discover_client_ids()

    profiles, calm_ids, aggressive_ids = build_non_iid_profiles(
        valid_client_ids=valid_client_ids,
        tail_fraction=args.tail_fraction,
    )

    selected_client_ids = interleave_profiles(calm_ids, aggressive_ids)
    num_clients = len(selected_client_ids)

    profile_code_by_id: Dict[int, int] = {
        client_id: CALM_CODE for client_id in calm_ids
    }
    profile_code_by_id.update(
        {client_id: AGGRESSIVE_CODE for client_id in aggressive_ids}
    )

    if args.num_rounds <= 0:
        raise ValueError("num_rounds must be greater than zero.")

    if not 0.0 < args.fraction_fit <= 1.0:
        raise ValueError("fraction_fit must be greater than 0 and at most 1.")

    if not 0.0 <= args.fraction_evaluate <= 1.0:
        raise ValueError("fraction_evaluate must be between 0 and 1.")

    if args.min_fit_clients > num_clients:
        raise ValueError("min_fit_clients cannot exceed the non-IID population.")

    if args.min_evaluate_clients > num_clients:
        raise ValueError(
            "min_evaluate_clients cannot exceed the non-IID population."
        )

    def fit_config(server_round: int) -> dict:
        return {
            "local_epochs": int(args.local_epochs),
            "server_round": int(server_round),
        }

    print("============================================================")
    print("T3.3 Flower FedAvg non-IID simulation")
    print("============================================================")
    print(f"Non-IID clients:          {num_clients}")
    print(f"Calm clients:             {len(calm_ids)}")
    print(f"Aggressive clients:       {len(aggressive_ids)}")
    print(f"Federated rounds:         {args.num_rounds}")
    print(f"Training participation:   {args.fraction_fit:.0%}")
    print(f"Evaluation participation: {args.fraction_evaluate:.0%}")
    print(f"Local epochs:             {args.local_epochs}")
    print(f"CPU per client:           {args.num_cpus}")
    print(f"GPU per client:           {args.num_gpus}")
    print("============================================================")

    def client_fn(context: Context):
        virtual_id = int(context.node_config["partition-id"])
        real_client_id = selected_client_ids[virtual_id]
        profile_code = profile_code_by_id[real_client_id]
        profile_name = "Calm" if profile_code == CALM_CODE else "Aggressive"

        print(
            f"[T3.3] Virtual client {virtual_id} mapped to "
            f"dataset client {real_client_id} ({profile_name})"
        )

        client = ProfiledFlowerClient(
            client_id=real_client_id,
            profile_code=profile_code,
        )
        return client.to_client()

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=args.fraction_fit,
        fraction_evaluate=args.fraction_evaluate,
        min_fit_clients=args.min_fit_clients,
        min_evaluate_clients=args.min_evaluate_clients,
        min_available_clients=num_clients,
        on_fit_config_fn=fit_config,
        fit_metrics_aggregation_fn=aggregate_fit_metrics,
        evaluate_metrics_aggregation_fn=aggregate_evaluate_metrics,
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

    save_non_iid_history(
        history=history,
        profiles=profiles,
        calm_ids=calm_ids,
        aggressive_ids=aggressive_ids,
        args=args,
    )

    return history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run T3.3 using the lowest and highest mean-acceleration "
            "client tails as Calm and Aggressive profiles."
        )
    )

    parser.add_argument(
        "--tail-fraction",
        type=float,
        default=0.25,
        help="Fraction assigned to each extreme profile.",
    )
    parser.add_argument("--num-rounds", type=int, default=20)
    parser.add_argument(
        "--fraction-fit",
        type=float,
        default=0.2,
        help=(
            "0.2 selects about 16 clients from an 82-client extreme population, "
            "matching the T3.2 client count per round."
        ),
    )
    parser.add_argument(
        "--fraction-evaluate",
        type=float,
        default=1.0,
        help=(
            "Evaluate all non-IID clients each round to remove evaluation "
            "sampling noise from the convergence curve."
        ),
    )
    parser.add_argument("--min-fit-clients", type=int, default=2)
    parser.add_argument("--min-evaluate-clients", type=int, default=2)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--num-cpus", type=float, default=1.0)
    parser.add_argument("--num-gpus", type=float, default=0.0)

    return parser.parse_args()


if __name__ == "__main__":
    run_non_iid_simulation(parse_args())
