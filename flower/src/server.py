from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path
from typing import List, Tuple

import flwr as fl
from flwr.common import Context, Metrics

# This works when the server is launched with:
# python src/server.py
import pandas as pd

from client import (
    FlowerClient,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    T_PAST,
    T_FUTURE,
)


# ============================================================
# Project paths
# ============================================================

# server.py is located at:
# SUMO_Barcelona/flower/src/server.py

FLOWER_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = PROJECT_ROOT / "dataset"
OUTPUT_DIR = FLOWER_ROOT / "outputs"


# ============================================================
# Discover the client CSV files
# ============================================================

def discover_client_ids() -> List[int]:
    """
    Discover client CSV files that contain enough valid rows
    to produce both training and validation temporal windows.
    """

    valid_client_ids: List[int] = []
    skipped_clients = []

    required_columns = list(
        dict.fromkeys(FEATURE_COLUMNS + [TARGET_COLUMN])
    )

    # Two windows are required so that both training and
    # validation sets contain at least one example.
    minimum_windows = 2

    for csv_path in DATASET_DIR.glob("client_*.csv"):
        suffix = csv_path.stem.removeprefix("client_")

        if not suffix.isdigit():
            continue

        client_id = int(suffix)

        try:
            df = pd.read_csv(
                csv_path,
                usecols=required_columns,
            )

            usable_rows = len(
                df.dropna(subset=required_columns)
            )

            num_windows = (
                usable_rows
                - T_PAST
                - T_FUTURE
                + 1
            )

            if num_windows < minimum_windows:
                skipped_clients.append(
                    (
                        client_id,
                        usable_rows,
                        max(num_windows, 0),
                    )
                )
                continue

            valid_client_ids.append(client_id)

        except (ValueError, pd.errors.EmptyDataError) as error:
            skipped_clients.append(
                (client_id, 0, f"invalid CSV: {error}")
            )

    valid_client_ids.sort()
    skipped_clients.sort(key=lambda item: item[0])

    print(
        f"Usable client datasets: {len(valid_client_ids)}"
    )
    print(
        f"Skipped client datasets: {len(skipped_clients)}"
    )

    for client_id, usable_rows, reason in skipped_clients:
        print(
            f"  Skipping client {client_id}: "
            f"{usable_rows} usable rows, "
            f"{reason} temporal windows"
        )

    if not valid_client_ids:
        raise FileNotFoundError(
            "No client CSV contains enough data to create "
            "training and validation temporal windows."
        )

    return valid_client_ids


# ============================================================
# Configuration sent to clients
# ============================================================

def fit_config(server_round: int) -> dict:
    """
    Send configuration values to FlowerClient.fit().

    We currently train each selected client for one local epoch.
    """

    return {
        "local_epochs": 1,
        "server_round": server_round,
    }


# ============================================================
# Metric aggregation
# ============================================================

def aggregate_mse_metrics(
    metrics: List[Tuple[int, Metrics]],
    mse_key: str,
    rmse_key: str,
) -> Metrics:
    """
    Aggregate client MSE values using the number of examples
    on each client as weights.

    Clients with more examples have more influence.

    The global RMSE is calculated from the aggregated MSE.
    """

    valid_metrics = [
        (num_examples, client_metrics)
        for num_examples, client_metrics in metrics
        if mse_key in client_metrics
    ]

    if not valid_metrics:
        return {}

    total_examples = sum(
        num_examples
        for num_examples, _ in valid_metrics
    )

    if total_examples == 0:
        return {}

    weighted_mse = sum(
        num_examples * float(client_metrics[mse_key])
        for num_examples, client_metrics in valid_metrics
    ) / total_examples

    weighted_rmse = math.sqrt(weighted_mse)

    return {
        mse_key: float(weighted_mse),
        rmse_key: float(weighted_rmse),
    }


def aggregate_fit_metrics(
    metrics: List[Tuple[int, Metrics]],
) -> Metrics:
    """
    Aggregate the training metrics returned by FlowerClient.fit().
    """

    return aggregate_mse_metrics(
        metrics=metrics,
        mse_key="train_loss",
        rmse_key="train_rmse",
    )


def aggregate_evaluate_metrics(
    metrics: List[Tuple[int, Metrics]],
) -> Metrics:
    """
    Aggregate the validation metrics returned by
    FlowerClient.evaluate().
    """

    return aggregate_mse_metrics(
        metrics=metrics,
        mse_key="val_loss",
        rmse_key="val_rmse",
    )


# ============================================================
# Save Flower History
# ============================================================

def metric_series_to_json(metric_series: dict) -> dict:
    """
    Convert Flower History metrics into JSON-compatible values.
    """

    return {
        metric_name: [
            [int(server_round), float(value)]
            for server_round, value in values
        ]
        for metric_name, values in metric_series.items()
    }


def save_history(
    history,
    num_clients: int,
    num_rounds: int,
) -> None:
    """
    Save the History returned by start_simulation().

    The Pickle file preserves the complete Flower History object.

    The JSON file provides readable metrics for later analysis
    and plotting in T3.4.
    """

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    base_name = (
        f"t3_2_history_{num_clients}_clients_"
        f"{num_rounds}_rounds"
    )

    pickle_path = OUTPUT_DIR / f"{base_name}.pkl"
    json_path = OUTPUT_DIR / f"{base_name}.json"

    # Save the complete History object
    with pickle_path.open("wb") as file:
        pickle.dump(history, file)

    # Save the important values in readable JSON format
    history_data = {
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
    print("History saved:")
    print(f"  Pickle: {pickle_path}")
    print(f"  JSON:   {json_path}")


# ============================================================
# Federated simulation
# ============================================================

def run_federated_simulation(args: argparse.Namespace):
    """
    Create the FedAvg strategy and run the Flower simulation.
    """

    all_client_ids = discover_client_ids()

    # --------------------------------------------------------
    # Validate command-line values
    # --------------------------------------------------------

    if args.num_clients <= 0:
        raise ValueError(
            "num_clients must be greater than zero."
        )

    if args.num_clients > len(all_client_ids):
        raise ValueError(
            f"Requested {args.num_clients} clients, but only "
            f"{len(all_client_ids)} client CSV files were found."
        )

    if args.num_rounds <= 0:
        raise ValueError(
            "num_rounds must be greater than zero."
        )

    if not 0.0 < args.fraction_fit <= 1.0:
        raise ValueError(
            "fraction_fit must be greater than 0 and at most 1."
        )

    if not 0.0 <= args.fraction_evaluate <= 1.0:
        raise ValueError(
            "fraction_evaluate must be between 0 and 1."
        )

    if args.min_fit_clients > args.num_clients:
        raise ValueError(
            "min_fit_clients cannot exceed num_clients."
        )

    if args.min_evaluate_clients > args.num_clients:
        raise ValueError(
            "min_evaluate_clients cannot exceed num_clients."
        )

    # Example with 10 clients:
    # selected_client_ids = [0, 1, 2, ..., 9]
    selected_client_ids = all_client_ids[:args.num_clients]

    # --------------------------------------------------------
    # Display simulation configuration
    # --------------------------------------------------------

    print("============================================================")
    print("T3.2 Flower FedAvg simulation")
    print("============================================================")
    print(f"Available client CSVs:    {len(all_client_ids)}")
    print(f"Clients in simulation:    {args.num_clients}")
    print(f"Federated rounds:         {args.num_rounds}")
    print(f"Training participation:   {args.fraction_fit:.0%}")
    print(f"Evaluation participation: {args.fraction_evaluate:.0%}")
    print(f"Minimum fit clients:      {args.min_fit_clients}")
    print(f"Minimum eval clients:     {args.min_evaluate_clients}")
    print(f"Minimum available:        {args.num_clients}")
    print("Local epochs:             1")
    print(f"CPU per client:           {args.num_cpus}")
    print(f"GPU per client:           {args.num_gpus}")
    print("============================================================")

    # --------------------------------------------------------
    # Flower client factory
    # --------------------------------------------------------

    def client_fn(context: Context):
        """
        Flower calls this function whenever it needs a client.

        context.node_config["partition-id"] contains the virtual
        client ID assigned by the simulation.
        """

        virtual_id = int(
            context.node_config["partition-id"]
        )

        real_client_id = selected_client_ids[virtual_id]

        print(
            f"[Server] Virtual client {virtual_id} "
            f"mapped to dataset client {real_client_id}"
        )

        client = FlowerClient(real_client_id)

        # Convert our NumPyClient into the Client type expected
        # by the Flower simulation engine.
        return client.to_client()

    # --------------------------------------------------------
    # FedAvg strategy
    # --------------------------------------------------------

    strategy = fl.server.strategy.FedAvg(
        # Percentage of clients selected for local training
        fraction_fit=args.fraction_fit,

        # Percentage of clients selected for validation
        fraction_evaluate=args.fraction_evaluate,

        # Minimum number of clients selected for training
        min_fit_clients=args.min_fit_clients,

        # Minimum number of clients selected for evaluation
        min_evaluate_clients=args.min_evaluate_clients,

        # Wait until all simulation clients are available
        min_available_clients=args.num_clients,

        # Send local_epochs to each selected client
        on_fit_config_fn=fit_config,

        # Aggregate training metrics
        fit_metrics_aggregation_fn=aggregate_fit_metrics,

        # Aggregate validation metrics
        evaluate_metrics_aggregation_fn=(
            aggregate_evaluate_metrics
        ),
    )

    # --------------------------------------------------------
    # Run simulation
    # --------------------------------------------------------

    history = fl.simulation.start_simulation(
        client_fn=client_fn,

        num_clients=args.num_clients,

        config=fl.server.ServerConfig(
            num_rounds=args.num_rounds,
        ),

        strategy=strategy,

        client_resources={
            "num_cpus": args.num_cpus,
            "num_gpus": args.num_gpus,
        },
    )

    # Save results for T3.4
    save_history(
        history=history,
        num_clients=args.num_clients,
        num_rounds=args.num_rounds,
    )

    return history


# ============================================================
# Command-line arguments
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the T3.2 Flower FedAvg simulation."
    )

    parser.add_argument(
        "--num-clients",
        type=int,
        default=169,
        help="Number of virtual Flower clients.",
    )

    parser.add_argument(
        "--num-rounds",
        type=int,
        default=20,
        help="Number of federated training rounds.",
    )

    parser.add_argument(
        "--fraction-fit",
        type=float,
        default=0.1,
        help=(
            "Fraction of available clients selected "
            "for local training."
        ),
    )

    parser.add_argument(
        "--fraction-evaluate",
        type=float,
        default=0.1,
        help=(
            "Fraction of available clients selected "
            "for local validation."
        ),
    )

    parser.add_argument(
        "--min-fit-clients",
        type=int,
        default=2,
        help="Minimum number of training clients per round.",
    )

    parser.add_argument(
        "--min-evaluate-clients",
        type=int,
        default=2,
        help="Minimum number of validation clients per round.",
    )

    parser.add_argument(
        "--num-cpus",
        type=float,
        default=1.0,
        help="CPU resources assigned to each virtual client.",
    )

    parser.add_argument(
        "--num-gpus",
        type=float,
        default=0.0,
        help="GPU resources assigned to each virtual client.",
    )

    return parser.parse_args()


# ============================================================
# Program entry point
# ============================================================

if __name__ == "__main__":
    simulation_args = parse_args()
    run_federated_simulation(simulation_args)