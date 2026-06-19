#!/usr/bin/env python3

import os
import json
import random
import argparse
import itertools
import time

import numpy as np
import pandas as pd
import joblib

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from feature_engineering import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    STEP_LENGTH,
    DATASET_DIR,
    OUTPUT_DIR,
    SCALER_OUTPUT_FILE,
    find_client_files,
    load_and_aggregate_csvs,
    fit_scaler,
    apply_scaler,
    SUMOPowerWindowDataset,
    inverse_scale_power
)

from model_architecture import PowerLSTMForecaster


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# JSON SAFETY
# ============================================================

def make_json_serializable(obj):
    """
    Convert NumPy/PyTorch numeric types into standard Python types
    so they can be saved with json.dump().
    """

    if isinstance(obj, dict):
        return {
            key: make_json_serializable(value)
            for key, value in obj.items()
        }

    if isinstance(obj, list):
        return [
            make_json_serializable(value)
            for value in obj
        ]

    if isinstance(obj, tuple):
        return tuple(
            make_json_serializable(value)
            for value in obj
        )

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        return float(obj)

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()

    return obj


# ============================================================
# FILE SPLIT
# ============================================================

def split_train_validation_files(csv_files, train_ratio=0.80, seed=42):
    """
    Split client files into train and validation sets.

    We split by client file, not by rows, to avoid leakage between
    windows from the same vehicle.
    """

    files = csv_files.copy()

    rng = random.Random(seed)
    rng.shuffle(files)

    n_total = len(files)

    if n_total < 2:
        raise ValueError(
            "Need at least 2 client files for hyperparameter tuning."
        )

    n_train = int(n_total * train_ratio)
    n_train = max(1, min(n_train, n_total - 1))

    train_files = files[:n_train]
    val_files = files[n_train:]

    return train_files, val_files


# ============================================================
# METRICS
# ============================================================

def compute_mse_rmse_mae(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))

    return mse, rmse, mae


# ============================================================
# TRAIN / EVALUATE
# ============================================================

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    total_samples = 0

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        y_pred = model(x_batch)

        loss = criterion(y_pred, y_batch)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        batch_size = x_batch.size(0)

        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples


def evaluate(model, loader, criterion, device, scaler):
    model.eval()

    total_loss = 0.0
    total_samples = 0

    all_targets_scaled = []
    all_predictions_scaled = []

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            y_pred = model(x_batch)

            loss = criterion(y_pred, y_batch)

            batch_size = x_batch.size(0)

            total_loss += loss.item() * batch_size
            total_samples += batch_size

            all_targets_scaled.append(y_batch.cpu().numpy())
            all_predictions_scaled.append(y_pred.cpu().numpy())

    avg_loss = total_loss / total_samples

    y_true_scaled = np.concatenate(all_targets_scaled, axis=0).reshape(-1)
    y_pred_scaled = np.concatenate(all_predictions_scaled, axis=0).reshape(-1)

    mse_scaled, rmse_scaled, mae_scaled = compute_mse_rmse_mae(
        y_true_scaled,
        y_pred_scaled
    )

    y_true_watts = inverse_scale_power(y_true_scaled, scaler)
    y_pred_watts = inverse_scale_power(y_pred_scaled, scaler)

    mse_watts, rmse_watts, mae_watts = compute_mse_rmse_mae(
        y_true_watts,
        y_pred_watts
    )

    return {
        "loss": float(avg_loss),
        "mse_scaled": float(mse_scaled),
        "rmse_scaled": float(rmse_scaled),
        "mae_scaled": float(mae_scaled),
        "mse_watts": float(mse_watts),
        "rmse_watts": float(rmse_watts),
        "mae_watts": float(mae_watts)
    }


# ============================================================
# SINGLE CONFIGURATION TRAINING
# ============================================================

def train_single_configuration(
    config_id,
    train_scaled_df,
    val_scaled_df,
    scaler,
    learning_rate,
    hidden_size,
    t_past,
    t_future,
    batch_size,
    max_epochs,
    patience,
    min_delta,
    device,
    seed
):
    """
    Train one LSTM configuration and return its best validation result.
    """

    set_seed(seed)

    print("\n" + "=" * 80)
    print(f"GRID CONFIG {config_id}")
    print("=" * 80)
    print(f"Learning rate: {learning_rate}")
    print(f"Hidden size:   {hidden_size}")
    print(f"Tpast:         {t_past} seconds")
    print(f"Tfuture:       {t_future} seconds")

    train_dataset = SUMOPowerWindowDataset(
        train_scaled_df,
        t_past_seconds=t_past,
        t_future_seconds=t_future,
        step_length=STEP_LENGTH,
        stride=1
    )

    val_dataset = SUMOPowerWindowDataset(
        val_scaled_df,
        t_past_seconds=t_past,
        t_future_seconds=t_future,
        step_length=STEP_LENGTH,
        stride=1
    )

    if len(train_dataset) == 0:
        raise RuntimeError(
            f"Training dataset has 0 windows for Tpast={t_past}."
        )

    if len(val_dataset) == 0:
        raise RuntimeError(
            f"Validation dataset has 0 windows for Tpast={t_past}."
        )

    print(f"Training windows:   {len(train_dataset)}")
    print(f"Validation windows: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False
    )

    model = PowerLSTMForecaster(
        input_size=len(FEATURE_COLUMNS),
        hidden_size=hidden_size,
        num_layers=1,
        dropout=0.0
    ).to(device)

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    best_val_rmse_watts = float("inf")
    best_val_rmse_scaled = float("inf")
    best_val_mae_watts = float("inf")
    best_epoch = 0
    best_state_dict = None

    epochs_without_improvement = 0
    config_history = []

    start_time = time.time()

    for epoch in range(1, max_epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device
        )

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            scaler=scaler
        )

        train_rmse_scaled = np.sqrt(train_loss)

        row = {
            "config_id": int(config_id),
            "epoch": int(epoch),
            "learning_rate": float(learning_rate),
            "hidden_size": int(hidden_size),
            "t_past": float(t_past),
            "t_future": float(t_future),
            "train_mse_scaled": float(train_loss),
            "train_rmse_scaled": float(train_rmse_scaled),
            "val_mse_scaled": float(val_metrics["mse_scaled"]),
            "val_rmse_scaled": float(val_metrics["rmse_scaled"]),
            "val_mae_scaled": float(val_metrics["mae_scaled"]),
            "val_mse_watts": float(val_metrics["mse_watts"]),
            "val_rmse_watts": float(val_metrics["rmse_watts"]),
            "val_mae_watts": float(val_metrics["mae_watts"])
        }

        config_history.append(row)

        print(
            f"Epoch [{epoch:03d}/{max_epochs}] "
            f"Train RMSE: {train_rmse_scaled:.6f} | "
            f"Val RMSE: {val_metrics['rmse_scaled']:.6f} | "
            f"Val RMSE Watts: {val_metrics['rmse_watts']:.2f} W"
        )

        improvement = best_val_rmse_watts - val_metrics["rmse_watts"]

        if improvement > min_delta:
            best_val_rmse_watts = float(val_metrics["rmse_watts"])
            best_val_rmse_scaled = float(val_metrics["rmse_scaled"])
            best_val_mae_watts = float(val_metrics["mae_watts"])
            best_epoch = int(epoch)
            epochs_without_improvement = 0

            best_state_dict = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(
                f"Early stopping for config {config_id} at epoch {epoch}. "
                f"Best epoch: {best_epoch}."
            )
            break

    elapsed_time = time.time() - start_time

    result = {
        "config_id": int(config_id),
        "learning_rate": float(learning_rate),
        "hidden_size": int(hidden_size),
        "t_past": float(t_past),
        "t_future": float(t_future),
        "batch_size": int(batch_size),
        "max_epochs": int(max_epochs),
        "best_epoch": int(best_epoch),
        "best_val_rmse_scaled": float(best_val_rmse_scaled),
        "best_val_rmse_watts": float(best_val_rmse_watts),
        "best_val_mae_watts": float(best_val_mae_watts),
        "train_windows": int(len(train_dataset)),
        "val_windows": int(len(val_dataset)),
        "elapsed_seconds": float(elapsed_time)
    }

    return result, config_history, best_state_dict


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="T2.4 Hyperparameter Tuning for centralized LSTM."
    )

    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=DATASET_DIR,
        help="Directory containing client_<number>.csv files."
    )

    parser.add_argument(
        "--num-clients",
        type=int,
        default=1000,
        help="Number of valid client CSV files to use."
    )

    parser.add_argument(
        "--min-rows",
        type=int,
        default=100,
        help="Minimum number of rows required for a client CSV to be used."
    )

    parser.add_argument(
        "--t-future",
        type=float,
        default=5.0,
        help="Future prediction window in seconds."
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size."
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=60,
        help="Maximum epochs per hyperparameter configuration."
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=8,
        help="Early stopping patience based on validation RMSE."
    )

    parser.add_argument(
        "--min-delta",
        type=float,
        default=1.0,
        help="Minimum improvement in validation RMSE watts."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed."
    )

    args = parser.parse_args()

    set_seed(args.seed)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    tuning_results_file = os.path.join(
        OUTPUT_DIR,
        "hyperparameter_tuning_results.csv"
    )

    tuning_history_file = os.path.join(
        OUTPUT_DIR,
        "hyperparameter_tuning_history.csv"
    )

    best_params_file = os.path.join(
        OUTPUT_DIR,
        "best_hyperparameters_for_federated.json"
    )

    best_model_file = os.path.join(
        OUTPUT_DIR,
        "best_tuned_centralized_lstm_model.pt"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\nT2.4 Hyperparameter Tuning started.")
    print(f"Device: {device}")
    print(f"Dataset directory: {args.dataset_dir}")
    print(f"Minimum rows per client: {args.min_rows}")
    print(f"Features: {FEATURE_COLUMNS}")
    print(f"Target: average future {TARGET_COLUMN}")
    print(f"Tfuture fixed at: {args.t_future} seconds")
    print("Grid:")
    print("  learning_rate: [1e-3, 1e-4]")
    print("  hidden_size:   [32, 64]")
    print("  Tpast:         [30, 60] seconds")

    # --------------------------------------------------------
    # 1. Load files and split train/validation
    # --------------------------------------------------------

    csv_files = find_client_files(
        dataset_dir=args.dataset_dir,
        num_clients=args.num_clients,
        client_ids=None,
        min_rows=args.min_rows
    )

    train_files, val_files = split_train_validation_files(
        csv_files,
        train_ratio=0.80,
        seed=args.seed
    )

    print(f"\nSelected valid client files: {len(csv_files)}")
    print(f"Training files: {len(train_files)}")
    print(f"Validation files: {len(val_files)}")

    # --------------------------------------------------------
    # 2. Aggregate once
    # --------------------------------------------------------

    print("\nAggregating training data...")
    train_df = load_and_aggregate_csvs(train_files)

    print("\nAggregating validation data...")
    val_df = load_and_aggregate_csvs(val_files)

    print("\nRaw data summary:")
    print(f"Training rows: {len(train_df)}")
    print(f"Validation rows: {len(val_df)}")
    print(f"Training vehicles: {train_df['vehicle_id'].nunique()}")
    print(f"Validation vehicles: {val_df['vehicle_id'].nunique()}")

    # --------------------------------------------------------
    # 3. Fit scaler on training data only
    # --------------------------------------------------------

    scaler = fit_scaler(train_df)
    joblib.dump(scaler, SCALER_OUTPUT_FILE)

    print(f"\nMinMaxScaler saved to: {SCALER_OUTPUT_FILE}")

    train_scaled_df = apply_scaler(train_df, scaler)
    val_scaled_df = apply_scaler(val_df, scaler)

    # --------------------------------------------------------
    # 4. Grid search
    # --------------------------------------------------------

    learning_rates = [1e-3, 1e-4]
    hidden_sizes = [32, 64]
    t_past_values = [30.0, 60.0]

    grid = list(
        itertools.product(
            learning_rates,
            hidden_sizes,
            t_past_values
        )
    )

    all_results = []
    all_history = []

    best_result = None
    best_state_dict = None

    for config_id, (learning_rate, hidden_size, t_past) in enumerate(
        grid,
        start=1
    ):
        result, config_history, state_dict = train_single_configuration(
            config_id=config_id,
            train_scaled_df=train_scaled_df,
            val_scaled_df=val_scaled_df,
            scaler=scaler,
            learning_rate=learning_rate,
            hidden_size=hidden_size,
            t_past=t_past,
            t_future=args.t_future,
            batch_size=args.batch_size,
            max_epochs=args.epochs,
            patience=args.patience,
            min_delta=args.min_delta,
            device=device,
            seed=args.seed + config_id
        )

        all_results.append(result)
        all_history.extend(config_history)

        if best_result is None:
            best_result = result
            best_state_dict = state_dict

        elif result["best_val_rmse_watts"] < best_result["best_val_rmse_watts"]:
            best_result = result
            best_state_dict = state_dict

        # Save intermediate results after each config
        pd.DataFrame(all_results).to_csv(tuning_results_file, index=False)
        pd.DataFrame(all_history).to_csv(tuning_history_file, index=False)

    if best_result is None or best_state_dict is None:
        raise RuntimeError("Grid search failed: no valid best result found.")

    # --------------------------------------------------------
    # 5. Save best parameters
    # --------------------------------------------------------

    best_params = {
        "model_type": "lstm",
        "learning_rate": float(best_result["learning_rate"]),
        "hidden_size": int(best_result["hidden_size"]),
        "t_past": float(best_result["t_past"]),
        "t_future": float(best_result["t_future"]),
        "batch_size": int(best_result["batch_size"]),
        "num_layers": 1,
        "dropout": 0.0,
        "step_length": float(STEP_LENGTH),
        "feature_columns": list(FEATURE_COLUMNS),
        "target_column": str(TARGET_COLUMN),
        "selection_metric": "validation_rmse_watts",
        "best_val_rmse_watts": float(best_result["best_val_rmse_watts"]),
        "best_val_rmse_scaled": float(best_result["best_val_rmse_scaled"]),
        "best_val_mae_watts": float(best_result["best_val_mae_watts"]),
        "best_epoch": int(best_result["best_epoch"]),
        "scaler_file": str(SCALER_OUTPUT_FILE)
    }

    best_params = make_json_serializable(best_params)

    with open(best_params_file, "w") as f:
        json.dump(best_params, f, indent=4)

    torch.save(
        {
            "model_state_dict": best_state_dict,
            "best_hyperparameters": best_params
        },
        best_model_file
    )

    # --------------------------------------------------------
    # 6. Final report
    # --------------------------------------------------------

    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values("best_val_rmse_watts")

    print("\n" + "=" * 80)
    print("T2.4 GRID SEARCH RESULTS")
    print("=" * 80)

    print(
        results_df[
            [
                "config_id",
                "learning_rate",
                "hidden_size",
                "t_past",
                "best_epoch",
                "best_val_rmse_scaled",
                "best_val_rmse_watts",
                "best_val_mae_watts",
                "train_windows",
                "val_windows"
            ]
        ].to_string(index=False)
    )

    print("\nBest configuration:")
    print(f"Learning rate:       {best_params['learning_rate']}")
    print(f"Hidden size:         {best_params['hidden_size']}")
    print(f"Input window Tpast:  {best_params['t_past']} seconds")
    print(f"Future window:       {best_params['t_future']} seconds")
    print(f"Best epoch:          {best_params['best_epoch']}")
    print(f"Validation RMSE:     {best_params['best_val_rmse_watts']:.2f} W")
    print(f"Validation MAE:      {best_params['best_val_mae_watts']:.2f} W")

    print("\nSaved files:")
    print(f"Grid results:        {tuning_results_file}")
    print(f"Epoch history:       {tuning_history_file}")
    print(f"Best parameters:     {best_params_file}")
    print(f"Best tuned model:    {best_model_file}")

    print("\nT2.4 Hyperparameter Tuning completed successfully.")
    print("Use best_hyperparameters_for_federated.json for the Federated phase.")


if __name__ == "__main__":
    main()