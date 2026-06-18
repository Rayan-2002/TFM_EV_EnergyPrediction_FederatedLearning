#!/usr/bin/env python3

import os
import random
import argparse

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

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

from model_architecture import (
    PowerLSTMForecaster,
    PowerGRUForecaster
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):
    """
    Fix random seeds for reproducibility.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

def split_train_test_files(csv_files, train_ratio=0.80, seed=42):
    """
    Split client files into 80% train and 20% test.

    Important:
    We split by client file, not by individual rows.

    This avoids data leakage where windows from the same vehicle appear
    in both training and test sets.
    """

    files = csv_files.copy()

    rng = random.Random(seed)
    rng.shuffle(files)

    n_total = len(files)

    if n_total < 2:
        raise ValueError("Need at least 2 client files for train/test split.")

    n_train = int(n_total * train_ratio)

    n_train = max(1, min(n_train, n_total - 1))

    train_files = files[:n_train]
    test_files = files[n_train:]

    return train_files, test_files


# ============================================================
# METRICS
# ============================================================

def compute_mse_rmse_mae(y_true, y_pred):
    """
    Compute MSE, RMSE, and MAE.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))

    return mse, rmse, mae


# ============================================================
# TRAINING AND EVALUATION
# ============================================================

def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Train the model for one epoch.
    """

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

        # Small safety measure against exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        batch_size = x_batch.size(0)

        total_loss += loss.item() * batch_size
        total_samples += batch_size

    average_loss = total_loss / total_samples

    return average_loss


def evaluate(model, loader, criterion, device, scaler=None):
    """
    Evaluate the model.

    Returns metrics in normalized scale and, if scaler is provided,
    also in watts.
    """

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

    average_loss = total_loss / total_samples

    y_true_scaled = np.concatenate(all_targets_scaled, axis=0).reshape(-1)
    y_pred_scaled = np.concatenate(all_predictions_scaled, axis=0).reshape(-1)

    mse_scaled, rmse_scaled, mae_scaled = compute_mse_rmse_mae(
        y_true_scaled,
        y_pred_scaled
    )

    results = {
        "loss": average_loss,
        "mse_scaled": mse_scaled,
        "rmse_scaled": rmse_scaled,
        "mae_scaled": mae_scaled
    }

    if scaler is not None:
        y_true_watts = inverse_scale_power(y_true_scaled, scaler)
        y_pred_watts = inverse_scale_power(y_pred_scaled, scaler)

        mse_watts, rmse_watts, mae_watts = compute_mse_rmse_mae(
            y_true_watts,
            y_pred_watts
        )

        results.update({
            "mse_watts": mse_watts,
            "rmse_watts": rmse_watts,
            "mae_watts": mae_watts
        })

    return results


def collect_predictions(model, loader, device, scaler):
    """
    Collect true and predicted values from the test DataLoader.

    Returns:
        y_true_watts
        y_pred_watts
    """

    model.eval()

    all_targets_scaled = []
    all_predictions_scaled = []

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)

            y_pred = model(x_batch)

            all_targets_scaled.append(y_batch.cpu().numpy())
            all_predictions_scaled.append(y_pred.cpu().numpy())

    y_true_scaled = np.concatenate(all_targets_scaled, axis=0).reshape(-1)
    y_pred_scaled = np.concatenate(all_predictions_scaled, axis=0).reshape(-1)

    y_true_watts = inverse_scale_power(y_true_scaled, scaler)
    y_pred_watts = inverse_scale_power(y_pred_scaled, scaler)

    return y_true_watts, y_pred_watts


# ============================================================
# PLOTTING
# ============================================================

def plot_rmse_curves(history_df, output_file):
    """
    Plot training RMSE and test RMSE over epochs.
    """

    plt.figure(figsize=(10, 6))

    plt.plot(
        history_df["epoch"],
        history_df["train_rmse_scaled"],
        label="Train RMSE scaled"
    )

    plt.plot(
        history_df["epoch"],
        history_df["test_rmse_scaled"],
        label="Test RMSE scaled"
    )

    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.title("Centralized Model RMSE Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(output_file, dpi=300)
    plt.close()

    print(f"RMSE curve saved to: {output_file}")


def plot_predictions_vs_true(
    y_true_watts,
    y_pred_watts,
    output_file,
    max_points=500
):
    """
    Plot true power and predicted power over a subset of test samples.
    """

    n = min(max_points, len(y_true_watts))

    plt.figure(figsize=(12, 6))

    plt.plot(
        range(n),
        y_true_watts[:n],
        label="True power"
    )

    plt.plot(
        range(n),
        y_pred_watts[:n],
        label="Predicted power"
    )

    plt.xlabel("Test sample index")
    plt.ylabel("Power (W)")
    plt.title("Predicted vs True Future Average Power")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(output_file, dpi=300)
    plt.close()

    print(f"Prediction curve saved to: {output_file}")


def plot_true_vs_pred_scatter(
    y_true_watts,
    y_pred_watts,
    output_file,
    max_points=2000
):
    """
    Scatter plot of true power vs predicted power.

    Perfect predictions should lie near the diagonal.
    """

    n = min(max_points, len(y_true_watts))

    y_true_subset = y_true_watts[:n]
    y_pred_subset = y_pred_watts[:n]

    min_value = min(y_true_subset.min(), y_pred_subset.min())
    max_value = max(y_true_subset.max(), y_pred_subset.max())

    plt.figure(figsize=(7, 7))

    plt.scatter(
        y_true_subset,
        y_pred_subset,
        alpha=0.4,
        s=12
    )

    plt.plot(
        [min_value, max_value],
        [min_value, max_value],
        linestyle="--",
        label="Perfect prediction"
    )

    plt.xlabel("True power (W)")
    plt.ylabel("Predicted power (W)")
    plt.title("True vs Predicted Power")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(output_file, dpi=300)
    plt.close()

    print(f"True-vs-predicted scatter plot saved to: {output_file}")


# ============================================================
# PRINTING HELPERS
# ============================================================

def print_file_split_summary(train_files, test_files):
    """
    Print train/test split summary.
    """

    print("\nTrain/test split:")
    print(f"Training files: {len(train_files)}")
    print(f"Test files: {len(test_files)}")


def print_dataset_summary(train_dataset, test_dataset):
    """
    Print dataset information.
    """

    print("\nDataset summary:")
    print(f"Training trajectories: {len(train_dataset.trajectories)}")
    print(f"Test trajectories: {len(test_dataset.trajectories)}")
    print(f"Training windows: {len(train_dataset)}")
    print(f"Test windows: {len(test_dataset)}")
    print(
        f"Input shape per sample: "
        f"({train_dataset.past_steps}, {len(FEATURE_COLUMNS)})"
    )
    print("Target shape per sample: (1,)")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="T2.3 Centralized Training for SUMO power prediction."
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
        "--t-past",
        type=float,
        default=10.0,
        help="Past time window in seconds."
    )

    parser.add_argument(
        "--t-future",
        type=float,
        default=5.0,
        help="Future time window in seconds."
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Sliding-window stride in rows."
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
        default=100,
        help="Maximum number of training epochs."
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Adam learning rate."
    )

    parser.add_argument(
        "--hidden-size",
        type=int,
        default=64,
        help="Hidden size of LSTM/GRU."
    )

    parser.add_argument(
        "--num-layers",
        type=int,
        default=1,
        help="Number of recurrent layers."
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="Dropout between recurrent layers. Ignored if num_layers=1."
    )

    parser.add_argument(
        "--model-type",
        type=str,
        choices=["lstm", "gru"],
        default="lstm",
        help="Model architecture to use."
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=12,
        help="Early stopping patience based on training loss."
    )

    parser.add_argument(
        "--min-delta",
        type=float,
        default=1e-6,
        help="Minimum training loss improvement required for convergence."
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

    plots_dir = os.path.join(OUTPUT_DIR, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    model_output_file = os.path.join(
        OUTPUT_DIR,
        f"centralized_{args.model_type}_model.pt"
    )

    metrics_output_file = os.path.join(
        OUTPUT_DIR,
        f"centralized_{args.model_type}_metrics.csv"
    )

    final_metrics_output_file = os.path.join(
        OUTPUT_DIR,
        f"centralized_{args.model_type}_final_metrics.csv"
    )

    rmse_curve_file = os.path.join(
        plots_dir,
        f"centralized_{args.model_type}_rmse_curve.png"
    )

    prediction_plot_file = os.path.join(
        plots_dir,
        f"centralized_{args.model_type}_predictions_vs_true.png"
    )

    scatter_plot_file = os.path.join(
        plots_dir,
        f"centralized_{args.model_type}_scatter_true_vs_pred.png"
    )

    print("\nT2.3 Centralized Training started.")
    print(f"Dataset directory: {args.dataset_dir}")
    print(f"Model type: {args.model_type.upper()}")
    print(f"Tpast: {args.t_past} seconds")
    print(f"Tfuture: {args.t_future} seconds")
    print(f"STEP_LENGTH: {STEP_LENGTH} second")
    print(f"Features: {FEATURE_COLUMNS}")
    print(f"Target: average future {TARGET_COLUMN}")
    print("Train/test split: 80% / 20%")
    print(f"Minimum rows per client: {args.min_rows}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")

    # --------------------------------------------------------
    # 1. Load valid client files
    # --------------------------------------------------------

    csv_files = find_client_files(
        dataset_dir=args.dataset_dir,
        num_clients=args.num_clients,
        client_ids=None,
        min_rows=args.min_rows
    )

    print(f"\nSelected {len(csv_files)} valid client files.")

    # --------------------------------------------------------
    # 2. Split 80% train / 20% test
    # --------------------------------------------------------

    train_files, test_files = split_train_test_files(
        csv_files,
        train_ratio=0.80,
        seed=args.seed
    )

    print_file_split_summary(train_files, test_files)

    # --------------------------------------------------------
    # 3. Aggregate CSVs
    # --------------------------------------------------------

    print("\nAggregating training data...")
    train_df = load_and_aggregate_csvs(train_files)

    print("\nAggregating test data...")
    test_df = load_and_aggregate_csvs(test_files)

    print("\nRaw data summary:")
    print(f"Training rows: {len(train_df)}")
    print(f"Test rows: {len(test_df)}")
    print(f"Training vehicles: {train_df['vehicle_id'].nunique()}")
    print(f"Test vehicles: {test_df['vehicle_id'].nunique()}")

    # --------------------------------------------------------
    # 4. Fit scaler on training data only
    # --------------------------------------------------------

    scaler = fit_scaler(train_df)

    joblib.dump(scaler, SCALER_OUTPUT_FILE)

    print(f"\nMinMaxScaler saved to: {SCALER_OUTPUT_FILE}")

    train_scaled_df = apply_scaler(train_df, scaler)
    test_scaled_df = apply_scaler(test_df, scaler)

    # --------------------------------------------------------
    # 5. Create PyTorch datasets
    # --------------------------------------------------------

    train_dataset = SUMOPowerWindowDataset(
        train_scaled_df,
        t_past_seconds=args.t_past,
        t_future_seconds=args.t_future,
        step_length=STEP_LENGTH,
        stride=args.stride
    )

    test_dataset = SUMOPowerWindowDataset(
        test_scaled_df,
        t_past_seconds=args.t_past,
        t_future_seconds=args.t_future,
        step_length=STEP_LENGTH,
        stride=args.stride
    )

    if len(train_dataset) == 0:
        raise RuntimeError(
            "Training dataset has 0 windows. "
            "Reduce --t-past or --t-future, or generate longer trajectories."
        )

    if len(test_dataset) == 0:
        raise RuntimeError(
            "Test dataset has 0 windows. "
            "Reduce --t-past or --t-future, or generate longer trajectories."
        )

    print_dataset_summary(train_dataset, test_dataset)

    # --------------------------------------------------------
    # 6. Create DataLoaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False
    )

    # --------------------------------------------------------
    # 7. Build model
    # --------------------------------------------------------

    input_size = len(FEATURE_COLUMNS)

    if args.model_type == "lstm":
        model = PowerLSTMForecaster(
            input_size=input_size,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout
        )
    else:
        model = PowerGRUForecaster(
            input_size=input_size,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout
        )

    model = model.to(device)

    print("\nModel architecture:")
    print(model)

    # --------------------------------------------------------
    # 8. Loss and optimizer
    # --------------------------------------------------------

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate
    )

    # --------------------------------------------------------
    # 9. Training loop
    # --------------------------------------------------------

    history = []

    best_train_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    print("\nTraining started...")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device
        )

        train_rmse_scaled = np.sqrt(train_loss)

        test_metrics = evaluate(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
            scaler=scaler
        )

        row = {
            "epoch": epoch,
            "train_mse_scaled": train_loss,
            "train_rmse_scaled": train_rmse_scaled,
            "test_mse_scaled": test_metrics["mse_scaled"],
            "test_rmse_scaled": test_metrics["rmse_scaled"],
            "test_mae_scaled": test_metrics["mae_scaled"],
            "test_mse_watts": test_metrics["mse_watts"],
            "test_rmse_watts": test_metrics["rmse_watts"],
            "test_mae_watts": test_metrics["mae_watts"]
        }

        history.append(row)

        print(
            f"Epoch [{epoch:03d}/{args.epochs}] "
            f"Train MSE: {train_loss:.8f} | "
            f"Train RMSE: {train_rmse_scaled:.6f} | "
            f"Test RMSE: {test_metrics['rmse_scaled']:.6f} | "
            f"Test RMSE Watts: {test_metrics['rmse_watts']:.2f} W"
        )

        # ----------------------------------------------------
        # Early stopping based on training loss convergence.
        # We do not use test loss to decide stopping.
        # ----------------------------------------------------

        improvement = best_train_loss - train_loss

        if improvement > args.min_delta:
            best_train_loss = train_loss
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_type": args.model_type,
                    "input_size": input_size,
                    "hidden_size": args.hidden_size,
                    "num_layers": args.num_layers,
                    "dropout": args.dropout,
                    "t_past": args.t_past,
                    "t_future": args.t_future,
                    "step_length": STEP_LENGTH,
                    "feature_columns": FEATURE_COLUMNS,
                    "target_column": TARGET_COLUMN,
                    "scaler_file": SCALER_OUTPUT_FILE,
                    "best_epoch": best_epoch,
                    "best_train_loss": best_train_loss
                },
                model_output_file
            )

        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience:
            print(
                f"\nEarly stopping triggered at epoch {epoch}. "
                f"Best epoch: {best_epoch}."
            )
            break

    # --------------------------------------------------------
    # 10. Save history and RMSE curve
    # --------------------------------------------------------

    history_df = pd.DataFrame(history)
    history_df.to_csv(metrics_output_file, index=False)

    print(f"\nTraining history saved to: {metrics_output_file}")
    print(f"Best model saved to: {model_output_file}")

    plot_rmse_curves(
        history_df=history_df,
        output_file=rmse_curve_file
    )

    # --------------------------------------------------------
    # 11. Load best model and compute final test RMSE
    # --------------------------------------------------------

    checkpoint = torch.load(
        model_output_file,
        map_location=device
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    final_test_metrics = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        scaler=scaler
    )

    final_metrics_df = pd.DataFrame([
        {
            "model_type": args.model_type,
            "best_epoch": checkpoint["best_epoch"],
            "t_past": args.t_past,
            "t_future": args.t_future,
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "train_files": len(train_files),
            "test_files": len(test_files),
            "train_windows": len(train_dataset),
            "test_windows": len(test_dataset),
            "final_test_mse_scaled": final_test_metrics["mse_scaled"],
            "final_test_rmse_scaled": final_test_metrics["rmse_scaled"],
            "final_test_mae_scaled": final_test_metrics["mae_scaled"],
            "final_test_mse_watts": final_test_metrics["mse_watts"],
            "final_test_rmse_watts": final_test_metrics["rmse_watts"],
            "final_test_mae_watts": final_test_metrics["mae_watts"]
        }
    ])

    final_metrics_df.to_csv(final_metrics_output_file, index=False)

    print(f"\nFinal metrics saved to: {final_metrics_output_file}")

    # --------------------------------------------------------
    # 12. Prediction plots
    # --------------------------------------------------------

    y_true_watts, y_pred_watts = collect_predictions(
        model=model,
        loader=test_loader,
        device=device,
        scaler=scaler
    )

    plot_predictions_vs_true(
        y_true_watts=y_true_watts,
        y_pred_watts=y_pred_watts,
        output_file=prediction_plot_file,
        max_points=500
    )

    plot_true_vs_pred_scatter(
        y_true_watts=y_true_watts,
        y_pred_watts=y_pred_watts,
        output_file=scatter_plot_file,
        max_points=2000
    )

    # --------------------------------------------------------
    # 13. Final console report
    # --------------------------------------------------------

    print("\nFinal centralized test results:")
    print(f"Best epoch:              {checkpoint['best_epoch']}")
    print(f"Final Test MSE scaled:   {final_test_metrics['mse_scaled']:.8f}")
    print(f"Final Test RMSE scaled:  {final_test_metrics['rmse_scaled']:.6f}")
    print(f"Final Test MAE scaled:   {final_test_metrics['mae_scaled']:.6f}")

    print(f"\nFinal Test MSE watts:    {final_test_metrics['mse_watts']:.2f}")
    print(f"Final Test RMSE watts:   {final_test_metrics['rmse_watts']:.2f} W")
    print(f"Final Test MAE watts:    {final_test_metrics['mae_watts']:.2f} W")

    print("\nGenerated files:")
    print(f"Model:                  {model_output_file}")
    print(f"Training history:       {metrics_output_file}")
    print(f"Final metrics:          {final_metrics_output_file}")
    print(f"RMSE curve:             {rmse_curve_file}")
    print(f"Predictions curve:      {prediction_plot_file}")
    print(f"True-vs-pred scatter:   {scatter_plot_file}")

    print("\nT2.3 Centralized Training completed successfully.")


if __name__ == "__main__":
    main()