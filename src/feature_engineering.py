#!/usr/bin/env python3

import os
import re
import random
import argparse

import numpy as np
import pandas as pd
import joblib

import torch
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import MinMaxScaler


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = os.path.expanduser("~/Internship/SUMO_Barcelona")

DATASET_DIR = os.path.join(PROJECT_DIR, "dataset")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")

SCALER_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "minmax_scaler.joblib")

STEP_LENGTH = 1.0  # seconds

FEATURE_COLUMNS = [
    "speed",
    "acceleration",
    "road_grade_angle",
    "consumption_power_watts"
]

TARGET_COLUMN = "consumption_power_watts"
TARGET_INDEX = FEATURE_COLUMNS.index(TARGET_COLUMN)


# ============================================================
# FILE LOADING
# ============================================================

def extract_client_id(file_path):
    """
    Extract numeric client id from filenames like client_0.csv.
    """
    filename = os.path.basename(file_path)
    match = re.fullmatch(r"client_(\d+)\.csv", filename)

    if match is None:
        return float("inf")

    return int(match.group(1))


def count_csv_rows(file_path):
    """
    Count data rows in a CSV file, excluding the header.
    """
    with open(file_path, "r") as f:
        return max(0, sum(1 for _ in f) - 1)


def find_client_files(dataset_dir, num_clients=None, client_ids=None, min_rows=0):
    """
    Find client CSV files.

    Only files named client_<number>.csv are considered.
    This excludes files like client_mapping.csv.

    If client_ids is provided:
        - only those clients are loaded;
        - min_rows does not exclude them, but a warning is printed.

    Otherwise:
        - all client_<number>.csv files with at least min_rows are selected;
        - the first num_clients are returned after sorting by client id.
    """

    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(
            f"Dataset directory '{dataset_dir}' does not exist."
        )

    # --------------------------------------------------------
    # Case 1: explicit client IDs
    # --------------------------------------------------------

    if client_ids is not None and len(client_ids) > 0:
        selected_files = []

        for client_id in client_ids:
            expected_name = f"client_{client_id}.csv"
            expected_path = os.path.join(dataset_dir, expected_name)

            if not os.path.exists(expected_path):
                raise FileNotFoundError(
                    f"Requested client file '{expected_name}' does not exist "
                    f"in '{dataset_dir}'."
                )

            num_rows = count_csv_rows(expected_path)

            if num_rows < min_rows:
                print(
                    f"WARNING: {expected_name} has only {num_rows} rows, "
                    f"below min_rows={min_rows}, but it was explicitly requested."
                )

            selected_files.append(expected_path)

        selected_files.sort(key=extract_client_id)
        return selected_files

    # --------------------------------------------------------
    # Case 2: automatic client selection
    # --------------------------------------------------------

    all_files = []

    for filename in os.listdir(dataset_dir):
        # Very important:
        # This keeps client_0.csv, client_17.csv, etc.
        # This ignores client_mapping.csv.
        if re.fullmatch(r"client_\d+\.csv", filename):
            file_path = os.path.join(dataset_dir, filename)
            num_rows = count_csv_rows(file_path)

            if num_rows >= min_rows:
                all_files.append(file_path)

    all_files.sort(key=extract_client_id)

    if len(all_files) == 0:
        raise FileNotFoundError(
            f"No client CSV files found in '{dataset_dir}' "
            f"with at least min_rows={min_rows}."
        )

    if num_clients is not None:
        return all_files[:num_clients]

    return all_files


def load_and_aggregate_csvs(csv_files):
    """
    Load several client CSVs and aggregate them into one DataFrame.
    """

    dataframes = []

    print("\nLoading CSV files:")

    for file_path in csv_files:
        df = pd.read_csv(file_path)

        missing_columns = [col for col in FEATURE_COLUMNS if col not in df.columns]

        if len(missing_columns) > 0:
            raise ValueError(
                f"File {file_path} is missing feature columns: {missing_columns}"
            )

        required_columns = [
            "time",
            "vehicle_id",
            "segment_id"
        ]

        for col in required_columns:
            if col not in df.columns:
                raise ValueError(
                    f"File {file_path} is missing required column: {col}"
                )

        df["source_file"] = os.path.basename(file_path)

        dataframes.append(df)

        print(f"  - {os.path.basename(file_path)}: {len(df)} rows")

    if len(dataframes) == 0:
        raise RuntimeError("No DataFrames were loaded.")

    full_df = pd.concat(dataframes, ignore_index=True)

    full_df = full_df.sort_values(
        [
            "source_file",
            "vehicle_id",
            "segment_id",
            "time"
        ]
    ).reset_index(drop=True)

    print(f"Aggregated rows: {len(full_df)}")
    print(f"Aggregated source files: {full_df['source_file'].nunique()}")
    print(f"Aggregated vehicles: {full_df['vehicle_id'].nunique()}")

    if full_df["source_file"].nunique() != len(csv_files):
        print(
            "WARNING: Number of unique source files does not match "
            "number of input files."
        )

    return full_df


# ============================================================
# TRAIN / VALIDATION / TEST SPLIT
# ============================================================

def split_client_files(csv_files, train_ratio=0.70, val_ratio=0.15, seed=42):
    """
    Split by client files.

    This avoids data leakage where windows from the same vehicle appear
    in both training and validation/test sets.
    """

    files = csv_files.copy()

    rng = random.Random(seed)
    rng.shuffle(files)

    n = len(files)

    if n < 3:
        print(
            "WARNING: Fewer than 3 client files selected. "
            "Using all files for training only."
        )
        return files, [], []

    n_train = max(1, int(n * train_ratio))
    n_val = max(1, int(n * val_ratio))

    if n_train + n_val >= n:
        n_train = n - 2
        n_val = 1

    train_files = files[:n_train]
    val_files = files[n_train:n_train + n_val]
    test_files = files[n_train + n_val:]

    return train_files, val_files, test_files


# ============================================================
# NORMALIZATION
# ============================================================

def fit_scaler(train_df):
    """
    Fit MinMaxScaler only on the training data.

    This avoids leaking validation/test information into training.
    """

    scaler = MinMaxScaler()

    train_features = train_df[FEATURE_COLUMNS].astype(np.float32)
    scaler.fit(train_features)

    return scaler


def apply_scaler(df, scaler):
    """
    Apply an already-fitted MinMaxScaler.
    """

    if df is None or len(df) == 0:
        return df

    scaled_df = df.copy()

    scaled_df[FEATURE_COLUMNS] = scaler.transform(
        scaled_df[FEATURE_COLUMNS].astype(np.float32)
    )

    return scaled_df


def inverse_scale_power(y_scaled, scaler):
    """
    Convert normalized power values back to watts.

    Useful later when evaluating model predictions.
    """

    y_scaled = np.asarray(y_scaled)

    power_original = (
        y_scaled - scaler.min_[TARGET_INDEX]
    ) / scaler.scale_[TARGET_INDEX]

    return power_original


# ============================================================
# CUSTOM PYTORCH DATASET
# ============================================================

class SUMOPowerWindowDataset(Dataset):
    """
    PyTorch Dataset for SUMO power prediction.

    For each sample:

        X = past Tpast seconds of:
            [speed, acceleration, road_grade_angle, consumption_power_watts]

        Y = average future consumption_power_watts over Tfuture seconds

    With STEP_LENGTH = 1.0:

        Tpast = 10   -> 10 rows
        Tfuture = 5  -> 5 rows

    X shape for one sample:
        (past_steps, num_features)

    Y shape for one sample:
        (1,)
    """

    def __init__(
        self,
        df,
        t_past_seconds,
        t_future_seconds,
        step_length=1.0,
        stride=1
    ):
        super().__init__()

        if df is None or len(df) == 0:
            raise ValueError("Cannot create Dataset from an empty DataFrame.")

        self.df = df.copy()
        self.t_past_seconds = t_past_seconds
        self.t_future_seconds = t_future_seconds
        self.step_length = step_length
        self.stride = stride

        self.past_steps = int(round(t_past_seconds / step_length))
        self.future_steps = int(round(t_future_seconds / step_length))

        if self.past_steps <= 0:
            raise ValueError("past_steps must be > 0")

        if self.future_steps <= 0:
            raise ValueError("future_steps must be > 0")

        if stride <= 0:
            raise ValueError("stride must be > 0")

        self.trajectories = []
        self.samples = []

        self._build_index()

    def _build_index(self):
        """
        Build valid sliding-window indices.

        Windows are created inside each:
            source_file + vehicle_id + segment_id

        This prevents invalid windows across:
            - different client files
            - different vehicles
            - separated sequence segments
        """

        group_columns = [
            "source_file",
            "vehicle_id",
            "segment_id"
        ]

        total_required_steps = self.past_steps + self.future_steps

        for _, group in self.df.groupby(group_columns, sort=False):
            group = group.sort_values("time").copy()

            values = group[FEATURE_COLUMNS].to_numpy(dtype=np.float32)

            if len(values) < total_required_steps:
                continue

            trajectory_index = len(self.trajectories)
            self.trajectories.append(values)

            max_start = len(values) - total_required_steps

            for start_idx in range(0, max_start + 1, self.stride):
                self.samples.append((trajectory_index, start_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        trajectory_index, start_idx = self.samples[index]

        trajectory = self.trajectories[trajectory_index]

        past_start = start_idx
        past_end = past_start + self.past_steps

        future_start = past_end
        future_end = future_start + self.future_steps

        x = trajectory[past_start:past_end, :].copy()

        future_power = trajectory[
            future_start:future_end,
            TARGET_INDEX
        ]

        y = np.array([future_power.mean()], dtype=np.float32)

        return torch.from_numpy(x), torch.from_numpy(y.copy())


# ============================================================
# DIAGNOSTICS
# ============================================================

def print_dataframe_summary(name, df):
    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    if df is None or len(df) == 0:
        print("Empty DataFrame.")
        return

    print(f"Rows: {len(df)}")
    print(f"Unique vehicles: {df['vehicle_id'].nunique()}")
    print(f"Unique source files: {df['source_file'].nunique()}")

    unique_segments = (
        df.groupby(["source_file", "vehicle_id"])["segment_id"]
        .nunique()
        .sum()
    )

    print(f"Unique segments: {unique_segments}")

    print("\nFeature summary:")
    print(df[FEATURE_COLUMNS].describe())

    print("\nMissing values:")
    print(df[FEATURE_COLUMNS].isna().sum())


def print_dataset_summary(name, dataset):
    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    if dataset is None:
        print("Dataset not created.")
        return

    print(f"Number of trajectories: {len(dataset.trajectories)}")
    print(f"Number of sliding windows: {len(dataset)}")
    print(f"Tpast seconds: {dataset.t_past_seconds}")
    print(f"Tfuture seconds: {dataset.t_future_seconds}")
    print(f"Past steps: {dataset.past_steps}")
    print(f"Future steps: {dataset.future_steps}")
    print(f"Number of features: {len(FEATURE_COLUMNS)}")


def print_batch_example(loader, name):
    print("\n" + "=" * 70)
    print(f"Batch example: {name}")
    print("=" * 70)

    x_batch, y_batch = next(iter(loader))

    print(f"X batch shape: {x_batch.shape}")
    print(f"Y batch shape: {y_batch.shape}")

    print("\nExpected:")
    print("X shape = (batch_size, past_steps, num_features)")
    print("Y shape = (batch_size, 1)")

    print("\nFirst X sample:")
    print(x_batch[0])

    print("\nFirst Y sample:")
    print(y_batch[0])


def print_split_files(name, files):
    print(f"\n{name}: {len(files)} files")

    for file_path in files:
        print(f"  - {os.path.basename(file_path)}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="T2.1 Feature Engineering for SUMO power prediction."
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
        help="Number of valid client CSV files to aggregate."
    )

    parser.add_argument(
        "--min-rows",
        type=int,
        default=100,
        help="Minimum number of rows required for a client CSV to be used."
    )

    parser.add_argument(
        "--client-ids",
        type=int,
        nargs="*",
        default=None,
        help="Specific client IDs to use, e.g. --client-ids 0 1 2 3"
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
        default=32,
        help="Batch size for DataLoader."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/validation/test split."
    )

    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\nT2.1 Feature Engineering started.")
    print(f"Dataset directory: {args.dataset_dir}")
    print(f"Tpast: {args.t_past} seconds")
    print(f"Tfuture: {args.t_future} seconds")
    print(f"STEP_LENGTH: {STEP_LENGTH} second")
    print(f"Features: {FEATURE_COLUMNS}")
    print(f"Target: average future {TARGET_COLUMN}")
    print(f"Minimum rows per client: {args.min_rows}")

    # --------------------------------------------------------
    # 1. Select client CSV files
    # --------------------------------------------------------

    csv_files = find_client_files(
        dataset_dir=args.dataset_dir,
        num_clients=args.num_clients,
        client_ids=args.client_ids,
        min_rows=args.min_rows
    )

    print(f"\nSelected {len(csv_files)} valid client files:")

    for file_path in csv_files:
        print(
            f"  - {os.path.basename(file_path)} "
            f"({count_csv_rows(file_path)} rows)"
        )

    # --------------------------------------------------------
    # 2. Split client files
    # --------------------------------------------------------

    train_files, val_files, test_files = split_client_files(
        csv_files,
        train_ratio=0.70,
        val_ratio=0.15,
        seed=args.seed
    )

    print("\nClient split:")
    print(f"Train files: {len(train_files)}")
    print(f"Validation files: {len(val_files)}")
    print(f"Test files: {len(test_files)}")

    print_split_files("Training files", train_files)
    print_split_files("Validation files", val_files)
    print_split_files("Test files", test_files)

    # --------------------------------------------------------
    # 3. Aggregate CSVs
    # --------------------------------------------------------

    train_df = load_and_aggregate_csvs(train_files)

    val_df = (
        load_and_aggregate_csvs(val_files)
        if len(val_files) > 0
        else None
    )

    test_df = (
        load_and_aggregate_csvs(test_files)
        if len(test_files) > 0
        else None
    )

    print_dataframe_summary("Raw training data", train_df)
    print_dataframe_summary("Raw validation data", val_df)
    print_dataframe_summary("Raw test data", test_df)

    # --------------------------------------------------------
    # 4. Fit MinMaxScaler on training data only
    # --------------------------------------------------------

    scaler = fit_scaler(train_df)

    joblib.dump(scaler, SCALER_OUTPUT_FILE)
    print(f"\nMinMaxScaler saved to: {SCALER_OUTPUT_FILE}")

    # --------------------------------------------------------
    # 5. Apply scaler
    # --------------------------------------------------------

    train_scaled_df = apply_scaler(train_df, scaler)
    val_scaled_df = apply_scaler(val_df, scaler)
    test_scaled_df = apply_scaler(test_df, scaler)

    print_dataframe_summary("Scaled training data", train_scaled_df)
    print_dataframe_summary("Scaled validation data", val_scaled_df)
    print_dataframe_summary("Scaled test data", test_scaled_df)

    # --------------------------------------------------------
    # 6. Create PyTorch Datasets
    # --------------------------------------------------------

    train_dataset = SUMOPowerWindowDataset(
        train_scaled_df,
        t_past_seconds=args.t_past,
        t_future_seconds=args.t_future,
        step_length=STEP_LENGTH,
        stride=args.stride
    )

    val_dataset = (
        SUMOPowerWindowDataset(
            val_scaled_df,
            t_past_seconds=args.t_past,
            t_future_seconds=args.t_future,
            step_length=STEP_LENGTH,
            stride=args.stride
        )
        if val_scaled_df is not None and len(val_scaled_df) > 0
        else None
    )

    test_dataset = (
        SUMOPowerWindowDataset(
            test_scaled_df,
            t_past_seconds=args.t_past,
            t_future_seconds=args.t_future,
            step_length=STEP_LENGTH,
            stride=args.stride
        )
        if test_scaled_df is not None and len(test_scaled_df) > 0
        else None
    )

    print_dataset_summary("Training Dataset", train_dataset)
    print_dataset_summary("Validation Dataset", val_dataset)
    print_dataset_summary("Test Dataset", test_dataset)

    if len(train_dataset) == 0:
        raise RuntimeError(
            "Training dataset has 0 windows. "
            "Try reducing --t-past or --t-future, "
            "or generate longer vehicle trajectories."
        )

    # --------------------------------------------------------
    # 7. Create DataLoaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False
    )

    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False
        )
        if val_dataset is not None and len(val_dataset) > 0
        else None
    )

    test_loader = (
        DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False
        )
        if test_dataset is not None and len(test_dataset) > 0
        else None
    )

    # Suppress "unused variable" confusion for now.
    # These loaders will be used during T2.2 training.
    _ = val_loader
    _ = test_loader

    # --------------------------------------------------------
    # 8. Verify one batch
    # --------------------------------------------------------

    print_batch_example(train_loader, "Training loader")

    print("\nT2.1 Feature Engineering completed successfully.")

    print("\nFinal interpretation:")
    print("Each X contains the past driving sequence.")
    print("Each Y contains the normalized average future power.")
    print("Later, the model will learn: X -> Y.")

    print("\nTensor meaning:")
    print("X[:, :, 0] = normalized speed")
    print("X[:, :, 1] = normalized acceleration")
    print("X[:, :, 2] = normalized road grade angle")
    print("X[:, :, 3] = normalized consumption power")
    print("Y[:, 0]    = normalized average future consumption power")


if __name__ == "__main__":
    main()