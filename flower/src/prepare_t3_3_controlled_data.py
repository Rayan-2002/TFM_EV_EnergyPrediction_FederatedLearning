from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "speed",
    "acceleration",
    "road_grade_angle",
    "consumption_power_watts",
]
TARGET_COLUMN = "consumption_power_watts"
ACCELERATION_COLUMN = "acceleration"
T_PAST = 30
T_FUTURE = 5
TRAIN_SPLIT = 0.8
CALM_CODE = 0
AGGRESSIVE_CODE = 1


FLOWER_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "dataset"
DEFAULT_OUTPUT_DIR = FLOWER_ROOT / "data" / "t3_3_controlled"


def discover_candidate_ids() -> List[int]:
    client_ids: List[int] = []
    for path in DATASET_DIR.glob("client_*.csv"):
        suffix = path.stem.removeprefix("client_")
        if suffix.isdigit():
            client_ids.append(int(suffix))
    client_ids.sort()
    if not client_ids:
        raise FileNotFoundError(f"No client_<id>.csv files found in {DATASET_DIR}")
    return client_ids


def load_route(client_id: int) -> pd.DataFrame:
    path = DATASET_DIR / f"client_{client_id}.csv"
    required = list(dict.fromkeys(FEATURE_COLUMNS + [TARGET_COLUMN]))
    df = pd.read_csv(path)
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    return df.dropna(subset=required).reset_index(drop=True)


def build_raw_windows(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    if len(df) < T_PAST + T_FUTURE:
        raise ValueError(
            f"Need at least {T_PAST + T_FUTURE} rows, found {len(df)}."
        )

    features = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    target = df[TARGET_COLUMN].to_numpy(dtype=np.float32)
    num_windows = len(df) - T_PAST - T_FUTURE + 1

    x = np.empty(
        (num_windows, T_PAST, len(FEATURE_COLUMNS)),
        dtype=np.float32,
    )
    y = np.empty((num_windows, 1), dtype=np.float32)

    for start in range(num_windows):
        past_end = start + T_PAST
        future_end = past_end + T_FUTURE
        x[start] = features[start:past_end]
        y[start, 0] = float(target[past_end:future_end].mean())

    return x, y


def chronological_window_split(
    x: np.ndarray,
    y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split windows chronologically instead of randomly.

    The earliest 80% of windows are used for local training and the latest
    20% are reserved for one fixed evaluation set shared by both experiments.
    """

    train_count = int(TRAIN_SPLIT * len(x))
    train_count = max(1, min(train_count, len(x) - 1))
    return x[:train_count], y[:train_count], x[train_count:], y[train_count:]


def scan_eligible_routes() -> Tuple[Dict[int, dict], List[Tuple[int, str]]]:
    route_data: Dict[int, dict] = {}
    skipped: List[Tuple[int, str]] = []

    for client_id in discover_candidate_ids():
        try:
            df = load_route(client_id)
            x, y = build_raw_windows(df)
            if len(x) < 2:
                raise ValueError("fewer than two temporal windows")
            x_train, y_train, x_val, y_val = chronological_window_split(x, y)
            acceleration = pd.to_numeric(
                df[ACCELERATION_COLUMN], errors="coerce"
            ).dropna()
            if acceleration.empty:
                raise ValueError("no usable acceleration values")

            route_data[client_id] = {
                "x_train_raw": x_train,
                "y_train_raw": y_train,
                "x_val_raw": x_val,
                "y_val_raw": y_val,
                "num_rows": int(len(df)),
                "mean_acceleration": float(acceleration.mean()),
                "mean_abs_acceleration": float(acceleration.abs().mean()),
                "acceleration_std": float(acceleration.std(ddof=0)),
            }
        except Exception as error:  # Keep a complete audit of unusable routes.
            skipped.append((client_id, str(error)))

    if not route_data:
        raise RuntimeError("No route can be used for the controlled experiment.")

    return route_data, skipped


def assign_extreme_profiles(
    route_data: Dict[int, dict],
    tail_fraction: float,
) -> Tuple[pd.DataFrame, List[int], List[int]]:
    if not 0.0 < tail_fraction <= 0.5:
        raise ValueError("tail_fraction must be in (0, 0.5].")

    records = []
    for client_id, values in route_data.items():
        records.append(
            {
                "client_id": client_id,
                "num_rows": values["num_rows"],
                "num_train_windows": len(values["x_train_raw"]),
                "num_val_windows": len(values["x_val_raw"]),
                "mean_acceleration": values["mean_acceleration"],
                "mean_abs_acceleration": values["mean_abs_acceleration"],
                "acceleration_std": values["acceleration_std"],
            }
        )

    profiles = pd.DataFrame(records).sort_values(
        ["mean_acceleration", "client_id"]
    ).reset_index(drop=True)
    tail_count = max(1, int(len(profiles) * tail_fraction))
    if 2 * tail_count > len(profiles):
        raise ValueError("Profile tails overlap.")

    profiles["profile"] = "Middle"
    profiles["profile_code"] = -1
    calm_idx = profiles.index[:tail_count]
    aggressive_idx = profiles.index[-tail_count:]
    profiles.loc[calm_idx, ["profile", "profile_code"]] = ["Calm", CALM_CODE]
    profiles.loc[aggressive_idx, ["profile", "profile_code"]] = [
        "Aggressive",
        AGGRESSIVE_CODE,
    ]

    calm_ids = profiles.loc[calm_idx, "client_id"].astype(int).tolist()
    aggressive_ids = (
        profiles.loc[aggressive_idx, "client_id"].astype(int).tolist()
    )
    return profiles, calm_ids, aggressive_ids


def compute_global_scaler(
    route_data: Dict[int, dict], selected_ids: List[int]
) -> dict:
    """Compute one scaler from training windows only, shared by all clients."""

    all_x_train = np.concatenate(
        [route_data[cid]["x_train_raw"] for cid in selected_ids], axis=0
    )
    all_y_train = np.concatenate(
        [route_data[cid]["y_train_raw"] for cid in selected_ids], axis=0
    )

    feature_mean = all_x_train.mean(axis=(0, 1), dtype=np.float64)
    feature_std = all_x_train.std(axis=(0, 1), dtype=np.float64)
    feature_std = np.where(feature_std < 1e-8, 1.0, feature_std)

    target_mean = float(all_y_train.mean(dtype=np.float64))
    target_std = float(all_y_train.std(dtype=np.float64))
    if target_std < 1e-8:
        target_std = 1.0

    return {
        "feature_columns": FEATURE_COLUMNS,
        "feature_mean": feature_mean.tolist(),
        "feature_std": feature_std.tolist(),
        "target_column": TARGET_COLUMN,
        "target_mean": target_mean,
        "target_std": target_std,
        "computed_from": "all selected training windows only",
    }


def normalize(
    x: np.ndarray,
    y: np.ndarray,
    scaler: dict,
) -> Tuple[np.ndarray, np.ndarray]:
    feature_mean = np.asarray(scaler["feature_mean"], dtype=np.float32)
    feature_std = np.asarray(scaler["feature_std"], dtype=np.float32)
    target_mean = np.float32(scaler["target_mean"])
    target_std = np.float32(scaler["target_std"])

    x_norm = (x.astype(np.float32) - feature_mean) / feature_std
    y_norm = (y.astype(np.float32) - target_mean) / target_std
    return x_norm.astype(np.float32), y_norm.astype(np.float32)


def save_partition(
    path: Path,
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    partition_id: int,
    mode: str,
    slot_profile_code: int,
    source_client_id: int,
    num_calm_examples: int,
    num_aggressive_examples: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        x_train=x_train.astype(np.float32),
        y_train=y_train.astype(np.float32),
        partition_id=np.int64(partition_id),
        mode=np.asarray(mode),
        slot_profile_code=np.int64(slot_profile_code),
        source_client_id=np.int64(source_client_id),
        num_calm_examples=np.int64(num_calm_examples),
        num_aggressive_examples=np.int64(num_aggressive_examples),
    )


def prepare(args: argparse.Namespace) -> None:
    if args.output_dir.exists() and args.overwrite:
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    route_data, skipped = scan_eligible_routes()
    profiles, calm_ids, aggressive_ids = assign_extreme_profiles(
        route_data, args.tail_fraction
    )
    selected_ids = calm_ids + aggressive_ids
    profile_code_by_id = {cid: CALM_CODE for cid in calm_ids}
    profile_code_by_id.update({cid: AGGRESSIVE_CODE for cid in aggressive_ids})

    scaler = compute_global_scaler(route_data, selected_ids)

    normalized_by_id: Dict[int, dict] = {}
    for client_id in selected_ids:
        values = route_data[client_id]
        x_train, y_train = normalize(
            values["x_train_raw"], values["y_train_raw"], scaler
        )
        x_val, y_val = normalize(
            values["x_val_raw"], values["y_val_raw"], scaler
        )
        normalized_by_id[client_id] = {
            "x_train": x_train,
            "y_train": y_train,
            "x_val": x_val,
            "y_val": y_val,
        }

    # Use a deterministic slot order and retain the original profile code as a
    # reference label. In IID mode this label is arbitrary because every slot
    # receives a mixture of Calm and Aggressive examples.
    slot_client_ids = selected_ids
    num_partitions = len(slot_client_ids)

    partition_records = []
    for partition_id, client_id in enumerate(slot_client_ids):
        profile_code = profile_code_by_id[client_id]
        values = normalized_by_id[client_id]
        save_partition(
            args.output_dir / "non_iid" / f"partition_{partition_id:03d}.npz",
            values["x_train"],
            values["y_train"],
            partition_id=partition_id,
            mode="non_iid",
            slot_profile_code=profile_code,
            source_client_id=client_id,
            num_calm_examples=(len(values["x_train"]) if profile_code == 0 else 0),
            num_aggressive_examples=(
                len(values["x_train"]) if profile_code == 1 else 0
            ),
        )
        partition_records.append(
            {
                "partition_id": partition_id,
                "source_client_id": client_id,
                "slot_profile_code": profile_code,
                "slot_profile": "Calm" if profile_code == 0 else "Aggressive",
                "non_iid_num_examples": int(len(values["x_train"])),
            }
        )

    # Build a controlled IID redistribution from exactly the same training
    # windows. Calm and Aggressive windows are shuffled separately and divided
    # across all partitions, so every IID client sees both behaviours.
    calm_x = np.concatenate(
        [normalized_by_id[cid]["x_train"] for cid in calm_ids], axis=0
    )
    calm_y = np.concatenate(
        [normalized_by_id[cid]["y_train"] for cid in calm_ids], axis=0
    )
    aggressive_x = np.concatenate(
        [normalized_by_id[cid]["x_train"] for cid in aggressive_ids], axis=0
    )
    aggressive_y = np.concatenate(
        [normalized_by_id[cid]["y_train"] for cid in aggressive_ids], axis=0
    )

    rng = np.random.default_rng(args.seed)
    calm_order = rng.permutation(len(calm_x))
    aggressive_order = rng.permutation(len(aggressive_x))
    calm_chunks = np.array_split(calm_order, num_partitions)
    aggressive_chunks = np.array_split(aggressive_order, num_partitions)

    iid_total = 0
    for partition_id, client_id in enumerate(slot_client_ids):
        calm_idx = calm_chunks[partition_id]
        aggressive_idx = aggressive_chunks[partition_id]
        x_iid = np.concatenate([calm_x[calm_idx], aggressive_x[aggressive_idx]])
        y_iid = np.concatenate([calm_y[calm_idx], aggressive_y[aggressive_idx]])
        local_order = rng.permutation(len(x_iid))
        x_iid = x_iid[local_order]
        y_iid = y_iid[local_order]
        iid_total += len(x_iid)

        save_partition(
            args.output_dir / "iid" / f"partition_{partition_id:03d}.npz",
            x_iid,
            y_iid,
            partition_id=partition_id,
            mode="iid",
            slot_profile_code=profile_code_by_id[client_id],
            source_client_id=client_id,
            num_calm_examples=len(calm_idx),
            num_aggressive_examples=len(aggressive_idx),
        )
        partition_records[partition_id]["iid_num_examples"] = int(len(x_iid))
        partition_records[partition_id]["iid_calm_examples"] = int(len(calm_idx))
        partition_records[partition_id]["iid_aggressive_examples"] = int(
            len(aggressive_idx)
        )

    non_iid_total = sum(
        len(normalized_by_id[cid]["x_train"]) for cid in selected_ids
    )
    if iid_total != non_iid_total:
        raise AssertionError("IID redistribution changed the training example count.")

    calm_val_x = np.concatenate(
        [normalized_by_id[cid]["x_val"] for cid in calm_ids], axis=0
    )
    calm_val_y = np.concatenate(
        [normalized_by_id[cid]["y_val"] for cid in calm_ids], axis=0
    )
    aggressive_val_x = np.concatenate(
        [normalized_by_id[cid]["x_val"] for cid in aggressive_ids], axis=0
    )
    aggressive_val_y = np.concatenate(
        [normalized_by_id[cid]["y_val"] for cid in aggressive_ids], axis=0
    )
    global_val_x = np.concatenate([calm_val_x, aggressive_val_x], axis=0)
    global_val_y = np.concatenate([calm_val_y, aggressive_val_y], axis=0)

    eval_dir = args.output_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(eval_dir / "global.npz", x=global_val_x, y=global_val_y)
    np.savez_compressed(eval_dir / "calm.npz", x=calm_val_x, y=calm_val_y)
    np.savez_compressed(
        eval_dir / "aggressive.npz", x=aggressive_val_x, y=aggressive_val_y
    )

    profiles.to_csv(args.output_dir / "profiles.csv", index=False)
    pd.DataFrame(partition_records).to_csv(
        args.output_dir / "partitions.csv", index=False
    )
    with (args.output_dir / "scaler.json").open("w", encoding="utf-8") as file:
        json.dump(scaler, file, indent=2)

    calm_rows = profiles[profiles["profile"] == "Calm"]
    aggressive_rows = profiles[profiles["profile"] == "Aggressive"]
    manifest = {
        "task": "T3.3 controlled IID versus non-IID",
        "seed": args.seed,
        "tail_fraction": args.tail_fraction,
        "t_past": T_PAST,
        "t_future": T_FUTURE,
        "train_split": TRAIN_SPLIT,
        "split_method": "chronological split of temporal windows",
        "normalization": "one global z-score scaler computed from selected training windows only",
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "num_eligible_clients": len(route_data),
        "num_partitions": num_partitions,
        "num_calm_clients": len(calm_ids),
        "num_aggressive_clients": len(aggressive_ids),
        "calm_client_ids": calm_ids,
        "aggressive_client_ids": aggressive_ids,
        "calm_mean_acceleration_range": [
            float(calm_rows["mean_acceleration"].min()),
            float(calm_rows["mean_acceleration"].max()),
        ],
        "aggressive_mean_acceleration_range": [
            float(aggressive_rows["mean_acceleration"].min()),
            float(aggressive_rows["mean_acceleration"].max()),
        ],
        "total_training_examples_non_iid": non_iid_total,
        "total_training_examples_iid": iid_total,
        "global_validation_examples": int(len(global_val_x)),
        "calm_validation_examples": int(len(calm_val_x)),
        "aggressive_validation_examples": int(len(aggressive_val_x)),
        "fairness_checks": {
            "same_training_examples": iid_total == non_iid_total,
            "same_number_of_partitions": True,
            "same_fixed_evaluation_sets": True,
            "same_scaler": True,
        },
        "skipped_clients": [
            {"client_id": client_id, "reason": reason}
            for client_id, reason in skipped
        ],
    }
    with (args.output_dir / "manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)

    print("============================================================")
    print("T3.3 controlled data preparation complete")
    print("============================================================")
    print(f"Eligible routes:              {len(route_data)}")
    print(f"Calm/Aggressive routes:       {len(calm_ids)} / {len(aggressive_ids)}")
    print(f"Virtual partitions:           {num_partitions}")
    print(f"Training examples per mode:   {non_iid_total}")
    print(f"Fixed validation examples:    {len(global_val_x)}")
    print(f"Global target mean/std:       {scaler['target_mean']:.6f} / {scaler['target_std']:.6f}")
    print(f"Output directory:             {args.output_dir}")
    print("Fairness checks:              PASSED")
    print("============================================================")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare normalized, matched IID and non-IID partitions for T3.3."
        )
    )
    parser.add_argument("--tail-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete an existing prepared directory before writing new data.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    prepare(parse_args())
