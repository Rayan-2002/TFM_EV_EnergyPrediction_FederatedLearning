from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def get_metric(data: dict, name: str) -> Tuple[List[int], List[float]]:
    values = data.get("metrics_centralized", {}).get(name, [])
    return (
        [int(item[0]) for item in values],
        [float(item[1]) for item in values],
    )


def summarize(values: List[float]) -> Dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {}
    changes = np.diff(array)
    return {
        "first": float(array[0]),
        "final": float(array[-1]),
        "best": float(array.min()),
        "best_index": int(array.argmin()),
        "mean": float(array.mean()),
        "standard_deviation": float(array.std(ddof=0)),
        "final_minus_first": float(array[-1] - array[0]),
        "mean_absolute_round_change": (
            float(np.abs(changes).mean()) if changes.size else 0.0
        ),
    }


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= 1e-15:
        return 0.0
    return float(np.dot(a, b) / denominator)


def off_diagonal_mean(matrix: np.ndarray) -> float:
    if len(matrix) < 2:
        return 0.0
    mask = ~np.eye(len(matrix), dtype=bool)
    return float(matrix[mask].mean())


def analyze_round(round_dir: Path) -> dict:
    records = []
    for path in sorted(round_dir.glob("partition_*.npz")):
        values = np.load(path, allow_pickle=False)
        records.append(
            {
                "update": values["update"].astype(np.float64),
                "num_examples": int(values["num_examples"]),
                "group_code": int(values["group_code"]),
                "partition_id": int(values["partition_id"]),
            }
        )

    if len(records) < 2:
        raise ValueError(f"Not enough update files in {round_dir}")

    updates = np.stack([record["update"] for record in records])
    weights = np.asarray([record["num_examples"] for record in records], dtype=float)
    groups = np.asarray([record["group_code"] for record in records], dtype=int)
    norms = np.linalg.norm(updates, axis=1)
    safe_norms = np.where(norms > 1e-15, norms, 1.0)
    unit_updates = updates / safe_norms[:, None]
    cosine_matrix = np.clip(unit_updates @ unit_updates.T, -1.0, 1.0)

    upper = cosine_matrix[np.triu_indices(len(records), k=1)]
    mean_pairwise = float(upper.mean()) if upper.size else 0.0
    negative_fraction = float((upper < 0.0).mean()) if upper.size else 0.0

    group_0 = np.where(groups == 0)[0]
    group_1 = np.where(groups == 1)[0]
    if len(group_0) == 0 or len(group_1) == 0:
        raise ValueError(f"Both reference groups are required in {round_dir}")

    between = cosine_matrix[np.ix_(group_0, group_1)]
    within_0 = cosine_matrix[np.ix_(group_0, group_0)]
    within_1 = cosine_matrix[np.ix_(group_1, group_1)]

    weighted_mean_all = np.average(updates, axis=0, weights=weights)
    weighted_mean_norm = float(np.linalg.norm(weighted_mean_all))
    weighted_mean_individual_norm = float(np.average(norms, weights=weights))
    cancellation_ratio = (
        weighted_mean_norm / weighted_mean_individual_norm
        if weighted_mean_individual_norm > 1e-15
        else 0.0
    )

    mean_group_0 = np.average(
        updates[group_0], axis=0, weights=weights[group_0]
    )
    mean_group_1 = np.average(
        updates[group_1], axis=0, weights=weights[group_1]
    )

    return {
        "round": int(round_dir.name.removeprefix("round_")),
        "num_clients": len(records),
        "mean_pairwise_cosine": mean_pairwise,
        "negative_pair_fraction": negative_fraction,
        "between_group_pairwise_cosine": float(between.mean()),
        "within_group_0_cosine": off_diagonal_mean(within_0),
        "within_group_1_cosine": off_diagonal_mean(within_1),
        "group_mean_update_cosine": cosine(mean_group_0, mean_group_1),
        "mean_update_norm": float(norms.mean()),
        "mean_update_norm_group_0": float(norms[group_0].mean()),
        "mean_update_norm_group_1": float(norms[group_1].mean()),
        "cancellation_ratio": float(cancellation_ratio),
    }


def analyze_updates(directory: Path) -> List[dict]:
    round_dirs = sorted(directory.glob("round_*"))
    if not round_dirs:
        raise FileNotFoundError(f"No round directories found in {directory}")
    return [analyze_round(path) for path in round_dirs]


def average_field(records: List[dict], field: str) -> float:
    return float(np.mean([record[field] for record in records]))


def plot_series(
    output: Path,
    x_a: List[int],
    y_a: List[float],
    label_a: str,
    x_b: List[int],
    y_b: List[float],
    label_b: str,
    ylabel: str,
    title: str,
    horizontal_zero: bool = False,
) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(x_a, y_a, marker="o", label=label_a)
    plt.plot(x_b, y_b, marker="o", label=label_b)
    if horizontal_zero:
        plt.axhline(0.0, linewidth=1)
    plt.xlabel("Federated round")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=200)
    plt.close()


def main(args: argparse.Namespace) -> None:
    iid = load_json(args.iid_history)
    non_iid = load_json(args.non_iid_history)
    iid_updates = analyze_updates(args.iid_updates)
    non_iid_updates = analyze_updates(args.non_iid_updates)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    iid_rounds, iid_global = get_metric(iid, "global_rmse_w")
    non_iid_rounds, non_iid_global = get_metric(non_iid, "global_rmse_w")
    _, iid_calm = get_metric(iid, "calm_rmse_w")
    _, iid_aggressive = get_metric(iid, "aggressive_rmse_w")
    _, non_iid_calm = get_metric(non_iid, "calm_rmse_w")
    _, non_iid_aggressive = get_metric(non_iid, "aggressive_rmse_w")

    plot_series(
        args.output_dir / "t3_3_controlled_global_rmse.png",
        iid_rounds,
        iid_global,
        "Controlled IID",
        non_iid_rounds,
        non_iid_global,
        "Controlled non-IID",
        "Fixed validation RMSE (W)",
        "Fair IID versus non-IID convergence",
    )

    plt.figure(figsize=(10, 6))
    plt.plot(iid_rounds, iid_calm, marker="o", label="IID evaluated on Calm")
    plt.plot(
        iid_rounds,
        iid_aggressive,
        marker="o",
        label="IID evaluated on Aggressive",
    )
    plt.plot(
        non_iid_rounds,
        non_iid_calm,
        marker="o",
        label="Non-IID evaluated on Calm",
    )
    plt.plot(
        non_iid_rounds,
        non_iid_aggressive,
        marker="o",
        label="Non-IID evaluated on Aggressive",
    )
    plt.xlabel("Federated round")
    plt.ylabel("Fixed validation RMSE (W)")
    plt.title("Profile performance under matched evaluation")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output_dir / "t3_3_controlled_profile_rmse.png", dpi=200)
    plt.close()

    update_rounds_iid = [record["round"] for record in iid_updates]
    update_rounds_non_iid = [record["round"] for record in non_iid_updates]

    plot_series(
        args.output_dir / "t3_3_group_mean_update_cosine.png",
        update_rounds_iid,
        [record["group_mean_update_cosine"] for record in iid_updates],
        "IID reference groups",
        update_rounds_non_iid,
        [record["group_mean_update_cosine"] for record in non_iid_updates],
        "Non-IID Calm vs Aggressive",
        "Cosine similarity",
        "Alignment of mean client updates",
        horizontal_zero=True,
    )

    plot_series(
        args.output_dir / "t3_3_negative_update_fraction.png",
        update_rounds_iid,
        [record["negative_pair_fraction"] for record in iid_updates],
        "IID",
        update_rounds_non_iid,
        [record["negative_pair_fraction"] for record in non_iid_updates],
        "Non-IID",
        "Fraction of client-update pairs with cosine < 0",
        "Opposing client-update directions",
    )

    plot_series(
        args.output_dir / "t3_3_update_cancellation_ratio.png",
        update_rounds_iid,
        [record["cancellation_ratio"] for record in iid_updates],
        "IID",
        update_rounds_non_iid,
        [record["cancellation_ratio"] for record in non_iid_updates],
        "Non-IID",
        "Cancellation ratio",
        "How strongly local updates cancel during FedAvg",
    )

    iid_divergence = {
        field: average_field(iid_updates, field)
        for field in [
            "mean_pairwise_cosine",
            "negative_pair_fraction",
            "between_group_pairwise_cosine",
            "group_mean_update_cosine",
            "cancellation_ratio",
            "mean_update_norm",
        ]
    }
    non_iid_divergence = {
        field: average_field(non_iid_updates, field)
        for field in [
            "mean_pairwise_cosine",
            "negative_pair_fraction",
            "between_group_pairwise_cosine",
            "group_mean_update_cosine",
            "cancellation_ratio",
            "mean_update_norm",
        ]
    }

    evidence = {
        "non_iid_group_means_less_aligned": (
            non_iid_divergence["group_mean_update_cosine"]
            < iid_divergence["group_mean_update_cosine"]
        ),
        "non_iid_more_negative_pairs": (
            non_iid_divergence["negative_pair_fraction"]
            > iid_divergence["negative_pair_fraction"]
        ),
        "non_iid_more_update_cancellation": (
            non_iid_divergence["cancellation_ratio"]
            < iid_divergence["cancellation_ratio"]
        ),
        "non_iid_final_rmse_worse": (
            bool(iid_global and non_iid_global)
            and non_iid_global[-1] > iid_global[-1]
        ),
    }
    evidence["divergence_indicators_supported"] = int(
        sum(bool(value) for value in evidence.values())
    )

    summary = {
        "comparison_design": {
            "same_raw_training_windows": True,
            "same_global_scaler": True,
            "same_model_initialization": True,
            "same_number_of_clients": True,
            "same_clients_each_round": True,
            "same_fixed_evaluation_sets": True,
            "only_partitioning_changes": "IID mixed partitions versus route/profile non-IID partitions",
        },
        "iid_global_rmse_w": summarize(iid_global),
        "non_iid_global_rmse_w": summarize(non_iid_global),
        "iid_update_divergence": iid_divergence,
        "non_iid_update_divergence": non_iid_divergence,
        "evidence": evidence,
        "iid_round_details": iid_updates,
        "non_iid_round_details": non_iid_updates,
        "interpretation_note": (
            "Model-update vectors are empirical proxies for local optimization "
            "directions. Lower cosine similarity, more negative pairs, and a "
            "lower cancellation ratio indicate stronger client drift and update "
            "conflict under FedAvg."
        ),
    }

    summary_path = args.output_dir / "t3_3_controlled_analysis_summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print(json.dumps(summary, indent=2))
    print()
    print(f"Summary: {summary_path}")
    print(f"Plots:   {args.output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyse fair T3.3 IID/non-IID convergence and update divergence."
    )
    parser.add_argument("--iid-history", type=Path, required=True)
    parser.add_argument("--non-iid-history", type=Path, required=True)
    parser.add_argument("--iid-updates", type=Path, required=True)
    parser.add_argument("--non-iid-updates", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
