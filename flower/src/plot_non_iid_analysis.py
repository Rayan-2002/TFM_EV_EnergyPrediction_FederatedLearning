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


def get_series(data: dict, section: str, metric: str) -> Tuple[List[int], List[float]]:
    values = data.get(section, {}).get(metric, [])
    rounds = [int(item[0]) for item in values]
    metrics = [float(item[1]) for item in values]
    return rounds, metrics


def summarize(values: List[float]) -> Dict[str, float]:
    array = np.asarray(values, dtype=float)

    if array.size == 0:
        return {}

    changes = np.diff(array)

    return {
        "first": float(array[0]),
        "final": float(array[-1]),
        "best": float(array.min()),
        "best_round_index_1_based": int(array.argmin() + 1),
        "mean": float(array.mean()),
        "standard_deviation": float(array.std(ddof=0)),
        "mean_absolute_round_change": (
            float(np.abs(changes).mean()) if changes.size else 0.0
        ),
        "final_minus_first": float(array[-1] - array[0]),
    }


def main(args: argparse.Namespace) -> None:
    baseline = load_json(args.baseline)
    non_iid = load_json(args.non_iid)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    baseline_rounds, baseline_val_rmse = get_series(
        baseline,
        "metrics_distributed",
        "val_rmse",
    )
    non_iid_rounds, non_iid_val_rmse = get_series(
        non_iid,
        "metrics_distributed",
        "val_rmse",
    )
    calm_rounds, calm_val_rmse = get_series(
        non_iid,
        "metrics_distributed",
        "val_rmse_calm",
    )
    aggressive_rounds, aggressive_val_rmse = get_series(
        non_iid,
        "metrics_distributed",
        "val_rmse_aggressive",
    )
    gap_rounds, profile_gap = get_series(
        non_iid,
        "metrics_distributed",
        "val_rmse_profile_gap",
    )

    if not non_iid_val_rmse:
        raise ValueError("The non-IID history does not contain val_rmse.")

    plt.figure(figsize=(10, 6))
    if baseline_val_rmse:
        plt.plot(
            baseline_rounds,
            baseline_val_rmse,
            marker="o",
            label="T3.2 baseline",
        )
    plt.plot(
        non_iid_rounds,
        non_iid_val_rmse,
        marker="o",
        label="T3.3 non-IID extremes",
    )
    plt.xlabel("Federated round")
    plt.ylabel("Validation RMSE (W)")
    plt.title("Baseline versus non-IID global convergence")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    global_plot = args.output_dir / "t3_3_global_convergence_comparison.png"
    plt.savefig(global_plot, dpi=200)
    plt.close()

    if calm_val_rmse and aggressive_val_rmse:
        plt.figure(figsize=(10, 6))
        plt.plot(calm_rounds, calm_val_rmse, marker="o", label="Calm")
        plt.plot(
            aggressive_rounds,
            aggressive_val_rmse,
            marker="o",
            label="Aggressive",
        )
        plt.xlabel("Federated round")
        plt.ylabel("Validation RMSE (W)")
        plt.title("Non-IID performance by acceleration profile")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        profile_plot = args.output_dir / "t3_3_profile_rmse.png"
        plt.savefig(profile_plot, dpi=200)
        plt.close()
    else:
        profile_plot = None

    if profile_gap:
        plt.figure(figsize=(10, 6))
        plt.plot(gap_rounds, profile_gap, marker="o")
        plt.xlabel("Federated round")
        plt.ylabel("Absolute Calm/Aggressive RMSE gap (W)")
        plt.title("Non-IID profile performance gap")
        plt.grid(True)
        plt.tight_layout()
        gap_plot = args.output_dir / "t3_3_profile_gap.png"
        plt.savefig(gap_plot, dpi=200)
        plt.close()
    else:
        gap_plot = None

    summary = {
        "baseline_val_rmse": summarize(baseline_val_rmse),
        "non_iid_global_val_rmse": summarize(non_iid_val_rmse),
        "non_iid_calm_val_rmse": summarize(calm_val_rmse),
        "non_iid_aggressive_val_rmse": summarize(aggressive_val_rmse),
        "non_iid_profile_gap": summarize(profile_gap),
    }

    summary_path = args.output_dir / "t3_3_analysis_summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print(json.dumps(summary, indent=2))
    print()
    print(f"Global convergence plot: {global_plot}")
    if profile_plot is not None:
        print(f"Profile RMSE plot:        {profile_plot}")
    if gap_plot is not None:
        print(f"Profile gap plot:         {gap_plot}")
    print(f"Summary JSON:             {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare T3.2 baseline and T3.3 non-IID convergence."
    )
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--non-iid", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
