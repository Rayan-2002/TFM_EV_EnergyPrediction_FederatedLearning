from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt


FLOWER_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = FLOWER_ROOT / "outputs"
DEFAULT_CENTRALIZED_HISTORY = (
    DEFAULT_OUTPUT_DIR / "t3_4_controlled_centralized_history.json"
)
DEFAULT_FEDERATED_HISTORY = (
    DEFAULT_OUTPUT_DIR
    / "t3_3_controlled_non_iid_82_clients_20_rounds.json"
)


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def extract_federated_rmse(history: dict) -> List[Tuple[int, float]]:
    metrics = history.get("metrics_centralized", {})
    raw_series = metrics.get("global_rmse_w")
    if raw_series is None:
        raw_series = [
            [record["round"], record["global_rmse_w"]]
            for record in history.get("evaluation_records", [])
        ]
    if not raw_series:
        raise KeyError(
            "Federated history does not contain "
            "metrics_centralized.global_rmse_w or evaluation_records."
        )

    series = sorted((int(round_number), float(value)) for round_number, value in raw_series)
    rounds = [round_number for round_number, _ in series]
    if len(rounds) != len(set(rounds)):
        raise ValueError("Federated RMSE series contains duplicate rounds.")
    return series


def extract_constant_example_count(history: dict) -> int | None:
    metrics = history.get("metrics_centralized", {})
    values = metrics.get("global_examples", [])
    if not values:
        records = history.get("evaluation_records", [])
        values = [[record["round"], record["global_examples"]] for record in records]
    if not values:
        return None
    counts = {int(float(value)) for _, value in values}
    if len(counts) != 1:
        raise ValueError(
            f"Federated evaluation set changed across rounds: {sorted(counts)}"
        )
    return counts.pop()


def choose_centralized_baseline(history: dict, selection: str) -> Tuple[float, int]:
    if selection not in history:
        raise KeyError(f"Centralized history has no '{selection}' result.")
    record = history[selection]
    return float(record["global_rmse_w"]), int(record["epoch"])


def percent_change(initial: float, final: float) -> float:
    if initial == 0:
        return math.nan
    return (initial - final) / initial * 100.0


def save_main_plot(
    *,
    rounds: Sequence[int],
    federated_rmse: Sequence[float],
    centralized_rmse: float,
    centralized_epoch: int,
    federated_label: str,
    png_path: Path,
    pdf_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(10, 6))
    axis.plot(
        rounds,
        federated_rmse,
        marker="o",
        markersize=4,
        linewidth=2,
        label=federated_label,
    )
    axis.axhline(
        y=centralized_rmse,
        linestyle="--",
        linewidth=2,
        label=f"Centralized baseline (best epoch {centralized_epoch})",
    )

    best_index = min(range(len(federated_rmse)), key=federated_rmse.__getitem__)
    best_round = rounds[best_index]
    best_value = federated_rmse[best_index]
    final_round = rounds[-1]
    final_value = federated_rmse[-1]

    axis.scatter([best_round], [best_value], zorder=3)
    axis.annotate(
        f"Best federated: {best_value:,.1f} W\nRound {best_round}",
        xy=(best_round, best_value),
        xytext=(8, 14),
        textcoords="offset points",
    )
    axis.annotate(
        f"Final: {final_value:,.1f} W",
        xy=(final_round, final_value),
        xytext=(-92, -28),
        textcoords="offset points",
    )

    axis.set_title("Centralized Baseline vs Federated Validation RMSE")
    axis.set_xlabel("Federated round")
    axis.set_ylabel("Validation RMSE (W)")
    axis.set_xticks(rounds)
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(png_path, dpi=300, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)


def save_optional_three_line_plot(
    *,
    primary_series: Sequence[Tuple[int, float]],
    secondary_series: Sequence[Tuple[int, float]],
    centralized_rmse: float,
    centralized_epoch: int,
    primary_label: str,
    secondary_label: str,
    output_path: Path,
) -> None:
    primary_rounds = [round_number for round_number, _ in primary_series]
    secondary_rounds = [round_number for round_number, _ in secondary_series]
    if primary_rounds != secondary_rounds:
        raise ValueError(
            "Primary and secondary federated histories do not use identical rounds."
        )

    figure, axis = plt.subplots(figsize=(10, 6))
    axis.plot(
        primary_rounds,
        [value for _, value in primary_series],
        marker="o",
        markersize=4,
        linewidth=2,
        label=primary_label,
    )
    axis.plot(
        secondary_rounds,
        [value for _, value in secondary_series],
        marker="s",
        markersize=4,
        linewidth=2,
        label=secondary_label,
    )
    axis.axhline(
        y=centralized_rmse,
        linestyle="--",
        linewidth=2,
        label=f"Centralized baseline (best epoch {centralized_epoch})",
    )
    axis.set_title("Centralized, Federated IID, and Federated Non-IID RMSE")
    axis.set_xlabel("Federated round")
    axis.set_ylabel("Validation RMSE (W)")
    axis.set_xticks(primary_rounds)
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def write_comparison_csv(
    path: Path,
    series: Sequence[Tuple[int, float]],
    centralized_rmse: float,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "round",
                "federated_rmse_w",
                "centralized_baseline_rmse_w",
                "federated_minus_centralized_w",
            ],
        )
        writer.writeheader()
        for round_number, federated_value in series:
            writer.writerow(
                {
                    "round": round_number,
                    "federated_rmse_w": federated_value,
                    "centralized_baseline_rmse_w": centralized_rmse,
                    "federated_minus_centralized_w": (
                        federated_value - centralized_rmse
                    ),
                }
            )


def run(args: argparse.Namespace) -> dict:
    centralized_history = load_json(args.centralized_history)
    federated_history = load_json(args.federated_history)

    centralized_rmse, centralized_epoch = choose_centralized_baseline(
        centralized_history,
        args.centralized_selection,
    )
    federated_series = extract_federated_rmse(federated_history)
    rounds = [round_number for round_number, _ in federated_series]
    federated_rmse = [value for _, value in federated_series]

    centralized_examples = int(
        centralized_history["evaluation_data"]["global_examples"]
    )
    federated_examples = extract_constant_example_count(federated_history)
    if federated_examples is not None and federated_examples != centralized_examples:
        raise ValueError(
            "Unfair evaluation comparison: centralized uses "
            f"{centralized_examples} examples but federated uses "
            f"{federated_examples}."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    png_path = args.output_dir / "t3_4_centralized_vs_federated_rmse.png"
    pdf_path = args.output_dir / "t3_4_centralized_vs_federated_rmse.pdf"
    csv_path = args.output_dir / "t3_4_rmse_per_round.csv"
    summary_path = args.output_dir / "t3_4_scientific_summary.json"

    save_main_plot(
        rounds=rounds,
        federated_rmse=federated_rmse,
        centralized_rmse=centralized_rmse,
        centralized_epoch=centralized_epoch,
        federated_label=args.federated_label,
        png_path=png_path,
        pdf_path=pdf_path,
    )
    write_comparison_csv(csv_path, federated_series, centralized_rmse)

    best_position = min(range(len(federated_rmse)), key=federated_rmse.__getitem__)
    best_round = rounds[best_position]
    best_federated = federated_rmse[best_position]
    initial_federated = federated_rmse[0]
    final_federated = federated_rmse[-1]
    final_gap = final_federated - centralized_rmse
    best_gap = best_federated - centralized_rmse

    summary = {
        "task": "T3.4 Scientific Synthesis",
        "comparison_design": {
            "centralized_history": str(args.centralized_history),
            "federated_history": str(args.federated_history),
            "centralized_selection": args.centralized_selection,
            "same_fixed_evaluation_examples": True,
            "evaluation_examples": centralized_examples,
            "rmse_unit": "watts",
        },
        "centralized_baseline": {
            "rmse_w": centralized_rmse,
            "selected_epoch": centralized_epoch,
        },
        "federated": {
            "label": args.federated_label,
            "initial_round": rounds[0],
            "initial_rmse_w": initial_federated,
            "final_round": rounds[-1],
            "final_rmse_w": final_federated,
            "best_round": best_round,
            "best_rmse_w": best_federated,
            "improvement_w": initial_federated - final_federated,
            "improvement_percent": percent_change(
                initial_federated, final_federated
            ),
        },
        "comparison": {
            "final_federated_minus_centralized_w": final_gap,
            "final_relative_gap_percent": (
                final_gap / centralized_rmse * 100.0
                if centralized_rmse != 0
                else math.nan
            ),
            "best_federated_minus_centralized_w": best_gap,
            "best_federated_reached_or_beat_centralized": (
                best_federated <= centralized_rmse
            ),
        },
        "outputs": {
            "plot_png": str(png_path),
            "plot_pdf": str(pdf_path),
            "comparison_csv": str(csv_path),
        },
    }

    if args.secondary_federated_history is not None:
        secondary_history = load_json(args.secondary_federated_history)
        secondary_series = extract_federated_rmse(secondary_history)
        three_line_path = (
            args.output_dir / "t3_4_centralized_iid_non_iid_rmse.png"
        )
        save_optional_three_line_plot(
            primary_series=federated_series,
            secondary_series=secondary_series,
            centralized_rmse=centralized_rmse,
            centralized_epoch=centralized_epoch,
            primary_label=args.federated_label,
            secondary_label=args.secondary_federated_label,
            output_path=three_line_path,
        )
        summary["outputs"]["three_line_plot_png"] = str(three_line_path)

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print(json.dumps(summary, indent=2))
    print()
    print(f"PNG:     {png_path}")
    print(f"PDF:     {pdf_path}")
    print(f"CSV:     {csv_path}")
    print(f"Summary: {summary_path}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot the controlled centralized validation RMSE as a flat line "
            "against the federated global RMSE curve stored in Flower history."
        )
    )
    parser.add_argument(
        "--centralized-history",
        type=Path,
        default=DEFAULT_CENTRALIZED_HISTORY,
    )
    parser.add_argument(
        "--federated-history",
        type=Path,
        default=DEFAULT_FEDERATED_HISTORY,
    )
    parser.add_argument(
        "--centralized-selection",
        choices=["best", "final"],
        default="best",
    )
    parser.add_argument(
        "--federated-label",
        default="Federated FedAvg (controlled non-IID)",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--secondary-federated-history",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--secondary-federated-label",
        default="Federated FedAvg (controlled IID)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
