"""Plot week 5 REINFORCE variation results.

Reads the JSON summary produced by run_l2_variations.py and writes a compact
set of comparison plots to results/week_5/.
"""

from __future__ import annotations

from typing import Any, Dict, List

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

DEFAULT_RESULTS = Path(__file__).resolve().parents[2] / "results" / "week_5"
DEFAULT_JSON = DEFAULT_RESULTS / "l2_variations.json"


def load_results(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_series(run: Dict[str, Any]) -> tuple[List[int], List[float], List[float]]:
    episodes = [int(item["episode"]) for item in run["history"]]
    means = [float(item["mean_return"]) for item in run["history"]]
    stds = [float(item["std_return"]) for item in run["history"]]
    return episodes, means, stds


def plot_learning_curves(results: Dict[str, Any], output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {
        "cartpole_full": "#2a9d8f",
        "cartpole_trunc_20": "#e76f51",
        "cartpole_trunc_50": "#f4a261",
        "lunarlander_same_hparams": "#264653",
    }

    for key, run in results.items():
        episodes, means, stds = extract_series(run)
        label = key.replace("cartpole_", "CartPole ").replace(
            "lunarlander_same_hparams", "LunarLander same hparams"
        )
        color = colors.get(key)
        ax.plot(episodes, means, label=label, linewidth=2.2, color=color)
        ax.fill_between(
            episodes,
            [m - s for m, s in zip(means, stds)],
            [m + s for m, s in zip(means, stds)],
            color=color,
            alpha=0.15,
        )

    ax.axhline(
        475.0,
        linestyle="--",
        linewidth=1.2,
        color="#666666",
        label="CartPole solve threshold",
    )
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Evaluation mean return")
    ax.set_title("Week 5 REINFORCE: trajectory length and transfer comparison")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    plt.tight_layout()

    output_path = output_dir / "l2_learning_curves.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_sample_efficiency(results: Dict[str, Any], output_dir: Path) -> Path:
    labels = []
    solved_steps = []

    for key in [
        "cartpole_full",
        "cartpole_trunc_20",
        "cartpole_trunc_50",
        "lunarlander_same_hparams",
    ]:
        run = results[key]
        labels.append(
            key.replace("cartpole_", "CartPole ").replace(
                "lunarlander_same_hparams", "LunarLander same hparams"
            )
        )
        solved_steps.append(run["solved_step"])

    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = list(range(len(labels)))
    values = [step if step is not None else 0 for step in solved_steps]
    bars = ax.bar(x, values, color=["#2a9d8f", "#e76f51", "#f4a261", "#264653"])

    for idx, step in enumerate(solved_steps):
        if step is None:
            ax.text(
                idx, 5, "not solved", ha="center", va="bottom", fontsize=9, rotation=90
            )
        else:
            ax.text(
                idx,
                step + max(values) * 0.02,
                f"{step}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Environment steps to solve")
    ax.set_title("Week 5 REINFORCE sample complexity summary")
    ax.set_ylim(0, max(values) * 1.15 if max(values) > 0 else 1)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    output_path = output_dir / "l2_sample_complexity.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def print_summary(results: Dict[str, Any]) -> None:
    print("\n=== Week 5 REINFORCE variation summary ===")
    for key, run in results.items():
        last = run["history"][-1]
        print(
            f"{key}: final mean return {last['mean_return']:.1f}, "
            f"solved_step={run['solved_step']}, effective_env={run['effective_eval_env']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot week 5 REINFORCE variations.")
    parser.add_argument(
        "--input", default=str(DEFAULT_JSON), help="Input JSON summary file."
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_RESULTS),
        help="Directory for generated plots.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = load_results(input_path)
    plot1 = plot_learning_curves(results, output_dir)
    plot2 = plot_sample_efficiency(results, output_dir)
    print_summary(results)
    print(f"Saved {plot1}")
    print(f"Saved {plot2}")


if __name__ == "__main__":
    main()
