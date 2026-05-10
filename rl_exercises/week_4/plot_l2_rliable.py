"""
Level 2: RLiable plotting for robust statistics across seeds.
Computes IQM, mean, median with confidence intervals.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from rliable import metrics

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "week_4" / "l2_dqn"


def discover_seeds() -> list[int]:
    """Discover available seed CSVs in the results directory."""
    seeds = []
    for path in RESULTS_DIR.glob("seed_*.csv"):
        try:
            seeds.append(int(path.stem.split("_")[1]))
        except (IndexError, ValueError):
            continue
    return sorted(seeds)


def load_seed_data() -> tuple[np.ndarray, np.ndarray]:
    """Load and align seed data to common frame grid."""
    seeds = discover_seeds()
    if not seeds:
        raise FileNotFoundError(f"No seed_*.csv files found in {RESULTS_DIR}")

    seed_frames_list = []
    seed_rewards_list = []

    for seed in seeds:
        df = pd.read_csv(RESULTS_DIR / f"seed_{seed}.csv")
        frames = df["frame"].to_numpy(dtype=float)
        rewards = df["reward"].to_numpy(dtype=float)
        seed_frames_list.append(frames)
        seed_rewards_list.append(rewards)

    # Create common frame grid from all data
    all_frames = np.concatenate(seed_frames_list)
    min_frame = all_frames.min()
    max_frame = all_frames.max()
    frame_grid = np.linspace(min_frame, max_frame, 50)

    # Interpolate all seeds to common grid
    aligned_rewards = []
    for frames, rewards in zip(seed_frames_list, seed_rewards_list):
        # Handle edge cases: use first/last value for extrapolation
        aligned = np.interp(
            frame_grid, frames, rewards, left=rewards[0], right=rewards[-1]
        )
        aligned_rewards.append(aligned)

    # Stack to shape (n_seeds, n_frames)
    reward_matrix = np.stack(aligned_rewards, axis=0)
    return frame_grid, reward_matrix


def plot_rliable_curves() -> None:
    """Generate RLiable plots with IQM, mean, median."""
    seeds = discover_seeds()
    n_seeds = len(seeds)
    frame_grid, reward_matrix = load_seed_data()

    print(f"Loaded {n_seeds} seeds with shape {reward_matrix.shape}")

    # Define aggregation functions
    iqm = lambda scores: np.array(
        [metrics.aggregate_iqm(scores[:, i]) for i in range(scores.shape[1])]
    )
    mean_fn = lambda scores: np.array(
        [scores[:, i].mean() for i in range(scores.shape[1])]
    )
    median_fn = lambda scores: np.array(
        [np.median(scores[:, i]) for i in range(scores.shape[1])]
    )

    # Compute aggregations
    print("Computing statistics...")
    iqm_agg = iqm(reward_matrix)
    mean_agg = mean_fn(reward_matrix)
    median_agg = median_fn(reward_matrix)

    # Use bootstrap to compute 95% CI (matching RLiable style)
    REPS = 2000
    rng = np.random.default_rng(0)
    n_frames = reward_matrix.shape[1]

    def bootstrap_metric(metric_fn):
        boots = np.zeros((REPS, n_frames), dtype=float)
        for r in range(REPS):
            idx = rng.integers(0, n_seeds, size=n_seeds)
            sample = reward_matrix[idx, :]
            boots[r] = metric_fn(sample)
        lower = np.percentile(boots, 2.5, axis=0)
        upper = np.percentile(boots, 97.5, axis=0)
        return lower, upper, boots

    print(f"Running bootstrap with {REPS} reps for CI estimation...")
    iqm_lower, iqm_upper, _ = bootstrap_metric(iqm)
    mean_lower, mean_upper, _ = bootstrap_metric(mean_fn)
    median_lower, median_upper, _ = bootstrap_metric(median_fn)

    # Plot 1: IQM with bootstrap confidence band
    fig1, ax = plt.subplots(figsize=(10, 6))
    ax.plot(frame_grid + 1, iqm_agg, label="IQM", linewidth=2.5, color="#2a9d8f")
    ax.fill_between(
        frame_grid + 1,
        iqm_lower,
        iqm_upper,
        alpha=0.25,
        color="#2a9d8f",
        label="95% CI (bootstrap)",
    )
    ax.set_xlabel("Frames")
    ax.set_ylabel("IQM Reward")
    ax.set_title(f"Level 2: IQM Reward Across {n_seeds} Seeds (bootstrap CI)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig1.savefig(RESULTS_DIR / "l2_iqm_curve.png", dpi=180)
    plt.close(fig1)
    print(f"Saved IQM plot to {RESULTS_DIR / 'l2_iqm_curve.png'}")

    # Plot 2: Mean vs Median comparison with bootstrap bands
    fig2, ax = plt.subplots(figsize=(10, 6))
    ax.plot(frame_grid + 1, mean_agg, label="Mean", linewidth=2, color="#1f77b4")
    ax.fill_between(frame_grid + 1, mean_lower, mean_upper, alpha=0.2, color="#1f77b4")
    ax.plot(frame_grid + 1, median_agg, label="Median", linewidth=2, color="#ff7f0e")
    ax.fill_between(
        frame_grid + 1, median_lower, median_upper, alpha=0.2, color="#ff7f0e"
    )
    ax.set_xlabel("Frames")
    ax.set_ylabel("Reward")
    ax.set_title("Level 2: Mean vs Median with 95% Bootstrap CI")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig2.savefig(RESULTS_DIR / "l2_mean_vs_median.png", dpi=180)
    plt.close(fig2)
    print(f"Saved mean/median plot to {RESULTS_DIR / 'l2_mean_vs_median.png'}")

    # Plot 3: Individual seed traces
    frame_grid_full, reward_matrix_full = load_seed_data()
    fig3, ax = plt.subplots(figsize=(10, 6))
    for i, seed in enumerate(seeds):
        ax.plot(
            frame_grid_full + 1,
            reward_matrix_full[i],
            label=f"seed={seed}",
            alpha=0.6,
            linewidth=1.5,
        )
    ax.set_xlabel("Frames")
    ax.set_ylabel("Reward")
    ax.set_title("Level 2: Individual Seed Trajectories")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig3.savefig(RESULTS_DIR / "l2_individual_seeds.png", dpi=180)
    plt.close(fig3)
    print(f"Saved individual seeds plot to {RESULTS_DIR / 'l2_individual_seeds.png'}")

    # Summary statistics (using bootstrap CI)
    print(f"\n=== Summary Statistics ({n_seeds} seeds, bootstrap CI) ===")
    print(
        f"Final IQM: {iqm_agg[-1]:.2f} (95% CI: {iqm_lower[-1]:.2f} - {iqm_upper[-1]:.2f})"
    )
    print(
        f"Final Mean: {mean_agg[-1]:.2f} (95% CI: {mean_lower[-1]:.2f} - {mean_upper[-1]:.2f})"
    )
    print(
        f"Final Median: {median_agg[-1]:.2f} (95% CI: {median_lower[-1]:.2f} - {median_upper[-1]:.2f})"
    )

    # Save summary to CSV with bootstrap intervals
    summary = pd.DataFrame(
        {
            "frame": frame_grid + 1,
            "iqm": iqm_agg,
            "iqm_lower": iqm_lower,
            "iqm_upper": iqm_upper,
            "mean": mean_agg,
            "mean_lower": mean_lower,
            "mean_upper": mean_upper,
            "median": median_agg,
            "median_lower": median_lower,
            "median_upper": median_upper,
        }
    )
    summary.to_csv(RESULTS_DIR / "l2_rliable_summary.csv", index=False)
    print(f"Saved summary CSV to {RESULTS_DIR / 'l2_rliable_summary.csv'}")


if __name__ == "__main__":
    plot_rliable_curves()
