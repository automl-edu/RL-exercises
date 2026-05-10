from __future__ import annotations

from typing import Any

from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from rl_exercises.week_4.dqn import DQNAgent, set_seed


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    hidden_dim: int
    depth: int
    buffer_capacity: int
    batch_size: int
    color: str


RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "week_4" / "l1_dqn"
SEEDS = [0, 1, 2, 3, 4]
NUM_FRAMES = 2500
SMOOTHING_WINDOW = 10
GRID_POINTS = 120
LEARNING_RATE = 1e-3
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_FINAL = 0.01
EPSILON_DECAY = 500
TARGET_UPDATE_FREQ = 1000

CONFIGS = [
    ExperimentConfig(
        name="small",
        hidden_dim=32,
        depth=1,
        buffer_capacity=2000,
        batch_size=32,
        color="#d1495b",
    ),
    ExperimentConfig(
        name="baseline",
        hidden_dim=64,
        depth=2,
        buffer_capacity=10000,
        batch_size=32,
        color="#30638e",
    ),
    ExperimentConfig(
        name="large",
        hidden_dim=128,
        depth=3,
        buffer_capacity=20000,
        batch_size=64,
        color="#2a9d8f",
    ),
]


def run_single_seed(
    config: ExperimentConfig, seed: int, num_frames: int
) -> pd.DataFrame:
    env = gym.make("CartPole-v1")
    set_seed(env, seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    agent = DQNAgent(
        env,
        buffer_capacity=config.buffer_capacity,
        batch_size=config.batch_size,
        lr=LEARNING_RATE,
        gamma=GAMMA,
        epsilon_start=EPSILON_START,
        epsilon_final=EPSILON_FINAL,
        epsilon_decay=EPSILON_DECAY,
        target_update_freq=TARGET_UPDATE_FREQ,
        hidden_dim=config.hidden_dim,
        depth=config.depth,
        seed=seed,
    )

    state, _ = env.reset(seed=seed)
    episode_reward = 0.0
    frame = 0
    episode_frames: list[int] = []
    episode_rewards: list[float] = []

    print(f"  seed={seed}: running {num_frames} frames")

    while frame < num_frames:
        action = agent.predict_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        agent.buffer.add(state, action, reward, next_state, done, {})
        if len(agent.buffer) >= agent.batch_size:
            batch = agent.buffer.sample(agent.batch_size)
            agent.update_agent(batch)

        episode_reward += float(reward)
        frame += 1
        state = next_state

        if done:
            episode_frames.append(frame)
            episode_rewards.append(episode_reward)
            state, _ = env.reset(seed=seed)
            episode_reward = 0.0

    env.close()

    if episode_frames:
        rolling_rewards = (
            pd.Series(episode_rewards)
            .rolling(window=SMOOTHING_WINDOW, min_periods=1)
            .mean()
            .to_numpy()
        )
        return pd.DataFrame(
            {
                "frame": episode_frames,
                "episode_reward": episode_rewards,
                "smoothed_reward": rolling_rewards,
            }
        )

    return pd.DataFrame(
        {
            "frame": [num_frames],
            "episode_reward": [0.0],
            "smoothed_reward": [0.0],
        }
    )


def aggregate_curves(
    seed_frames: dict[int, np.ndarray], seed_rewards: dict[int, np.ndarray]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame_grid = np.linspace(0, NUM_FRAMES, GRID_POINTS)
    aligned_rewards = []

    for seed in SEEDS:
        frames = seed_frames[seed]
        rewards = seed_rewards[seed]
        if len(frames) == 1:
            aligned = np.full_like(frame_grid, rewards[0], dtype=float)
        else:
            left = rewards[0]
            right = rewards[-1]
            aligned = np.interp(frame_grid, frames, rewards, left=left, right=right)
        aligned_rewards.append(aligned)

    stacked = np.stack(aligned_rewards, axis=0)
    return frame_grid, stacked.mean(axis=0), stacked.std(axis=0)


def plot_configuration(config: ExperimentConfig) -> dict[str, Any]:
    config_dir = RESULTS_DIR / config.name
    config_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Running {config.name} | width={config.hidden_dim}, depth={config.depth}, buffer={config.buffer_capacity}, batch={config.batch_size}"
    )

    per_seed_frames: dict[int, np.ndarray] = {}
    per_seed_rewards: dict[int, np.ndarray] = {}
    raw_runs: list[pd.DataFrame] = []

    for seed in SEEDS:
        run_df = run_single_seed(config, seed, NUM_FRAMES)
        run_df.to_csv(config_dir / f"seed_{seed}.csv", index=False)
        run_df["seed"] = seed
        raw_runs.append(run_df)
        per_seed_frames[seed] = run_df["frame"].to_numpy(dtype=float)
        per_seed_rewards[seed] = run_df["smoothed_reward"].to_numpy(dtype=float)

    all_runs = pd.concat(raw_runs, ignore_index=True)
    all_runs.to_csv(config_dir / "all_seeds.csv", index=False)

    frame_grid, mean_curve, std_curve = aggregate_curves(
        per_seed_frames, per_seed_rewards
    )
    summary = pd.DataFrame(
        {
            "frame": frame_grid,
            "mean_reward": mean_curve,
            "std_reward": std_curve,
        }
    )
    summary.to_csv(config_dir / "summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        frame_grid, mean_curve, color=config.color, linewidth=2.5, label="mean reward"
    )
    ax.fill_between(
        frame_grid,
        mean_curve - std_curve,
        mean_curve + std_curve,
        color=config.color,
        alpha=0.18,
        label="±1 std across seeds",
    )
    ax.set_title(
        f"{config.name.replace('_', ' ').title()} | width={config.hidden_dim}, depth={config.depth}, buffer={config.buffer_capacity}, batch={config.batch_size}"
    )
    ax.set_xlabel("Frames")
    ax.set_ylabel("Mean reward")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(config_dir / "training_curve.png", dpi=180)
    plt.close(fig)

    print(f"  saved {config_dir / 'training_curve.png'}")

    final_window = summary["mean_reward"].tail(20)
    return {
        "name": config.name,
        "final_mean_reward": float(final_window.mean()),
        "peak_mean_reward": float(summary["mean_reward"].max()),
        "config_dir": str(config_dir),
    }


def main() -> None:
    torch.set_num_threads(1)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summaries = [plot_configuration(config) for config in CONFIGS]
    pd.DataFrame(summaries).to_csv(RESULTS_DIR / "summary_table.csv", index=False)
    print(pd.DataFrame(summaries).to_string(index=False))

    # combined plot across all configurations
    fig, ax = plt.subplots(figsize=(10, 6))
    for config in CONFIGS:
        csv = RESULTS_DIR / config.name / "summary.csv"
        if not csv.exists():
            print(f"Skipping combined plot for missing {csv}")
            continue
        df = pd.read_csv(csv)
        x = df["frame"].to_numpy()
        y = df["mean_reward"].to_numpy()
        std = df["std_reward"].to_numpy()
        ax.plot(
            x, y, label=config.name.replace("_", " "), color=config.color, linewidth=2
        )
        ax.fill_between(x, y - std, y + std, color=config.color, alpha=0.18)

    ax.set_xlabel("Frames")
    ax.set_ylabel("Mean reward")
    ax.set_title("DQN L1: Architecture Ablation")
    ax.legend()
    out = RESULTS_DIR / "combined_training_curve.png"
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"Saved combined plot to: {out}")


if __name__ == "__main__":
    main()
