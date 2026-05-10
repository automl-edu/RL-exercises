"""
Level 2: Run DQN with multiple seeds for robust statistics.
Focuses on the baseline configuration from Level 1 to study seed sensitivity.
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from rl_exercises.week_4.dqn import DQNAgent, set_seed

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "week_4" / "l2_dqn"
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
NUM_FRAMES = 5000
EVAL_INTERVAL = 500

# Use the "large" config from L1 (best performer)
CONFIG = {
    "hidden_dim": 128,
    "depth": 3,
    "buffer_capacity": 20000,
    "batch_size": 64,
    "lr": 1e-3,
    "gamma": 0.99,
    "epsilon_start": 1.0,
    "epsilon_final": 0.01,
    "epsilon_decay": 500,
    "target_update_freq": 1000,
}


def run_single_seed(seed: int, num_frames: int) -> pd.DataFrame:
    """Run DQN for one seed, collecting eval rewards at intervals."""
    env = gym.make("CartPole-v1")
    set_seed(env, seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    agent = DQNAgent(
        env,
        buffer_capacity=CONFIG["buffer_capacity"],
        batch_size=CONFIG["batch_size"],
        lr=CONFIG["lr"],
        gamma=CONFIG["gamma"],
        epsilon_start=CONFIG["epsilon_start"],
        epsilon_final=CONFIG["epsilon_final"],
        epsilon_decay=CONFIG["epsilon_decay"],
        target_update_freq=CONFIG["target_update_freq"],
        hidden_dim=CONFIG["hidden_dim"],
        depth=CONFIG["depth"],
        seed=seed,
    )

    state, _ = env.reset(seed=seed)
    episode_reward = 0.0
    frame = 0
    eval_frames: list[int] = []
    eval_rewards: list[float] = []
    episode_buffer: list[float] = []

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
            episode_buffer.append(episode_reward)
            state, _ = env.reset(seed=seed)
            episode_reward = 0.0

        # Eval at intervals: collect mean reward over last N episodes
        if frame % EVAL_INTERVAL == 0 and episode_buffer:
            mean_eval = float(np.mean(episode_buffer[-10:]))
            eval_frames.append(frame)
            eval_rewards.append(mean_eval)

    env.close()
    return pd.DataFrame({"frame": eval_frames, "reward": eval_rewards, "seed": seed})


def main() -> None:
    torch.set_num_threads(1)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Running L2 DQN with {len(SEEDS)} seeds, {NUM_FRAMES} frames each")
    print(
        f"Config: hidden_dim={CONFIG['hidden_dim']}, depth={CONFIG['depth']}, buffer={CONFIG['buffer_capacity']}, batch={CONFIG['batch_size']}"
    )

    all_data = []
    for seed in SEEDS:
        df = run_single_seed(seed, NUM_FRAMES)
        df.to_csv(RESULTS_DIR / f"seed_{seed}.csv", index=False)
        all_data.append(df)

    combined = pd.concat(all_data, ignore_index=True)
    combined.to_csv(RESULTS_DIR / "all_seeds_raw.csv", index=False)
    print(f"Saved raw data to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
