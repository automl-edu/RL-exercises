"""Run week 5 REINFORCE empirical variations.

This script runs the qualitative experiments from the week 5 README:
- different trajectory lengths on CartPole-v1
- a LunarLander smoke test with the same architecture and learning rate

The output is written as JSON so it can be inspected later or summarized in
`observations_l2.txt`.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import argparse
import json
from pathlib import Path

import gymnasium as gym
from rl_exercises.week_5.policy_gradient import REINFORCEAgent


def make_env(env_name: str) -> gym.Env:
    """Create an environment, with a LunarLander-v2 -> v3 fallback."""

    try:
        return gym.make(env_name)
    except Exception:
        if env_name == "LunarLander-v2":
            return gym.make("LunarLander-v3")
        raise


def run_training(
    env_name: str,
    *,
    episodes: int,
    lr: float,
    gamma: float,
    hidden_size: int,
    seed: int,
    eval_interval: int,
    eval_episodes: int,
    max_steps_per_episode: Optional[int] = None,
) -> Dict[str, Any]:
    """Train REINFORCE and collect a compact history."""

    env = make_env(env_name)
    eval_env = make_env(env.spec.id)
    agent = REINFORCEAgent(
        env,
        lr=lr,
        gamma=gamma,
        seed=seed,
        hidden_size=hidden_size,
    )

    history: List[Dict[str, Any]] = []
    total_env_steps = 0
    solved_step: Optional[int] = None

    for episode in range(1, episodes + 1):
        state, _ = env.reset()
        done = False
        batch: List[Any] = []
        episode_steps = 0

        while not done:
            action, info = agent.predict_action(state)
            next_state, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            batch.append((state, action, float(reward), next_state, done, info))
            state = next_state
            episode_steps += 1
            total_env_steps += 1

            if (
                max_steps_per_episode is not None
                and episode_steps >= max_steps_per_episode
            ):
                break

        loss = agent.update_agent(batch)

        if episode % eval_interval == 0:
            mean_ret, std_ret = agent.evaluate(eval_env, num_episodes=eval_episodes)
            history.append(
                {
                    "episode": episode,
                    "mean_return": float(mean_ret),
                    "std_return": float(std_ret),
                    "loss": float(loss),
                    "episode_steps": episode_steps,
                    "total_env_steps": total_env_steps,
                }
            )

            if (
                solved_step is None
                and env_name.startswith("CartPole")
                and mean_ret >= 475.0
            ):
                solved_step = total_env_steps

    env.close()
    eval_env.close()

    return {
        "env": env_name,
        "effective_eval_env": env.spec.id,
        "episodes": episodes,
        "lr": lr,
        "gamma": gamma,
        "hidden_size": hidden_size,
        "seed": seed,
        "max_steps_per_episode": max_steps_per_episode,
        "solved_step": solved_step,
        "history": history,
    }


def parse_max_steps(values: Iterable[str]) -> List[Optional[int]]:
    result: List[Optional[int]] = []
    for value in values:
        if value.lower() in {"none", "full", "all"}:
            result.append(None)
        else:
            result.append(int(value))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run week 5 REINFORCE variations.")
    parser.add_argument(
        "--episodes", type=int, default=80, help="Training episodes per run."
    )
    parser.add_argument(
        "--eval-interval", type=int, default=20, help="Evaluation interval in episodes."
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=5,
        help="Episodes used for each evaluation.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    parser.add_argument(
        "--hidden-size", type=int, default=128, help="Hidden layer width."
    )
    parser.add_argument(
        "--cartpole-lengths",
        nargs="+",
        default=["full", "20", "50"],
        help="Trajectory lengths to try on CartPole-v1. Use full/none for untruncated episodes.",
    )
    parser.add_argument(
        "--cartpole-env",
        default="CartPole-v1",
        help="CartPole environment id.",
    )
    parser.add_argument(
        "--lunarlander-env",
        default="LunarLander-v2",
        help="LunarLander environment id; v2 falls back to v3 if needed.",
    )
    parser.add_argument(
        "--output",
        default="results/week_5/l2_variations.json",
        help="Path to the JSON summary output.",
    )
    args = parser.parse_args()

    cartpole_lengths = parse_max_steps(args.cartpole_lengths)
    runs: Dict[str, Any] = {}

    for max_steps in cartpole_lengths:
        key = "cartpole_full" if max_steps is None else f"cartpole_trunc_{max_steps}"
        runs[key] = run_training(
            args.cartpole_env,
            episodes=args.episodes,
            lr=args.lr,
            gamma=args.gamma,
            hidden_size=args.hidden_size,
            seed=args.seed,
            eval_interval=args.eval_interval,
            eval_episodes=args.eval_episodes,
            max_steps_per_episode=max_steps,
        )

    runs["lunarlander_same_hparams"] = run_training(
        args.lunarlander_env,
        episodes=args.episodes,
        lr=args.lr,
        gamma=args.gamma,
        hidden_size=args.hidden_size,
        seed=args.seed,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(runs, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(runs, indent=2, sort_keys=True))
    print(f"Saved summary to {output_path}")


if __name__ == "__main__":
    main()
