# Ignore "imported but unused"
# flake8: noqa: F401
from typing import Any, List, SupportsFloat

import os
import warnings
from functools import partial

import gymnasium as gym
import hydra
import numpy as np
import pandas as pd
import rl_exercises
from gymnasium.core import Env
from gymnasium.wrappers import TimeLimit
from hydra.utils import get_class
from minigrid.wrappers import FlatObsWrapper
from omegaconf import DictConfig, OmegaConf
from rich import print as printr
from rl_exercises.agent import AbstractAgent, RandomAgent
from rl_exercises.agent.buffer import SimpleBuffer
from rl_exercises.environments import MarsRover
from rl_exercises.week_2.policy_iteration import PolicyIteration
from rl_exercises.week_2.value_iteration import ValueIteration

# from rl_exercises.week_4 import EpsilonGreedyPolicy as TabularEpsilonGreedyPolicy
# from rl_exercises.week_4 import SARSAAgent
# from rl_exercises.week_5 import EpsilonGreedyPolicy, TabularQAgent, VFAQAgent
# from rl_exercises.week_6 import DQN, ReplayBuffer
# from rl_exercises.week_7 import REINFORCE
# from rl_exercises.week_8 import EpsilonDecayPolicy, EZGreedyPolicy
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.monitor import Monitor
from tqdm import tqdm


@hydra.main("configs", "base", version_base="1.1")  # type: ignore[misc]
def train(cfg: DictConfig) -> float:
    """Train the agent.

    Parameters
    ----------
    cfg : DictConfig
        Agent/experiment configuration

    Returns
    -------
    float
        Mean return of n eval episodes

    Raises
    ------
    NotImplementedError
        _description_
    """
    env = make_env(cfg.env_name, cfg.env_kwargs)
    printr(cfg)
    if cfg.agent == "sb3":
        return train_sb3(env, cfg)
    elif cfg.agent == "random":
        agent = RandomAgent(env)
    else:
        # TODO: add your agent options here
        raise NotImplementedError

    buffer_cls = eval(cfg.buffer_cls)
    buffer = buffer_cls(**cfg.buffer_kwargs)
    state, info = env.reset(seed=cfg.seed)
    train_reward_buffer = {"steps": [], "train_rewards": []}
    eval_reward_buffer = {"eval_steps": [], "eval_rewards": []}

    for step in range(int(cfg.training_steps)):
        action, info = agent.predict_action(state, info)
        next_state, reward, terminated, truncated, info = env.step(action)

        buffer.add(state, action, reward, next_state, (truncated or terminated), info)
        train_reward_buffer["steps"].append(step)
        train_reward_buffer["train_rewards"].append(reward)

        if len(buffer) > cfg.batch_size or (
            cfg.update_after_episode_end and (terminated or truncated)
        ):
            batch = buffer.sample(cfg.batch_size)
            agent.update_agent(batch)

        state = next_state

        if terminated or truncated:
            state, info = env.reset(seed=cfg.seed)

        if step % cfg.eval_every_n_steps == 0:
            eval_performance = evaluate(
                make_env(cfg.env_name, cfg.env_kwargs),
                agent,
                cfg.n_eval_episodes,
                cfg.seed,
            )
            print(f"Eval reward after {step} steps was {eval_performance}.")
            eval_reward_buffer["eval_steps"].append(step)
            eval_reward_buffer["eval_rewards"].append(eval_performance)

    agent.save(str(os.path.abspath("model.csv")))
    pd.DataFrame(train_reward_buffer).to_csv(
        os.path.abspath("train_rewards.csv"), index=False
    )
    pd.DataFrame(eval_reward_buffer).to_csv(
        os.path.abspath("eval_rewards.csv"), index=False
    )
    final_eval = evaluate(env, agent, cfg.n_eval_episodes)
    print(f"Final eval reward was: {final_eval}")
    return final_eval


def train_sb3(env: gym.Env, cfg: DictConfig) -> float:
    """Train stablebaselines agent on env.

    Parameters
    ----------
    env : gym.Env
        Environment
    cfg : DictConfig
        Agent/experiment configuration

    Returns
    -------
    float
        Mean rewards
    """
    # Create agent
    model = eval(cfg.agent_class)(
        "MlpPolicy",
        env,
        verbose=cfg.verbose,
        tensorboard_log=cfg.log_dir,
        seed=cfg.seed,
        **cfg.agent_kwargs,
    )

    # Train agent
    model.learn(total_timesteps=cfg.total_timesteps)

    # Save agent
    model.save(cfg.model_fn)

    # Evaluate
    env = Monitor(gym.make(cfg.env_id), filename="eval")
    means = evaluate(env, model, episodes=cfg.n_eval_episodes, seed=cfg.seed)
    performance = np.mean(means)
    return performance


def evaluate(
    env: gym.Env, agent: AbstractAgent, episodes: int = 100, seed: int = 0
) -> float:
    """Evaluate a given Policy on an Environment.

    Parameters
    ----------
    env: gym.Env
        Environment to evaluate on
    policy: Callable[[np.ndarray], int]
        Policy to evaluate
    episodes: int
        Evaluation episodes

    Returns
    -------
    float
        Mean evaluation rewards
    """
    episode_rewards: List[float] = []
    pbar = tqdm(total=episodes)
    for _ in range(episodes):
        obs, info = env.reset(seed=seed)
        episode_rewards.append(0)
        done = False
        episode_steps = 0
        while not done:
            action, _ = agent.predict_action(obs, info, evaluate=True)  # type: ignore[arg-type]
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_rewards[-1] += reward
            episode_steps += 1
            if terminated or truncated:
                done = True
                pbar.set_postfix(
                    {
                        "episode reward": episode_rewards[-1],
                        "episode step": episode_steps,
                    }
                )
        pbar.update(1)
    env.close()
    return np.mean(episode_rewards)


def make_env(env_name: str, env_kwargs: dict = {}) -> gym.Env:
    """Make environment based on name and kwargs.

    Parameters
    ----------
    env_name : str
        Environment name
    env_kwargs : dict, optional
        Optional env config, by default {}

    Returns
    -------
    gym.Env
        Instantiated env
    """
    if env_name == "MarsRover":
        env = MarsRover(**env_kwargs)
        # env = TimeLimit(env, max_episode_steps=env.horizon)
    elif "MiniGrid" in env_name:
        env = gym.make(env_name, **env_kwargs)
        # env = RGBImgObsWrapper(env)
        env = FlatObsWrapper(env)
    else:
        env = gym.make(env_name, **env_kwargs)
    env = Monitor(env, filename="train")
    return env


if __name__ == "__main__":
    train()
