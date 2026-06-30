"""
Deep Q-Learning with RND implementation.
"""

from typing import Any, Dict, List, Tuple

import gymnasium as gym
import hydra
import numpy as np
import torch
import torch.optim as optim
from omegaconf import DictConfig
from rl_exercises.week_4.dqn import DQNAgent, set_seed
from rl_exercises.week_7.rnd_utils import PredictorNetwork, TargetNetwork
from torch import nn


class RNDDQNAgent(DQNAgent):
    """
    Deep Q-Learning agent with Random Network Distillation exploration bonus.
    """

    def __init__(
        self,
        env: gym.Env,
        buffer_capacity: int = 10000,
        batch_size: int = 32,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_final: float = 0.01,
        epsilon_decay: int = 500,
        target_update_freq: int = 1000,
        seed: int = 0,
        rnd_hidden_size: int = 128,
        rnd_lr: float = 1e-3,
        rnd_update_freq: int = 1000,
        rnd_n_layers: int = 2,
        rnd_reward_weight: float = 0.1,
    ) -> None:
        super().__init__(
            env,
            buffer_capacity,
            batch_size,
            lr,
            gamma,
            epsilon_start,
            epsilon_final,
            epsilon_decay,
            target_update_freq,
            seed,
        )

        self.env = env
        self.seed = seed
        set_seed(env, seed)

        self.rnd_update_freq = rnd_update_freq
        self.rnd_reward_weight = rnd_reward_weight

        obs_dim = env.observation_space.shape[0]
        output_dim = rnd_hidden_size

        # Target network is random and frozen.
        self.target_network_rnd = TargetNetwork(
            obs_dim=obs_dim,
            output_dim=output_dim,
            hidden_dim=rnd_hidden_size,
            n_layers=rnd_n_layers,
        )

        # Predictor network is trained to predict target network output.
        self.predictor_network_rnd = PredictorNetwork(
            obs_dim=obs_dim,
            output_dim=output_dim,
            hidden_dim=rnd_hidden_size,
            n_layers=rnd_n_layers,
        )

        self.rnd_optimizer = optim.Adam(
            self.predictor_network_rnd.parameters(),
            lr=rnd_lr,
        )

    def update_rnd(
        self, training_batch: List[Tuple[Any, Any, float, Any, bool, Dict]]
    ) -> float:
        """
        Perform one gradient update on the RND predictor network.
        """

        _, _, _, next_states, _, _ = zip(*training_batch)
        next_states_t = torch.tensor(np.array(next_states), dtype=torch.float32)

        with torch.no_grad():
            target_embeddings = self.target_network_rnd(next_states_t)

        predictor_embeddings = self.predictor_network_rnd(next_states_t)

        mse = nn.MSELoss()(predictor_embeddings, target_embeddings)

        self.rnd_optimizer.zero_grad()
        mse.backward()
        self.rnd_optimizer.step()

        return float(mse.item())

    def get_rnd_bonus(self, state: np.ndarray) -> float:
        """
        Compute the RND bonus for a given state.
        """

        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            target_embedding = self.target_network_rnd(state_tensor)
            predictor_embedding = self.predictor_network_rnd(state_tensor)

            error = torch.mean((predictor_embedding - target_embedding) ** 2)

        bonus = self.rnd_reward_weight * float(error.item())

        return bonus

    def train(self, num_frames: int, eval_interval: int = 1000) -> None:
        """
        Run a training loop for a fixed number of frames.
        """

        state, _ = self.env.reset()
        ep_reward = 0.0
        recent_rewards: List[float] = []
        episode_rewards = []
        steps = []

        for frame in range(1, num_frames + 1):
            action = self.predict_action(state)
            next_state, reward, done, truncated, _ = self.env.step(action)

            # Add intrinsic RND exploration bonus.
            rnd_bonus = self.get_rnd_bonus(next_state)
            reward = float(reward) + rnd_bonus

            self.buffer.add(
                state,
                action,
                reward,
                next_state,
                done or truncated,
                {},
            )

            state = next_state
            ep_reward += reward

            if len(self.buffer) >= self.batch_size:
                batch = self.buffer.sample(self.batch_size)

                _ = self.update_agent(batch)

                if frame % self.rnd_update_freq == 0:
                    _ = self.update_rnd(batch)

            if done or truncated:
                state, _ = self.env.reset()
                recent_rewards.append(ep_reward)
                episode_rewards.append(ep_reward)
                steps.append(frame)
                ep_reward = 0.0

                if len(recent_rewards) % 10 == 0:
                    avg = np.mean(recent_rewards[-10:])
                    print(
                        f"Frame {frame}, AvgReward(10): {avg:.2f}, ε={self.epsilon():.3f}"
                    )

        print("Training complete.")


@hydra.main(config_path="../configs/agent/", config_name="rnd_dqn", version_base="1.1")
def main(cfg: DictConfig):
    env = gym.make(cfg.env.name)
    set_seed(env, cfg.seed)

    agent = RNDDQNAgent(
        env,
        buffer_capacity=cfg.agent.buffer_capacity,
        batch_size=cfg.agent.batch_size,
        lr=cfg.agent.learning_rate,
        gamma=cfg.agent.gamma,
        epsilon_start=cfg.agent.epsilon_start,
        epsilon_final=cfg.agent.epsilon_final,
        epsilon_decay=cfg.agent.epsilon_decay,
        target_update_freq=cfg.agent.target_update_freq,
        seed=cfg.seed,
        rnd_hidden_size=cfg.rnd.hidden_size,
        rnd_lr=cfg.rnd.learning_rate,
        rnd_update_freq=cfg.rnd.update_freq,
        rnd_n_layers=cfg.rnd.n_layers,
        rnd_reward_weight=cfg.rnd.reward_weight,
    )

    agent.train(cfg.train.num_frames, cfg.train.eval_interval)

    env.close()


if __name__ == "__main__":
    main()
