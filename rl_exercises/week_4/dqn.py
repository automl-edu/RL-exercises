"""
Deep Q-Learning implementation.
"""

from typing import Any, Dict, List, Tuple

import gymnasium as gym
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig
from rl_exercises.agent import AbstractAgent
from rl_exercises.week_4.buffers import ReplayBuffer
from rl_exercises.week_4.networks import QNetwork


def set_seed(env: gym.Env, seed: int = 0) -> None:
    """
    Seed Python, NumPy, PyTorch and the Gym environment for reproducibility.

    Parameters
    ----------
    env : gym.Env
        The Gym environment to seed.
    seed : int
        Random seed.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    env.reset(seed=seed)

    if hasattr(env.action_space, "seed"):
        env.action_space.seed(seed)

    if hasattr(env.observation_space, "seed"):
        env.observation_space.seed(seed)


class DQNAgent(AbstractAgent):
    """
    Deep Q-Learning agent with epsilon-greedy policy and target network.
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
    ) -> None:
        """
        Initialize replay buffer, Q-networks, optimizer, and hyperparameters.
        """
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
        set_seed(env, seed)

        obs_dim = env.observation_space.shape[0]
        n_actions = env.action_space.n

        # Main Q-network and target Q-network
        self.q = QNetwork(obs_dim, n_actions)
        self.target_q = QNetwork(obs_dim, n_actions)
        self.target_q.load_state_dict(self.q.state_dict())

        self.optimizer = optim.Adam(self.q.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_capacity)

        # Hyperparameters
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq

        self.total_steps = 0

    def epsilon(self) -> float:
        """
        Compute current epsilon by exponential decay.

        epsilon = epsilon_final + (epsilon_start - epsilon_final)
                  * exp(-total_steps / epsilon_decay)
        """
        return self.epsilon_final + (self.epsilon_start - self.epsilon_final) * np.exp(
            -self.total_steps / self.epsilon_decay
        )

    def predict_action(
        self,
        state: np.ndarray,
        info: Dict[str, Any] = {},
        evaluate: bool = False,
    ) -> int:
        """
        Choose action using epsilon-greedy policy.

        If evaluate=True, always choose the greedy action.
        """

        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        # Evaluation mode: no exploration
        if evaluate:
            with torch.no_grad():
                qvals = self.q(state_tensor)

            action = int(torch.argmax(qvals, dim=1).item())
            return action

        # Training mode: epsilon-greedy
        if np.random.rand() < self.epsilon():
            action = int(self.env.action_space.sample())
        else:
            with torch.no_grad():
                qvals = self.q(state_tensor)

            action = int(torch.argmax(qvals, dim=1).item())

        return action

    def save(self, path: str) -> None:
        """
        Save model and optimizer state to disk.
        """
        torch.save(
            {
                "parameters": self.q.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        """
        Load model and optimizer state from disk.
        """
        checkpoint = torch.load(path)
        self.q.load_state_dict(checkpoint["parameters"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

        # Also sync target network after loading
        self.target_q.load_state_dict(self.q.state_dict())

    def update_agent(
        self,
        training_batch: List[Tuple[Any, Any, float, Any, bool, Dict]],
    ) -> float:
        """
        Perform one DQN update on a batch of transitions.

        Each transition is:
        (state, action, reward, next_state, done, info)
        """

        # Unpack batch
        states, actions, rewards, next_states, dones, _ = zip(*training_batch)

        s = torch.tensor(np.array(states), dtype=torch.float32)
        a = torch.tensor(np.array(actions), dtype=torch.int64).unsqueeze(1)
        r = torch.tensor(np.array(rewards), dtype=torch.float32)
        s_next = torch.tensor(np.array(next_states), dtype=torch.float32)
        done_mask = torch.tensor(np.array(dones), dtype=torch.float32)

        # Q(s, a) for the actions that were actually taken
        pred = self.q(s).gather(1, a).squeeze(1)

        # DQN target:
        # r + gamma * max_a' Q_target(s', a')
        # If done, future value is 0
        with torch.no_grad():
            next_q_values = self.target_q(s_next)
            max_next_q_values = next_q_values.max(dim=1)[0]
            target = r + self.gamma * (1 - done_mask) * max_next_q_values

        loss = nn.MSELoss()(pred, target)

        # Gradient update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Sync target network sometimes
        if self.total_steps % self.target_update_freq == 0:
            self.target_q.load_state_dict(self.q.state_dict())

        self.total_steps += 1

        return float(loss.item())

    def train(self, num_frames: int, eval_interval: int = 1000) -> None:
        """
        Run training loop for a fixed number of frames.
        """

        state, _ = self.env.reset()
        ep_reward = 0.0
        recent_rewards: List[float] = []

        for frame in range(1, num_frames + 1):
            action = self.predict_action(state)

            next_state, reward, done, truncated, _ = self.env.step(action)

            terminal = done or truncated

            # Store transition in replay buffer
            self.buffer.add(
                state,
                action,
                reward,
                next_state,
                terminal,
                {},
            )

            state = next_state
            ep_reward += reward

            # Update if enough samples are available
            if len(self.buffer) >= self.batch_size:
                batch = self.buffer.sample(self.batch_size)
                _ = self.update_agent(batch)

            # End of episode
            if terminal:
                state, _ = self.env.reset()

                recent_rewards.append(ep_reward)
                ep_reward = 0.0

                # Print average reward every 10 episodes
                if len(recent_rewards) % 10 == 0:
                    avg = np.mean(recent_rewards[-10:])
                    print(
                        f"Frame {frame}, AvgReward(10): {avg:.2f}, epsilon={self.epsilon():.3f}"
                    )

        print("Training complete.")


@hydra.main(config_path="../configs/agent/", config_name="dqn", version_base="1.1")
def main(cfg: DictConfig):
    # 1) Build environment
    env = gym.make(cfg.env.name)
    set_seed(env, cfg.seed)

    # 2) Map config to agent arguments
    agent_kwargs = dict(
        buffer_capacity=cfg.buffer_capacity,
        batch_size=cfg.batch_size,
        lr=cfg.lr,
        gamma=cfg.gamma,
        epsilon_start=cfg.epsilon_start,
        epsilon_final=cfg.epsilon_final,
        epsilon_decay=cfg.epsilon_decay,
        target_update_freq=cfg.target_update_freq,
        seed=cfg.seed,
    )

    # 3) Instantiate and train agent
    agent = DQNAgent(env, **agent_kwargs)

    agent.train(
        num_frames=cfg.num_frames,
        eval_interval=cfg.eval_interval,
    )


if __name__ == "__main__":
    main()
