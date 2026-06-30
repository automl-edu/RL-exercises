from typing import Any, Dict, List, Tuple

import gymnasium as gym
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig
from rl_exercises.agent import AbstractAgent


def set_seed(env: gym.Env, seed: int = 0) -> None:
    """
    Seed random number generators for reproducibility.

    Parameters
    ----------
    env : gym.Env
        Gymnasium environment to seed.
    seed : int, optional
        Seed value for NumPy, PyTorch, and environment (default is 0).
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    env.reset(seed=seed)

    if hasattr(env.action_space, "seed"):
        env.action_space.seed(seed)

    if hasattr(env.observation_space, "seed"):
        env.observation_space.seed(seed)


class Policy(nn.Module):
    """
    Multi-layer perceptron mapping states to action probabilities.
    """

    def __init__(
        self,
        state_space: gym.spaces.Box,
        action_space: gym.spaces.Discrete,
        hidden_size: int = 128,
    ):
        """
        Initialize the policy network.
        """
        super().__init__()

        self.state_dim = int(np.prod(state_space.shape))
        self.fc1 = nn.Linear(self.state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_space.n)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute action probabilities for given state(s).
        """

        # If input is one single state, add batch dimension
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # Flatten input
        x = x.view(x.size(0), -1)

        # First linear layer + ReLU
        x = torch.relu(self.fc1(x))

        # Second linear layer gives logits
        logits = self.fc2(x)

        # Convert logits to action probabilities
        probs = torch.softmax(logits, dim=-1)

        return probs


class REINFORCEAgent(AbstractAgent):
    """
    REINFORCE agent performing on-policy Monte Carlo policy gradient updates.
    """

    def __init__(
        self,
        env: gym.Env,
        lr: float = 1e-2,
        gamma: float = 0.99,
        seed: int = 0,
        hidden_size: int = 128,
    ) -> None:
        """
        Initialize the REINFORCE agent.
        """

        set_seed(env, seed)

        self.env = env
        self.gamma = gamma

        self.policy = Policy(env.observation_space, env.action_space, hidden_size)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        self.total_episodes = 0

    def predict_action(
        self,
        state: np.ndarray,
        info: Dict[str, Any] = {},
        evaluate: bool = False,
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Select an action according to the current policy.
        """

        state_tensor = torch.tensor(state, dtype=torch.float32)

        probs = self.policy(state_tensor)

        if evaluate:
            action = int(torch.argmax(probs, dim=-1).item())
            return action, {}

        distribution = torch.distributions.Categorical(probs=probs)
        action_tensor = distribution.sample()
        log_prob = distribution.log_prob(action_tensor)

        action = int(action_tensor.item())

        return action, {"log_prob": log_prob.squeeze()}

    def compute_returns(self, rewards: List[float]) -> torch.Tensor:
        """
        Compute discounted reward-to-go for each timestep.
        """

        returns = []
        R = 0.0

        # Go backwards through the episode
        for reward in reversed(rewards):
            R = reward + self.gamma * R
            returns.insert(0, R)

        return torch.tensor(returns, dtype=torch.float32)

    def update_agent(
        self,
        training_batch: List[
            Tuple[np.ndarray, int, float, np.ndarray, bool, Dict[str, Any]]
        ],
    ) -> float:
        """
        Perform a policy-gradient update using one full episode.
        """

        # Extract log probabilities and rewards from episode
        log_probs = [t[5]["log_prob"] for t in training_batch]
        rewards = [t[2] for t in training_batch]

        # Compute discounted returns
        returns_t = self.compute_returns(rewards)

        # Normalize advantages / returns
        advantages = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        # Policy gradient loss
        lp_tensor = torch.stack(log_probs)
        loss = -torch.sum(lp_tensor * advantages)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return float(loss.item())

    def save(self, path: str) -> None:
        """
        Save policy network and optimizer state to a checkpoint.
        """
        torch.save(
            {
                "policy": self.policy.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        """
        Load policy network and optimizer state from checkpoint.
        """
        ckpt = torch.load(path)
        self.policy.load_state_dict(ckpt["policy"])
        self.optimizer.load_state_dict(ckpt["optimizer"])

    def evaluate(
        self,
        eval_env: gym.Env,
        num_episodes: int = 10,
    ) -> Tuple[float, float]:
        """
        Evaluate policy over multiple episodes.
        """

        self.policy.eval()

        returns: List[float] = []

        for _ in range(num_episodes):
            state, _ = eval_env.reset()
            done = False
            episode_return = 0.0

            while not done:
                action, _ = self.predict_action(state, evaluate=True)
                next_state, reward, term, trunc, _ = eval_env.step(action)

                done = term or trunc
                episode_return += float(reward)
                state = next_state

            returns.append(episode_return)

        self.policy.train()

        mean = np.mean(returns) if returns else 0.0
        std = np.std(returns) if returns else 0.0

        return float(mean), float(std)

    def train(
        self,
        num_episodes: int,
        eval_interval: int = 10,
        eval_episodes: int = 5,
    ) -> None:
        """
        Train the agent on-policy for a number of episodes.
        """

        eval_env = gym.make(self.env.spec.id)

        for ep in range(1, num_episodes + 1):
            state, _ = self.env.reset()
            done = False
            batch: List[Tuple[Any, ...]] = []

            while not done:
                action, info = self.predict_action(state)
                next_state, reward, term, trunc, _ = self.env.step(action)

                done = term or trunc

                batch.append(
                    (
                        state,
                        action,
                        float(reward),
                        next_state,
                        done,
                        info,
                    )
                )

                state = next_state

            loss = self.update_agent(batch)
            total_return = sum(r for _, _, r, *_ in batch)
            self.total_episodes += 1

            if ep % 10 == 0:
                print(f"[Train] Ep {ep:3d} Return {total_return:5.1f} Loss {loss:.3f}")

            if ep % eval_interval == 0:
                mean_ret, std_ret = self.evaluate(
                    eval_env,
                    num_episodes=eval_episodes,
                )
                print(f"[Eval ] Ep {ep:3d} AvgReturn {mean_ret:5.1f} ± {std_ret:4.1f}")

        eval_env.close()
        print("Training complete.")


@hydra.main(
    config_path="../configs/agent/",
    config_name="reinforce",
    version_base="1.1",
)
def main(cfg: DictConfig) -> None:
    """
    Main entry point for training with Hydra configuration.
    """

    print(f"config: {cfg}")

    env = gym.make(cfg.env.name)
    set_seed(env, cfg.seed)

    agent = REINFORCEAgent(
        env=env,
        lr=cfg.agent.lr,
        gamma=cfg.agent.gamma,
        seed=cfg.seed,
        hidden_size=cfg.agent.hidden_size,
    )

    agent.train(
        num_episodes=cfg.train.episodes,
        eval_interval=cfg.train.eval_interval,
        eval_episodes=cfg.train.eval_episodes,
    )

    env.close()


if __name__ == "__main__":
    main()
