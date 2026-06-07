"""
On-policy Actor-Critic supporting four baseline modes: 'none', 'avg', 'value', and 'gae',
trained for a total number of environment steps with periodic evaluation.
Adds GAE for low-variance advantage estimation.
"""

from typing import Any, List, Tuple

import gymnasium as gym
import hydra
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from omegaconf import DictConfig
from rl_exercises.agent import AbstractAgent
from rl_exercises.week_6.networks import Policy, ValueNetwork
from torch.distributions import Categorical


def set_seed(env: gym.Env, seed: int = 0) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    env.reset(seed=seed)

    if hasattr(env.action_space, "seed"):
        env.action_space.seed(seed)

    if hasattr(env.observation_space, "seed"):
        env.observation_space.seed(seed)


class ActorCriticAgent(AbstractAgent):
    def __init__(
        self,
        env: gym.Env,
        lr_actor: float = 5e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        seed: int = 0,
        hidden_size: int = 128,
        baseline_type: str = "value",
        baseline_decay: float = 0.9,
    ) -> None:
        set_seed(env, seed)

        assert baseline_type in (
            "none",
            "avg",
            "value",
            "gae",
        ), "baseline_type must be one of: none, avg, value, gae"

        self.env = env
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.baseline_type = baseline_type
        self.baseline_decay = baseline_decay

        # Actor / policy network
        self.policy = Policy(env.observation_space, env.action_space, hidden_size)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr_actor)

        # Critic / value network for value and GAE baselines
        if baseline_type in ("value", "gae"):
            self.value_fn = ValueNetwork(env.observation_space, hidden_size)
            self.value_optimizer = optim.Adam(self.value_fn.parameters(), lr=lr_critic)

        # Running average baseline
        if baseline_type == "avg":
            self.running_return = 0.0

    def predict_action(
        self, state: np.ndarray, evaluate: bool = False
    ) -> Tuple[int, torch.Tensor | None]:
        state_tensor = torch.from_numpy(state).float()
        probs = self.policy(state_tensor).squeeze(0)

        if evaluate:
            action = int(torch.argmax(probs).item())
            return action, None

        dist = Categorical(probs=probs)
        action_tensor = dist.sample()
        log_prob = dist.log_prob(action_tensor)

        return int(action_tensor.item()), log_prob

    def compute_returns(self, rewards: List[float]) -> torch.Tensor:
        R = 0.0
        returns = []

        for reward in reversed(rewards):
            R = reward + self.gamma * R
            returns.insert(0, R)

        return torch.tensor(returns, dtype=torch.float32)

    def compute_advantages(
        self, states: List[np.ndarray], rewards: List[float]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Discounted Monte Carlo returns
        returns = self.compute_returns(rewards)

        # Batch states
        states_t = torch.tensor(np.array(states), dtype=torch.float32)

        # Value estimates V(s)
        values = self.value_fn(states_t)

        # Advantage = return - value
        advantages = returns - values.detach()

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (
            advantages.std(unbiased=False) + 1e-8
        )

        return advantages.detach(), returns.detach()

    def compute_gae(
        self,
        states: List[np.ndarray],
        rewards: List[float],
        next_states: List[np.ndarray],
        dones: List[bool],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        states_t = torch.tensor(np.array(states), dtype=torch.float32)
        next_states_t = torch.tensor(np.array(next_states), dtype=torch.float32)
        rewards_t = torch.tensor(rewards, dtype=torch.float32)
        dones_t = torch.tensor(dones, dtype=torch.float32)

        with torch.no_grad():
            values = self.value_fn(states_t)
            next_values = self.value_fn(next_states_t)

            # TD error:
            # delta_t = r_t + gamma * V(s_{t+1}) * (1 - done) - V(s_t)
            deltas = rewards_t + self.gamma * next_values * (1 - dones_t) - values

            advantages = torch.zeros_like(rewards_t)
            gae = 0.0

            # Accumulate GAE backwards
            for t in reversed(range(len(rewards))):
                gae = deltas[t] + self.gamma * self.gae_lambda * (1 - dones_t[t]) * gae
                advantages[t] = gae

            # Return target for critic
            returns = advantages + values

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (
            advantages.std(unbiased=False) + 1e-8
        )

        return advantages.detach(), returns.detach()

    def update_agent(
        self,
        trajectory: List[Tuple[np.ndarray, int, float, np.ndarray, bool, Any]],
    ) -> Tuple[float, float]:
        states, actions, rewards, next_states, dones, log_probs = zip(*trajectory)

        # Select baseline / advantage method
        if self.baseline_type == "gae":
            adv, ret = self.compute_gae(
                list(states),
                list(rewards),
                list(next_states),
                list(dones),
            )

        elif self.baseline_type == "value":
            adv, ret = self.compute_advantages(
                list(states),
                list(rewards),
            )

        elif self.baseline_type == "avg":
            ret = self.compute_returns(list(rewards))

            # Advantage = return - running average baseline
            adv = ret - self.running_return

            # Normalize advantages
            adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

            # Update running average baseline
            mean_return = float(ret.mean().item())
            self.running_return = (
                self.baseline_decay * self.running_return
                + (1 - self.baseline_decay) * mean_return
            )

        else:
            # No learned baseline, only normalized returns
            ret = self.compute_returns(list(rewards))
            adv = (ret - ret.mean()) / (ret.std(unbiased=False) + 1e-8)

        # Policy update
        logp_t = torch.stack(log_probs)
        policy_loss = -(logp_t * adv.detach()).mean()

        self.policy_optimizer.zero_grad()

        if policy_loss.requires_grad:
            policy_loss.backward()
            self.policy_optimizer.step()

        # Critic update
        if self.baseline_type in ("value", "gae"):
            states_t = torch.tensor(np.array(states), dtype=torch.float32)
            vals = self.value_fn(states_t)

            value_loss = F.mse_loss(vals, ret)

            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()

        else:
            value_loss = torch.tensor(0.0)

        return float(policy_loss.item()), float(value_loss.item())

    def evaluate(
        self, eval_env: gym.Env, num_episodes: int = 10
    ) -> Tuple[float, float]:
        self.policy.eval()

        returns: List[float] = []

        with torch.no_grad():
            for _ in range(num_episodes):
                state, _ = eval_env.reset()
                done = False
                total_r = 0.0

                while not done:
                    action, _ = self.predict_action(state, evaluate=True)
                    state, reward, term, trunc, _ = eval_env.step(action)

                    done = term or trunc
                    total_r += float(reward)

                returns.append(total_r)

        self.policy.train()

        return float(np.mean(returns)), float(np.std(returns))

    def train(
        self,
        total_steps: int,
        eval_interval: int = 10000,
        eval_episodes: int = 5,
    ) -> None:
        eval_env = gym.make(self.env.spec.id)
        step_count = 0

        while step_count < total_steps:
            state, _ = self.env.reset()
            done = False
            trajectory: List[Any] = []

            while not done and step_count < total_steps:
                action, logp = self.predict_action(state)
                next_state, reward, term, trunc, _ = self.env.step(action)

                done = term or trunc

                trajectory.append(
                    (
                        state,
                        action,
                        float(reward),
                        next_state,
                        done,
                        logp,
                    )
                )

                state = next_state
                step_count += 1

                if step_count % eval_interval == 0:
                    mean_r, std_r = self.evaluate(
                        eval_env,
                        num_episodes=eval_episodes,
                    )
                    print(
                        f"[Eval ] Step {step_count:6d} AvgReturn {mean_r:5.1f} ± {std_r:4.1f}"
                    )

            policy_loss, value_loss = self.update_agent(trajectory)
            total_return = sum(r for _, _, r, *_ in trajectory)

            print(
                f"[Train] Step {step_count:6d} Return {total_return:5.1f} "
                f"Policy Loss {policy_loss:.3f} Value Loss {value_loss:.3f}"
            )

        eval_env.close()
        print("Training complete.")

    def save(self, path: str) -> None:
        checkpoint = {
            "policy": self.policy.state_dict(),
            "baseline_type": self.baseline_type,
        }

        if self.baseline_type in ("value", "gae"):
            checkpoint["value_fn"] = self.value_fn.state_dict()

        torch.save(checkpoint, path)

    def load(self, path: str) -> None:
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint["policy"])

        if self.baseline_type in ("value", "gae") and "value_fn" in checkpoint:
            self.value_fn.load_state_dict(checkpoint["value_fn"])


@hydra.main(
    config_path="../configs/agent/",
    config_name="actor-critic",
    version_base="1.1",
)
def main(cfg: DictConfig) -> None:
    env = gym.make(cfg.env.name)
    set_seed(env, cfg.seed)

    agent = ActorCriticAgent(
        env,
        lr_actor=cfg.agent.lr_actor,
        lr_critic=cfg.agent.lr_critic,
        gamma=cfg.agent.gamma,
        gae_lambda=cfg.agent.get("gae_lambda", 0.95),
        seed=cfg.seed,
        hidden_size=cfg.agent.hidden_size,
        baseline_type=cfg.agent.baseline_type,
        baseline_decay=cfg.agent.get("baseline_decay", 0.9),
    )

    agent.train(
        cfg.train.total_steps,
        cfg.train.eval_interval,
        cfg.train.eval_episodes,
    )

    env.close()


if __name__ == "__main__":
    main()
