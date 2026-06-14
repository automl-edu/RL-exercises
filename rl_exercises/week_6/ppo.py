"""
On-policy Proximal Policy Optimization (PPO) with GAE, clipped surrogate objective,
value-loss coefficient, and entropy bonus, trained for a total number of environment steps.
"""

from typing import Any, List, Tuple

import os
import random

import gymnasium as gym
import hydra
import numpy as np
import torch
import torch.optim as optim
from omegaconf import DictConfig
from rl_exercises.agent import AbstractAgent
from rl_exercises.week_6.networks import Policy, ValueNetwork
from torch.distributions import Categorical

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def set_seed(env: gym.Env, seed: int = 0) -> None:
    env.reset(seed=seed)

    if hasattr(env.action_space, "seed"):
        env.action_space.seed(seed)

    if hasattr(env.observation_space, "seed"):
        env.observation_space.seed(seed)

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


class PPOAgent(AbstractAgent):
    def __init__(
        self,
        env: gym.Env,
        lr_actor: float = 5e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        epochs: int = 4,
        batch_size: int = 64,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        seed: int = 0,
        hidden_size: int = 128,
        max_grad_norm: float = 0.5,
        target_kl: float = 0.03,
        value_clip_eps: float = 0.2,
    ) -> None:
        set_seed(env, seed)

        self.seed = seed
        self.env = env
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef

        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.value_clip_eps = value_clip_eps

        self.policy = Policy(env.observation_space, env.action_space, hidden_size)
        self.value_fn = ValueNetwork(env.observation_space, hidden_size)

        self.optimizer = optim.Adam(
            [
                {"params": self.policy.parameters(), "lr": lr_actor},
                {"params": self.value_fn.parameters(), "lr": lr_critic},
            ]
        )

    def predict(
        self, state: np.ndarray
    ) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        state_tensor = torch.from_numpy(state).float()

        probs = self.policy(state_tensor).squeeze(0)
        dist = Categorical(probs=probs)

        action_tensor = dist.sample()
        action = int(action_tensor.item())

        log_prob = dist.log_prob(action_tensor)
        entropy = dist.entropy()
        value = self.value_fn(state_tensor).squeeze()

        return action, log_prob, entropy, value

    def predict_action(
        self,
        state: np.ndarray,
        info: dict | None = None,
        evaluate: bool = False,
    ) -> Tuple[int, dict]:
        state_tensor = torch.from_numpy(state).float()

        probs = self.policy(state_tensor).squeeze(0)

        if evaluate:
            action = int(torch.argmax(probs).item())
            return action, {}

        dist = Categorical(probs=probs)
        action_tensor = dist.sample()
        action = int(action_tensor.item())

        return action, {"log_prob": dist.log_prob(action_tensor)}

    def compute_gae(
        self,
        rewards: List[float],
        values: torch.Tensor,
        next_values: torch.Tensor,
        dones: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        rewards_t = torch.tensor(rewards, dtype=torch.float32)

        values = values.detach().view(-1)
        next_values = next_values.detach().view(-1)
        dones = dones.detach().float().view(-1)

        deltas = rewards_t + self.gamma * next_values * (1 - dones) - values

        advantages = torch.zeros_like(rewards_t)
        gae = 0.0

        for t in reversed(range(len(rewards))):
            gae = deltas[t] + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + values

        advantages = (advantages - advantages.mean()) / (
            advantages.std(unbiased=False) + 1e-8
        )

        return advantages.detach(), returns.detach()

    def update(self, trajectory: List[Any]) -> Tuple[float, float, float]:
        states = torch.stack([torch.from_numpy(t[0]).float() for t in trajectory])
        actions = torch.tensor([t[1] for t in trajectory], dtype=torch.int64)
        old_logps = torch.stack([t[2] for t in trajectory]).detach()
        rewards = [t[4] for t in trajectory]
        dones = torch.tensor([t[5] for t in trajectory], dtype=torch.float32)
        next_states = torch.stack([torch.from_numpy(t[6]).float() for t in trajectory])

        with torch.no_grad():
            values = self.value_fn(states)
            next_values = self.value_fn(next_states)

        advantages, returns = self.compute_gae(
            rewards=rewards,
            values=values,
            next_values=next_values,
            dones=dones,
        )

        dataset = torch.utils.data.TensorDataset(
            states,
            actions,
            old_logps,
            advantages,
            returns,
        )

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

        last_policy_loss = torch.tensor(0.0)
        last_value_loss = torch.tensor(0.0)
        last_entropy_loss = torch.tensor(0.0)

        for _ in range(self.epochs):
            stop_early = False

            for b_states, b_actions, b_oldlogp, b_adv, b_ret in loader:
                probs = self.policy(b_states)
                dist = Categorical(probs=probs)

                new_logp = dist.log_prob(b_actions)

                ratio = torch.exp(new_logp - b_oldlogp)

                unclipped = ratio * b_adv
                clipped = (
                    torch.clamp(
                        ratio,
                        1.0 - self.clip_eps,
                        1.0 + self.clip_eps,
                    )
                    * b_adv
                )

                policy_loss = -torch.mean(torch.min(unclipped, clipped))

                values_pred = self.value_fn(b_states)

                values_pred_clipped = b_ret + torch.clamp(
                    values_pred - b_ret,
                    -self.value_clip_eps,
                    self.value_clip_eps,
                )

                value_loss_unclipped = (values_pred - b_ret) ** 2
                value_loss_clipped = (values_pred_clipped - b_ret) ** 2

                value_loss = 0.5 * torch.mean(
                    torch.max(value_loss_unclipped, value_loss_clipped)
                )

                entropy_loss = -dist.entropy().mean()

                loss = (
                    policy_loss
                    + self.vf_coef * value_loss
                    + self.ent_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    list(self.policy.parameters()) + list(self.value_fn.parameters()),
                    self.max_grad_norm,
                )

                self.optimizer.step()

                approx_kl = torch.mean(b_oldlogp - new_logp).detach()

                if approx_kl > self.target_kl:
                    stop_early = True

                last_policy_loss = policy_loss.detach()
                last_value_loss = value_loss.detach()
                last_entropy_loss = entropy_loss.detach()

            if stop_early:
                break

        return (
            float(last_policy_loss.item()),
            float(last_value_loss.item()),
            float(last_entropy_loss.item()),
        )

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
                action, logp, ent, val = self.predict(state)
                next_state, reward, term, trunc, _ = self.env.step(action)

                done = term or trunc

                trajectory.append(
                    (
                        state,
                        action,
                        logp,
                        ent,
                        float(reward),
                        float(done),
                        next_state,
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
                        f"[Eval ] Step {step_count:6d} "
                        f"AvgReturn {mean_r:5.1f} ± {std_r:4.1f}"
                    )

            policy_loss, value_loss, entropy_loss = self.update(trajectory)
            total_return = sum(t[4] for t in trajectory)

            print(
                f"[Train] Step {step_count:6d} Return {total_return:5.1f} "
                f"Policy Loss {policy_loss:.3f} "
                f"Value Loss {value_loss:.3f} "
                f"Entropy Loss {entropy_loss:.3f}"
            )

        eval_env.close()
        print("Training complete.")

    def evaluate(
        self,
        eval_env: gym.Env,
        num_episodes: int = 10,
    ) -> Tuple[float, float]:
        returns = []

        self.policy.eval()

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

    def save(self, path: str) -> None:
        torch.save(
            {
                "policy": self.policy.state_dict(),
                "value_fn": self.value_fn.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint["policy"])
        self.value_fn.load_state_dict(checkpoint["value_fn"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])


@hydra.main(
    config_path="../configs/agent/",
    config_name="ppo",
    version_base="1.1",
)
def main(cfg: DictConfig) -> None:
    env = gym.make(cfg.env.name)
    set_seed(env, cfg.seed)

    agent = PPOAgent(
        env,
        lr_actor=cfg.agent.lr_actor,
        lr_critic=cfg.agent.lr_critic,
        gamma=cfg.agent.gamma,
        gae_lambda=cfg.agent.gae_lambda,
        clip_eps=cfg.agent.clip_eps,
        epochs=cfg.agent.epochs,
        batch_size=cfg.agent.batch_size,
        ent_coef=cfg.agent.ent_coef,
        vf_coef=cfg.agent.vf_coef,
        seed=cfg.seed,
        hidden_size=cfg.agent.hidden_size,
        max_grad_norm=cfg.agent.get("max_grad_norm", 0.5),
        target_kl=cfg.agent.get("target_kl", 0.03),
        value_clip_eps=cfg.agent.get("value_clip_eps", 0.2),
    )

    agent.train(
        cfg.train.total_steps,
        cfg.train.eval_interval,
        cfg.train.eval_episodes,
    )

    env.close()


if __name__ == "__main__":
    main()
