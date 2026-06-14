"""
NovelD rewards the agent for moving into novel states rather than simply being in a novel state.
The intrinsic reward is:
    r_i(s_t, a_t, s_{t+1}) = max(novelty(s_{t+1}) - alpha * novelty(s_t), 0)
                             * indicator(first_visit(s_{t+1}))

where novelty(s) = RND prediction error at state s.
"""

from typing import Any, List, Set, Tuple

import gymnasium as gym
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from omegaconf import DictConfig
from rl_exercises.week_6.ppo import PPOAgent, set_seed
from rl_exercises.week_7.rnd_utils import (
    DualHeadValueNetwork,
    PredictorNetwork,
    RewardForwardFilter,
    TargetNetwork,
)
from stable_baselines3.common.running_mean_std import RunningMeanStd
from torch.distributions import Categorical

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class NovelDPPOAgent(PPOAgent):
    """
    PPO agent with NovelD exploration criterion.
    """

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
        rnd_hidden_size: int = 128,
        rnd_n_layers: int = 2,
        noveld_alpha: float = 1.0,
        combined_lr: float = 1e-4,
        update_proportion: float = 0.25,
        int_coef: float = 1.0,
        ext_coef: float = 2.0,
        int_gamma: float = 0.99,
        num_iterations_obs_norm_init: int = 50,
    ) -> None:
        super().__init__(
            env,
            lr_actor=lr_actor,
            lr_critic=lr_critic,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_eps=clip_eps,
            epochs=epochs,
            batch_size=batch_size,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            seed=seed,
            hidden_size=hidden_size,
        )

        self.noveld_alpha = noveld_alpha
        self.update_proportion = update_proportion
        self.int_coef = int_coef
        self.ext_coef = ext_coef
        self.int_gamma = int_gamma
        self.num_iterations_obs_norm_init = num_iterations_obs_norm_init

        obs_dim = env.observation_space.shape[0]

        self.value_fn = DualHeadValueNetwork(obs_dim, hidden_size)

        output_dim = rnd_hidden_size

        self.target_rnd = TargetNetwork(
            obs_dim,
            output_dim,
            hidden_dim=rnd_hidden_size,
            n_layers=rnd_n_layers,
        )

        self.predictor_rnd = PredictorNetwork(
            obs_dim,
            output_dim,
            hidden_dim=rnd_hidden_size,
            n_layers=rnd_n_layers,
        )

        for param in self.target_rnd.parameters():
            param.requires_grad = False

        combined_parameters = (
            list(self.policy.parameters())
            + list(self.value_fn.parameters())
            + list(self.predictor_rnd.parameters())
        )

        self.optimizer = optim.Adam(combined_parameters, lr=combined_lr)

        self.obs_rms = RunningMeanStd(shape=(obs_dim,))
        self.reward_rms = RunningMeanStd()
        self.discounted_reward = RewardForwardFilter(self.int_gamma)

        self._episode_visited: Set[bytes] = set()

    def _normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """Normalize a single observation using running statistics."""
        obs_norm = (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + 1e-8)
        return obs_norm.astype(np.float32)

    def _rnd_error(self, obs_norm: np.ndarray) -> float:
        """Compute raw RND prediction error for a single normalized observation."""
        obs_tensor = torch.from_numpy(obs_norm).float().unsqueeze(0)

        with torch.no_grad():
            target_emb = self.target_rnd(obs_tensor)
            predictor_emb = self.predictor_rnd(obs_tensor)
            error = torch.mean((predictor_emb - target_emb) ** 2)

        return float(error.item())

    def _is_first_visit(self, obs: np.ndarray) -> bool:
        """
        Return True if this is the first visit to obs in the current episode.
        """
        key = np.round(obs, decimals=2).tobytes()

        if key in self._episode_visited:
            return False

        self._episode_visited.add(key)
        return True

    def get_noveld_bonus(
        self,
        state: np.ndarray,
        next_state: np.ndarray,
    ) -> float:
        """
        Compute the NovelD intrinsic reward for a transition.
        """

        if not self._is_first_visit(next_state):
            return 0.0

        novelty_next = self._rnd_error(next_state)
        novelty_curr = self._rnd_error(state)

        bonus = max(novelty_next - self.noveld_alpha * novelty_curr, 0.0)

        return float(bonus)

    def _init_obs_normalization(self) -> None:
        """
        Warm-up phase for observation and intrinsic reward normalization.
        """
        print(
            f"[Warmup] Initializing obs/reward normalization over "
            f"{self.num_iterations_obs_norm_init} episodes..."
        )

        for _ in range(self.num_iterations_obs_norm_init):
            self.env.reset(seed=self.seed)
            done = False
            prev_obs_norm = None

            while not done:
                action = self.env.action_space.sample()
                next_state, _, term, trunc, _ = self.env.step(action)
                done = term or trunc

                self.obs_rms.update(next_state[np.newaxis])
                next_obs_norm = self._normalize_obs(next_state)

                if prev_obs_norm is not None:
                    novelty_next = self._rnd_error(next_obs_norm)
                    novelty_curr = self._rnd_error(prev_obs_norm)
                    int_reward_raw = max(
                        novelty_next - self.noveld_alpha * novelty_curr,
                        0.0,
                    )
                else:
                    int_reward_raw = self._rnd_error(next_obs_norm)

                discounted = self.discounted_reward.update(np.array([int_reward_raw]))
                self.reward_rms.update(discounted)

                prev_obs_norm = next_obs_norm

        print("[Warmup] Done.")

    def predict(
        self, state: np.ndarray
    ) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict action and return log probability, entropy, and both value estimates.
        """
        state_tensor = torch.from_numpy(state).float()

        probs = self.policy(state_tensor).squeeze(0)
        dist = Categorical(probs=probs)

        action_tensor = dist.sample()
        action = int(action_tensor.item())

        value_ext, value_int = self.value_fn(state_tensor)

        return (
            action,
            dist.log_prob(action_tensor),
            dist.entropy(),
            value_ext.squeeze(),
            value_int.squeeze(),
        )

    def compute_gae(
        self,
        rewards_ext: List[float],
        rewards_int: List[float],
        values_ext: torch.Tensor,
        values_int: torch.Tensor,
        next_values_ext: torch.Tensor,
        next_values_int: torch.Tensor,
        dones: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute separate GAE for extrinsic and intrinsic reward streams.
        """
        rews_ext = torch.tensor(rewards_ext, dtype=torch.float32)
        rews_int = torch.tensor(rewards_int, dtype=torch.float32)

        values_ext = values_ext.detach().view(-1)
        values_int = values_int.detach().view(-1)
        next_values_ext = next_values_ext.detach().view(-1)
        next_values_int = next_values_int.detach().view(-1)
        dones = dones.detach().float().view(-1)

        deltas_ext = rews_ext + self.gamma * next_values_ext * (1 - dones) - values_ext

        deltas_int = rews_int + self.int_gamma * next_values_int - values_int

        advs_ext: List[torch.Tensor] = []
        gae_ext = 0.0

        for delta, done in zip(reversed(deltas_ext), reversed(dones)):
            gae_ext = delta + self.gamma * self.gae_lambda * (1 - done) * gae_ext
            advs_ext.insert(0, gae_ext)

        advs_ext_t = torch.stack(advs_ext)

        advs_int: List[torch.Tensor] = []
        gae_int = 0.0

        for delta in reversed(deltas_int):
            gae_int = delta + self.int_gamma * self.gae_lambda * gae_int
            advs_int.insert(0, gae_int)

        advs_int_t = torch.stack(advs_int)

        returns_ext = advs_ext_t + values_ext
        returns_int = advs_int_t + values_int

        combined_advs = self.ext_coef * advs_ext_t + self.int_coef * advs_int_t
        combined_advs = (combined_advs - combined_advs.mean()) / (
            combined_advs.std(unbiased=False) + 1e-8
        )

        return (
            combined_advs.detach(),
            advs_ext_t.detach(),
            advs_int_t.detach(),
            returns_ext.detach(),
            returns_int.detach(),
        )

    def update(self, trajectory: List[Any]) -> Tuple[float, float, float, float]:
        """
        Perform PPO + RND predictor update with dual-head value network.
        """
        states = torch.stack([torch.from_numpy(t[0]).float() for t in trajectory])
        actions = torch.tensor([t[1] for t in trajectory], dtype=torch.int64)
        old_logps = torch.stack([t[2] for t in trajectory]).detach()
        rewards_ext = [t[4] for t in trajectory]
        rewards_int = [t[5] for t in trajectory]
        dones = torch.tensor([t[6] for t in trajectory], dtype=torch.float32)
        next_states = torch.stack([torch.from_numpy(t[7]).float() for t in trajectory])

        with torch.no_grad():
            values_ext, values_int = self.value_fn(states)
            next_values_ext, next_values_int = self.value_fn(next_states)

        combined_advs, _, _, returns_ext, returns_int = self.compute_gae(
            rewards_ext,
            rewards_int,
            values_ext,
            values_int,
            next_values_ext,
            next_values_int,
            dones,
        )

        dataset = torch.utils.data.TensorDataset(
            states,
            actions,
            old_logps,
            combined_advs,
            returns_ext,
            returns_int,
        )

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

        last_policy_loss = torch.tensor(0.0)
        last_value_loss = torch.tensor(0.0)
        last_entropy_loss = torch.tensor(0.0)
        last_rnd_loss = torch.tensor(0.0)

        for _ in range(self.epochs):
            for b_states, b_actions, b_oldlogp, b_adv, b_ret_ext, b_ret_int in loader:
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

                value_preds_ext, value_preds_int = self.value_fn(b_states)

                value_loss_ext = F.mse_loss(value_preds_ext, b_ret_ext)
                value_loss_int = F.mse_loss(value_preds_int, b_ret_int)
                value_loss = value_loss_ext + value_loss_int

                entropy_loss = -dist.entropy().mean()

                with torch.no_grad():
                    target_emb = self.target_rnd(b_states)

                predictor_emb = self.predictor_rnd(b_states)
                rnd_errors = torch.mean((predictor_emb - target_emb) ** 2, dim=1)

                mask = (torch.rand_like(rnd_errors) < self.update_proportion).float()

                if mask.sum() > 0:
                    rnd_loss = torch.sum(rnd_errors * mask) / mask.sum()
                else:
                    rnd_loss = torch.mean(rnd_errors)

                loss = (
                    policy_loss
                    + self.vf_coef * value_loss
                    + self.ent_coef * entropy_loss
                    + rnd_loss
                )

                self.optimizer.zero_grad()
                loss.backward()

                nn.utils.clip_grad_norm_(
                    list(self.policy.parameters())
                    + list(self.value_fn.parameters())
                    + list(self.predictor_rnd.parameters()),
                    max_norm=0.5,
                )

                self.optimizer.step()

                last_policy_loss = policy_loss.detach()
                last_value_loss = value_loss.detach()
                last_entropy_loss = entropy_loss.detach()
                last_rnd_loss = rnd_loss.detach()

        return (
            float(last_policy_loss.item()),
            float(last_value_loss.item()),
            float(last_entropy_loss.item()),
            float(last_rnd_loss.item()),
        )

    def train(
        self,
        total_steps: int,
        eval_interval: int = 10000,
        eval_episodes: int = 5,
    ) -> None:
        """
        Run training loop with NovelD intrinsic exploration bonus.
        """
        eval_env = gym.make(self.env.spec.id)
        step_count = 0

        self._init_obs_normalization()

        while step_count < total_steps:
            state, _ = self.env.reset(seed=self.seed)
            done = False
            trajectory: List[Any] = []

            self._episode_visited = set()

            self.obs_rms.update(state[np.newaxis])
            prev_obs_norm = self._normalize_obs(state)

            while not done and step_count < total_steps:
                action, logp, entropy, _, _ = self.predict(state)
                next_state, ext_reward, term, trunc, _ = self.env.step(action)

                done = term or trunc

                self.obs_rms.update(next_state[np.newaxis])
                next_obs_norm = self._normalize_obs(next_state)

                int_reward_raw = self.get_noveld_bonus(prev_obs_norm, next_obs_norm)

                discounted = self.discounted_reward.update(np.array([int_reward_raw]))
                self.reward_rms.update(discounted)
                int_reward = int_reward_raw / np.sqrt(self.reward_rms.var + 1e-8)

                trajectory.append(
                    (
                        state,
                        action,
                        logp,
                        entropy,
                        float(ext_reward),
                        float(int_reward),
                        float(done),
                        next_state,
                    )
                )

                state = next_state
                prev_obs_norm = next_obs_norm
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

            policy_loss, value_loss, entropy_loss, rnd_loss = self.update(trajectory)

            total_return = sum(t[4] for t in trajectory)

            print(
                f"[Train] Step {step_count:6d} Return {total_return:5.1f} "
                f"Policy Loss {policy_loss:.3f} Value Loss {value_loss:.3f} "
                f"Entropy Loss {entropy_loss:.3f} RND Loss {rnd_loss:.3f}"
            )

        eval_env.close()
        print("Training complete.")

    def evaluate(
        self,
        eval_env: gym.Env,
        num_episodes: int = 10,
    ) -> Tuple[float, float]:
        """
        Evaluate the agent using only extrinsic rewards.
        """
        returns = []

        self.policy.eval()

        with torch.no_grad():
            for _ in range(num_episodes):
                state, _ = eval_env.reset(seed=self.seed)
                done = False
                total_r = 0.0

                while not done:
                    probs = self.policy(torch.from_numpy(state).float()).squeeze(0)
                    action = int(torch.argmax(probs).item())

                    state, reward, term, trunc, _ = eval_env.step(action)

                    done = term or trunc
                    total_r += float(reward)

                returns.append(total_r)

        self.policy.train()

        return float(np.mean(returns)), float(np.std(returns))


@hydra.main(
    config_path="../configs/agent/",
    config_name="noveid_ppo",
    version_base="1.1",
)
def main(cfg: DictConfig) -> None:
    """Main training entry point."""
    env = gym.make(cfg.env.name)
    set_seed(env, cfg.seed)

    agent = NovelDPPOAgent(
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
        rnd_hidden_size=cfg.rnd.hidden_size,
        rnd_n_layers=cfg.rnd.n_layers,
        noveld_alpha=cfg.noveld.alpha,
        combined_lr=cfg.noveld.combined_lr,
        update_proportion=cfg.rnd.update_proportion,
        int_coef=cfg.noveld.int_coef,
        ext_coef=cfg.noveld.ext_coef,
        int_gamma=cfg.noveld.int_gamma,
        num_iterations_obs_norm_init=cfg.noveld.num_iterations_obs_norm_init,
    )

    agent.train(
        cfg.train.total_steps,
        cfg.train.eval_interval,
        cfg.train.eval_episodes,
    )

    env.close()


if __name__ == "__main__":
    main()
