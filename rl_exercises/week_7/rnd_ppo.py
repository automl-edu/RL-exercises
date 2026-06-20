"""
On-policy Proximal Policy Optimization (PPO) with Random Network Distillation (RND) for exploration.
"""

from typing import Any, List, Tuple

import gymnasium as gym
import hydra
import numpy as np
import torch
import torch.nn.functional as F  # noqa: F401
import torch.optim as optim  # noqa: F401
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
torch.backends.cudnn.benchmark = True


class RNDPPOAgent(PPOAgent):
    """
    Proximal Policy Optimization (PPO) agent with Random Network Distillation (RND) for exploration.

    RND provides an intrinsic motivation bonus based on the prediction error of a frozen target network
    by a trainable predictor network.
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
        # RND parameters
        rnd_hidden_size: int = 128,
        combined_lr: float = 1e-4,
        rnd_update_freq: int = 4,
        rnd_n_layers: int = 2,
        rnd_reward_weight: float = 0.1,
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

        self.rnd_update_freq = rnd_update_freq
        self.rnd_reward_weight = rnd_reward_weight
        self.update_proportion = update_proportion
        self.int_coef = int_coef
        self.ext_coef = ext_coef
        self.int_gamma = int_gamma
        self.num_iterations_obs_norm_init = num_iterations_obs_norm_init

        obs_dim = env.observation_space.shape[0]

        self.value_fn = DualHeadValueNetwork(obs_dim, hidden_size)

        output_dim = rnd_hidden_size
        self.target_rnd = TargetNetwork(
            obs_dim, output_dim, hidden_dim=rnd_hidden_size, n_layers=rnd_n_layers
        )
        self.predictor_rnd = PredictorNetwork(
            obs_dim, output_dim, hidden_dim=rnd_hidden_size, n_layers=rnd_n_layers
        )

        # target network is frozen
        for param in self.target_rnd.parameters():
            param.requires_grad = False

        # TODO: Combined optimizer: policy + dual-head value + RND predictor
        combined_parameters = (
            list(self.policy.parameters())
            + list(self.value_fn.parameters())
            + list(self.predictor_rnd.parameters())
        )

        # TODO: Optimizer for combined_parameters with learning rate combined_lr (Adam)
        self.optimizer = ...

        # For normalization of observations and intrinsic rewards
        self.obs_rms = RunningMeanStd(shape=(obs_dim,))
        self.reward_rms = RunningMeanStd()
        self.discounted_reward = RewardForwardFilter(self.int_gamma)

        self.rnd_update_counter = 0

    def _init_obs_normalization(self) -> None:
        """
        Warm-up phase: collect random trajectories to initialize obs_rms
        and reward_rms before actual training begins.
        Runs for self.num_iterations_obs_norm_init episodes.
        """
        print(
            f"[Warmup] Initializing obs/reward normalization over "
            f"{self.num_iterations_obs_norm_init} episodes..."
        )

        for _ in range(self.num_iterations_obs_norm_init):
            self.env.reset(seed=self.seed)
            done = False

            while not done:
                # Random action — no learning, just collecting observations
                action = self.env.action_space.sample()
                next_state, _, term, trunc, _ = self.env.step(action)
                done = term or trunc

                # Update observation running stats
                self.obs_rms.update(next_state[np.newaxis])

                # Normalize obs and compute raw RND bonus
                obs_norm = (next_state - self.obs_rms.mean) / np.sqrt(
                    self.obs_rms.var + 1e-8
                )
                int_reward_raw = self.get_rnd_bonus(obs_norm.astype(np.float32))

                # Feed into RewardForwardFilter to build up reward_rms
                discounted = self.discounted_reward.update(np.array([int_reward_raw]))
                self.reward_rms.update(discounted)

        print("[Warmup] Done.")

    def predict(
        self, state: np.ndarray
    ) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict action and return log probability, entropy, and both value estimates.

        Parameters
        ----------
        state : np.ndarray
            Current state.

        Returns
        -------
        Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            Action, log probability, entropy, extrinsic value, intrinsic value.
        """
        t = torch.from_numpy(state).float()
        probs = self.policy(t).squeeze(0)
        dist = Categorical(probs)
        action = dist.sample().item()
        value_ext, value_int = self.value_fn(t)
        return (
            action,
            dist.log_prob(torch.tensor(action)),
            dist.entropy(),
            value_ext,
            value_int,
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

        Parameters
        ----------
        rewards_ext : List[float]
            Extrinsic rewards from the environment.
        rewards_int : List[float]
            Normalized intrinsic (RND) rewards.
        values_ext : torch.Tensor
            Extrinsic value estimates.
        values_int : torch.Tensor
            Intrinsic value estimates.
        next_values_ext : torch.Tensor
            Next state extrinsic values.
        next_values_int : torch.Tensor
            Next state intrinsic values.
        dones : torch.Tensor
            Done flags.

        Returns
        -------
        Tuple of: combined advantages, ext advantages, int advantages,
                  extrinsic returns, intrinsic returns.
        """
        # TODO: compute gae for both extrinsic and intrinsic streams separately
        # (Hint: extrinsic stream uses done mask; intrinsic stream is non-episodic — no done mask)
        rews_ext = torch.tensor(rewards_ext, dtype=torch.float32)
        rews_int = torch.tensor(rewards_int, dtype=torch.float32)

        deltas_ext = ...
        deltas_int = ...

        # GAE for extrinsic stream
        advs_ext: List[torch.Tensor] = []
        A = 0.0
        for delta, done in zip(reversed(deltas_ext), reversed(dones)):
            A = ...
            advs_ext.insert(0, A)
        advs_ext_t = torch.stack(advs_ext)

        # GAE for intrinsic stream (non-episodic: done mask not applied)
        advs_int: List[torch.Tensor] = []
        A = 0.0
        for delta in reversed(deltas_int):
            A = ...
            advs_int.insert(0, A)
        advs_int_t = torch.stack(advs_int)

        returns_ext = ...
        returns_int = ...

        # TODO: Combined advantages weighted by coefficients, then normalize
        combined_advs = ...

        return (
            combined_advs.detach(),
            advs_ext_t.detach(),
            advs_int_t.detach(),
            returns_ext,
            returns_int,
        )

    def get_rnd_bonus(self, state: np.ndarray) -> float:
        """
        Compute the RND bonus (intrinsic reward) for a given state.

        Parameters
        ----------
        state : np.ndarray
            The current state of the environment.

        Returns
        -------
        float
            The RND bonus for the state.
        """
        # TODO: extract current state as a tensor
        state_tensor = ...

        # TODO: compute MSE error between predictor and target embeddings as the bonus
        with torch.no_grad():
            target_emb = ...
            predictor_emb = ...
        error = ...

        # TODO: scale by reward weight and return
        bonus = ...
        return bonus

    def update(self, trajectory: List[Any]) -> Tuple[float, float, float, float]:
        """
        Perform PPO + RND predictor update with dual-head value network.

        Parameters
        ----------
        trajectory : List[Any]
            Trajectory of (state, action, logp, ent, ext_reward, int_reward, done, next_state).

        Returns
        -------
        Tuple[float, float, float, float]
            Policy loss, value loss, entropy loss, RND predictor loss.
        """
        states = torch.stack([torch.from_numpy(t[0]).float() for t in trajectory])
        actions = torch.tensor([t[1] for t in trajectory])
        old_logps = torch.stack([t[2] for t in trajectory]).detach()
        rewards_ext = [t[4] for t in trajectory]
        rewards_int = [t[5] for t in trajectory]
        dones = torch.tensor([t[6] for t in trajectory], dtype=torch.float32)
        next_states = torch.stack([torch.from_numpy(t[7]).float() for t in trajectory])

        # TODO: compute values and next values for both extrinsic and intrinsic streams without grad
        with torch.no_grad():
            values_ext, values_int = ...
            next_values_ext, next_values_int = ...

        # TODO: compute combined advantages and returns for extrinsic and intrinsic rewards
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
            states, actions, old_logps, combined_advs, returns_ext, returns_int
        )
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

        for _ in range(self.epochs):
            for b_states, b_actions, b_oldlogp, b_adv, b_ret_ext, b_ret_int in loader:
                # TODO: Policy loss (clipped PPO surrogate)
                probs = ...
                dist = ...
                new_logp = ...
                ratio = ...
                policy_loss = ...

                # TODO: Dual-head value loss (MSE for both ext and int heads)
                value_preds_ext, value_preds_int = ...
                value_loss = ...

                # TODO: Entropy loss
                entropy_loss = ...

                # TODO: RND predictor loss with update_proportion mask
                # (only a random subset of minibatch transitions updates the predictor)
                mask = ...
                rnd_loss = ...

                loss = (
                    policy_loss
                    + self.vf_coef * value_loss
                    + self.ent_coef * entropy_loss
                    + rnd_loss
                )
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        return (
            policy_loss.item(),
            value_loss.item(),
            entropy_loss.item(),
            rnd_loss.item(),
        )

    def train(
        self,
        total_steps: int,
        eval_interval: int = 10000,
        eval_episodes: int = 5,
    ) -> None:
        """
        Run a training loop for a fixed number of environment steps with RND exploration bonus.

        Parameters
        ----------
        total_steps : int
            Total environment steps to train for.
        eval_interval : int
            Every this many steps, evaluate the agent.
        eval_episodes : int
            Number of evaluation episodes.
        """
        eval_env = gym.make(self.env.spec.id)
        step_count = 0
        self.rnd_update_counter = 0

        # Warm-up phase for intrinsic reward normalization
        self._init_obs_normalization()

        while step_count < total_steps:
            state, _ = self.env.reset(seed=self.seed)
            done = False
            trajectory: List[Any] = []

            while not done and step_count < total_steps:
                action, logp, entropy, _, _ = self.predict(state)
                next_state, ext_reward, term, trunc, _ = self.env.step(action)
                done = term or trunc

                # --- Observation normalization ---
                self.obs_rms.update(next_state[np.newaxis])
                obs_norm = (next_state - self.obs_rms.mean) / np.sqrt(
                    self.obs_rms.var + 1e-8
                )

                # TODO: --- Intrinsic reward (RND bonus on normalized obs) ---
                int_reward_raw = self.get_rnd_bonus(obs_norm.astype(np.float32))

                # --- Normalize intrinsic reward via RewardForwardFilter + RunningMeanStd ---
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
                step_count += 1
                self.rnd_update_counter += 1

                if step_count % eval_interval == 0:
                    mean_r, std_r = self.evaluate(eval_env, num_episodes=eval_episodes)
                    print(
                        f"[Eval ] Step {step_count:6d} AvgReturn {mean_r:5.1f} ± {std_r:4.1f}"
                    )

            # PPO + RND update
            policy_loss, value_loss, entropy_loss, rnd_loss = self.update(trajectory)

            total_return = sum(t[4] for t in trajectory)
            print(
                f"[Train] Step {step_count:6d} Return {total_return:5.1f} "
                f"Policy Loss {policy_loss:.3f} Value Loss {value_loss:.3f} "
                f"Entropy Loss {entropy_loss:.3f} RND Loss {rnd_loss:.3f}"
            )

        print("Training complete.")

    def evaluate(
        self, eval_env: gym.Env, num_episodes: int = 10
    ) -> Tuple[float, float]:
        """
        Evaluate the agent without RND bonus (only extrinsic reward).

        Parameters
        ----------
        eval_env : gym.Env
            The evaluation environment.
        num_episodes : int
            Number of evaluation episodes.

        Returns
        -------
        Tuple[float, float]
            Mean and standard deviation of returns.
        """
        returns = []
        for _ in range(num_episodes):
            state, _ = eval_env.reset(seed=self.seed)
            done = False
            total_r = 0.0
            while not done:
                action, _, _, _, _ = self.predict(state)
                state, r, term, trunc, _ = eval_env.step(action)
                done = term or trunc
                total_r += r
            returns.append(total_r)
        return float(np.mean(returns)), float(np.std(returns))


@hydra.main(config_path="../configs/agent/", config_name="rnd_ppo", version_base="1.1")
def main(cfg: DictConfig) -> None:
    """Main training entry point."""
    env = gym.make(cfg.env.name)
    set_seed(env, cfg.seed)
    agent = RNDPPOAgent(
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
        rnd_update_freq=cfg.rnd.update_freq,
        rnd_n_layers=cfg.rnd.n_layers,
        rnd_reward_weight=cfg.rnd.reward_weight,
    )
    agent.train(
        cfg.train.total_steps,
        cfg.train.eval_interval,
        cfg.train.eval_episodes,
    )


if __name__ == "__main__":
    main()
