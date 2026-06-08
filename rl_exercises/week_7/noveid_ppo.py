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


class NovelDPPOAgent(PPOAgent):
    """
    PPO agent with NovelD exploration criterion.

    NovelD computes intrinsic rewards as the *increase* in novelty when
    transitioning from s_t to s_{t+1}, rather than the absolute novelty of
    s_{t+1}.

    Intrinsic reward formula (per step):
        r_int = max(novelty(s_{t+1}) - alpha * novelty(s_t), 0)
                * first_visit_indicator(s_{t+1})

    where novelty(s) is the RND prediction error at state s.
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
        """
        Initialize NovelD PPO agent.

        Parameters
        ----------
        env : gym.Env
            The Gym environment.
        lr_actor : float
            Learning rate for actor network.
        lr_critic : float
            Learning rate for critic network.
        gamma : float
            Discount factor for extrinsic rewards.
        gae_lambda : float
            GAE lambda parameter.
        clip_eps : float
            PPO clipping epsilon.
        epochs : int
            Number of training epochs per update.
        batch_size : int
            Mini-batch size for updates.
        ent_coef : float
            Entropy coefficient.
        vf_coef : float
            Value function loss coefficient.
        seed : int
            RNG seed.
        hidden_size : int
            Hidden size for policy and value networks.
        rnd_hidden_size : int
            Hidden size for RND networks.
        rnd_n_layers : int
            Number of hidden layers in RND networks.
        noveld_alpha : float
            The alpha scaling factor for novelty(s_t)
        combined_lr : float
            Learning rate for the combined optimizer.
        update_proportion : float
            Fraction of minibatch transitions used to update the RND predictor.
        int_coef : float
            Coefficient for intrinsic advantages in combined advantage.
        ext_coef : float
            Coefficient for extrinsic advantages in combined advantage.
        int_gamma : float
            Discount factor for the intrinsic reward stream (non-episodic).
        num_iterations_obs_norm_init : int
            Number of random-action episodes used in the warm-up phase to
            initialize obs_rms and reward_rms before training begins.
        """
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

        # Replace parent's single-head value network with a dual-head one
        self.value_fn = DualHeadValueNetwork(obs_dim, hidden_size)

        # RND backbone: frozen target + trainable predictor
        output_dim = rnd_hidden_size
        self.target_rnd = TargetNetwork(
            obs_dim, output_dim, hidden_dim=rnd_hidden_size, n_layers=rnd_n_layers
        )
        self.predictor_rnd = PredictorNetwork(
            obs_dim, output_dim, hidden_dim=rnd_hidden_size, n_layers=rnd_n_layers
        )

        # Target is frozen
        for param in self.target_rnd.parameters():
            param.requires_grad = False

        # TODO: Single combined optimizer: policy + dual-head value + RND predictor
        combined_parameters = (
            list(self.policy.parameters())
            + list(self.value_fn.parameters())
            + list(self.predictor_rnd.parameters())
        )

        # TODO: Optimizer for combined_parameters with learning rate combined_lr (Adam)
        self.optimizer = ...

        # Running statistics for observation and intrinsic reward normalization
        self.obs_rms = RunningMeanStd(shape=(obs_dim,))
        self.reward_rms = RunningMeanStd()
        self.discounted_reward = RewardForwardFilter(self.int_gamma)

        # Per-episode first-visit tracker (set of hashed observations)
        # Reset at the start of each episode in train()
        self._episode_visited: Set[bytes] = set()

    def _normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """Normalize a single observation using running statistics."""
        obs_norm = (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + 1e-8)
        return obs_norm.astype(np.float32)

    def _rnd_error(self, obs_norm: np.ndarray) -> float:
        """Compute raw RND prediction error for a single normalized observation."""
        t = torch.from_numpy(obs_norm).float().unsqueeze(0)
        # TODO: Compute RND prediction error (MSE) between predictor and target embeddings
        with torch.no_grad():
            target_emb = ...
            predictor_emb = ...
        return ...

    def _is_first_visit(self, obs: np.ndarray) -> bool:
        """
        Return True if this is the first visit to `obs` in the current episode.

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
        Compute the NovelD intrinsic reward for a (s_t, s_{t+1}) transition.

        The bonus is:
            r_int = max(novelty(s_{t+1}) - alpha * novelty(s_t), 0)
                    * first_visit_indicator(s_{t+1})

        Parameters
        ----------
        state : np.ndarray
            Current state (already normalized).
        next_state : np.ndarray
            Next state (already normalized).

        Returns
        -------
        float
            NovelD intrinsic reward (0 if not first visit).
        """
        # TODO: compute NovelD bonus using the formula above
        # Hint: use self._rnd_error() for novelty and self._is_first_visit() for the first-visit indicator
        if not self._is_first_visit(next_state):
            return 0.0
        novelty_next = ...
        novelty_curr = ...
        bonus = ...
        return bonus

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
            prev_obs_norm = None

            while not done:
                action = self.env.action_space.sample()
                next_state, _, term, trunc, _ = self.env.step(action)
                done = term or trunc

                # Update obs running stats
                self.obs_rms.update(next_state[np.newaxis])
                next_obs_norm = self._normalize_obs(next_state)

                # Compute raw NovelD bonus (needs both current and next normalized obs)
                if prev_obs_norm is not None:
                    novelty_next = self._rnd_error(next_obs_norm)
                    novelty_curr = self._rnd_error(prev_obs_norm)
                    int_reward_raw = max(
                        novelty_next - self.noveld_alpha * novelty_curr, 0.0
                    )
                else:
                    # First step of episode: no previous obs to compare against
                    int_reward_raw = self._rnd_error(next_obs_norm)

                # Update reward running stats via RewardForwardFilter
                discounted = self.discounted_reward.update(np.array([int_reward_raw]))
                self.reward_rms.update(discounted)

                prev_obs_norm = next_obs_norm

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
            Normalized NovelD intrinsic rewards.
        values_ext : torch.Tensor
            Extrinsic value estimates.
        values_int : torch.Tensor
            Intrinsic value estimates.
        next_values_ext : torch.Tensor
            Next-state extrinsic value estimates.
        next_values_int : torch.Tensor
            Next-state intrinsic value estimates.
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

        # GAE for extrinsic stream (episodic: done mask applied)
        advs_ext: List[torch.Tensor] = []
        A = 0.0
        for delta, done in zip(reversed(deltas_ext), reversed(dones)):
            A = ...
            advs_ext.insert(0, A)
        advs_ext_t = torch.stack(advs_ext)

        # GAE for intrinsic stream (non-episodic: done mask NOT applied)
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

    def update(self, trajectory: List[Any]) -> Tuple[float, float, float, float]:
        """
        Perform PPO + RND predictor update with dual-head value network.

        Parameters
        ----------
        trajectory : List[Any]
            Each entry: (state, action, logp, entropy, ext_reward,
                         int_reward, done, next_state).

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
                # TODO: --- Policy loss (clipped PPO surrogate) ---
                probs = ...
                dist = ...
                new_logp = ...
                ratio = ...
                policy_loss = ...

                # TODO: --- Dual-head value loss (MSE for both ext and int heads) ---
                value_preds_ext, value_preds_int = ...
                value_loss = ...

                # TODO: --- Entropy loss ---
                entropy_loss = ...

                # TODO: --- RND predictor loss (update_proportion mask) ---
                # Only a random subset of the minibatch is used to update the predictor
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
                nn.utils.clip_grad_norm_(
                    list(self.policy.parameters())
                    + list(self.value_fn.parameters())
                    + list(self.predictor_rnd.parameters()),
                    max_norm=0.5,
                )
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
        Run training loop with NovelD intrinsic exploration bonus.

        Parameters
        ----------
        total_steps : int
            Total environment steps to train for.
        eval_interval : int
            Every this many steps, run evaluation.
        eval_episodes : int
            Number of episodes per evaluation.
        """
        eval_env = gym.make(self.env.spec.id)
        step_count = 0

        # Warm-up: initialize obs_rms and reward_rms before policy updates start
        self._init_obs_normalization()

        while step_count < total_steps:
            state, _ = self.env.reset(seed=self.seed)
            done = False
            trajectory: List[Any] = []

            # Reset the per-episode first-visit tracker at the start of each episode
            self._episode_visited = set()

            # Normalize and cache the initial state's RND error for the first step
            self.obs_rms.update(state[np.newaxis])
            prev_obs_norm = self._normalize_obs(state)

            while not done and step_count < total_steps:
                action, logp, entropy, _, _ = self.predict(state)
                next_state, ext_reward, term, trunc, _ = self.env.step(action)
                done = term or trunc

                # --- Observation normalization ---
                self.obs_rms.update(next_state[np.newaxis])
                next_obs_norm = self._normalize_obs(next_state)

                # TODO: --- NovelD intrinsic reward ---
                # max(novelty(s_{t+1}) - alpha * novelty(s_t), 0) * first_visit(s_{t+1})
                int_reward_raw = self.get_noveld_bonus(prev_obs_norm, next_obs_norm)

                # --- Normalize intrinsic reward scale ---
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
                    mean_r, std_r = self.evaluate(eval_env, num_episodes=eval_episodes)
                    print(
                        f"[Eval ] Step {step_count:6d} AvgReturn {mean_r:5.1f} ± {std_r:4.1f}"
                    )

            # PPO + RND predictor update
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
        Evaluate the agent using only extrinsic rewards (no exploration bonus).

        Parameters
        ----------
        eval_env : gym.Env
            The evaluation environment.
        num_episodes : int
            Number of evaluation episodes.

        Returns
        -------
        Tuple[float, float]
            Mean and standard deviation of episode returns.
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


@hydra.main(
    config_path="../configs/agent/", config_name="noveid_ppo", version_base="1.1"
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


if __name__ == "__main__":
    main()
