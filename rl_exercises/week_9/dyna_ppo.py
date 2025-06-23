"""
Dyna-PPO Agent Implementation

This module defines a Dyna-style extension of the PPO algorithm that:
 1. Optionally learns a dynamics model from real environment transitions.
 2. Uses the model to generate imagined rollouts for additional policy updates.
 3. Falls back to base PPO logic if model-based learning is disabled, but maintains unified logging.
 4. Tracks real and imagination steps separately for better monitoring.
 5. Provides model evaluation and save/load functionality.

Classes:
    DynamicsModel: Neural network predicting next-state deltas and rewards.
    DynaPPOAgent: PPOAgent subclass integrating optional model learning and imagination.

Usage:
    python dyna_ppo.py use_model=True
"""

from typing import Any, Dict, List, Tuple

import os
import random

import gymnasium as gym
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from omegaconf import DictConfig
from rl_exercises.week_6.ppo import PPOAgent, set_seed


class DynamicsModel(nn.Module):
    """
    Dynamics model predicting next-state deltas and rewards given state and action.

    Args:
        state_dim (int): Dimension of the flattened state vector.
        action_dim (int): Number of discrete actions.
        hidden (int): Size of the hidden layer.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden)
        self.fc_s = nn.Linear(hidden, state_dim)
        self.fc_r = nn.Linear(hidden, 1)

    def forward(
        self, s: torch.Tensor, a_onehot: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the dynamics model.

        Args:
            s (torch.Tensor): Batch of states, shape [B, state_dim].
            a_onehot (torch.Tensor): Batch of one-hot actions, shape [B, action_dim].

        Returns:
            delta_s (torch.Tensor): Predicted change in state, shape [B, state_dim].
            r_pred (torch.Tensor): Predicted reward, shape [B].
        """
        x = torch.cat([s, a_onehot], dim=-1)
        h = torch.relu(self.fc1(x))
        delta_s = self.fc_s(h)
        r_pred = self.fc_r(h).squeeze(-1)
        return delta_s, r_pred


class DynaPPOAgent(PPOAgent):
    """
    Dyna-style PPO agent with optional model-based imagination and unified logging.

    Args:
        env (gym.Env): Gym environment.
        use_model (bool): If True, enables model learning and imagined rollouts.
        model_lr (float): Learning rate for the dynamics model.
        model_epochs (int): Number of epochs to train the model per update.
        model_batch_size (int): Minibatch size for model training.
        imag_horizon (int): Length of imagined rollout trajectories.
        imag_batches (int): Number of imagined trajectories per real update.
        max_buffer_size (int): Maximum size of the real transition buffer.
        **ppo_kwargs: Passed through to the PPOAgent constructor.
    """

    def __init__(
        self,
        env: gym.Env,
        use_model: bool = True,
        model_lr: float = 1e-3,
        model_epochs: int = 5,
        model_batch_size: int = 64,
        imag_horizon: int = 5,
        imag_batches: int = 20,
        max_buffer_size: int = 100000,
        **ppo_kwargs,
    ):
        super().__init__(env, **ppo_kwargs)
        self.use_model = use_model

        # Step tracking
        self.real_steps = 0
        self.imagination_steps = 0
        self.total_episodes = 0

        if self.use_model:
            # Initialize dynamics model and optimizer
            obs_dim = int(np.prod(env.observation_space.shape))
            act_dim = env.action_space.n
            self.model = DynamicsModel(
                obs_dim, act_dim, hidden=ppo_kwargs.get("hidden_size", 128)
            )
            self.model_opt = optim.Adam(self.model.parameters(), lr=model_lr)

            # Hyperparameters for model learning and imagination
            self.model_epochs = model_epochs
            self.model_batch_size = model_batch_size
            self.imag_horizon = imag_horizon
            self.imag_batches = imag_batches

            # Replay buffer for real transitions
            self.real_buffer: List[Tuple[np.ndarray, int, float, np.ndarray, bool]] = []
            self.max_buffer_size = max_buffer_size

    def store_real(self, traj: List[Any]) -> None:
        """
        Store real environment transitions into the replay buffer.

        Args:
            traj (List): List of transitions (s, a, logp, ent, r, done, s_next).
        """
        if not self.use_model:
            return
        for s, a, _, _, r, done, s2 in traj:
            self.real_buffer.append((s, a, r, s2, done))
        if len(self.real_buffer) > self.max_buffer_size:
            self.real_buffer = self.real_buffer[-self.max_buffer_size :]

    def train_model(self) -> Tuple[float, float]:
        """
        Train the dynamics model on a minibatch of real transitions.

        Returns:
            Tuple of (state_loss, reward_loss) for logging
        """
        if not self.use_model or len(self.real_buffer) < self.model_batch_size:
            return 0.0, 0.0

        total_state_loss = 0.0
        total_reward_loss = 0.0

        # TODO: Loop over multiple epochs for model training
        for _ in range(self.model_epochs):
            # Sample a minibatch of real transitions (s, a, r, s')
            batch = random.sample(self.real_buffer, self.model_batch_size)
            states, actions, rewards, next_states, _ = zip(*batch)

            # TODO: Predict next state delta and reward using the model
            # TODO: Compute loss for state prediction and reward prediction
            loss_s = ...  # Placeholder for state loss
            loss_r = ...  # Placeholder for reward loss
            loss = ...  # Placeholder for total loss

            self.model_opt.zero_grad()
            loss.backward()
            # Add gradient clipping for model training
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.model_opt.step()

            total_state_loss += loss_s.item()
            total_reward_loss += loss_r.item()

        return (
            total_state_loss / self.model_epochs,
            total_reward_loss / self.model_epochs,
        )

    def evaluate_model(self, num_samples: int = 1000) -> Dict[str, float]:
        """
        Evaluate the dynamics model accuracy on a validation set.

        Args:
            num_samples (int): Number of samples to use for evaluation.

        Returns:
            Dict containing model evaluation metrics.
        """
        if not self.use_model or len(self.real_buffer) < num_samples:
            return {
                "state_mae": 0.0,
                "reward_mae": 0.0,
                "state_mse": 0.0,
                "reward_mse": 0.0,
            }

        # TODO: Sample a batch of transitions from the replay buffer
        val_batch = ...
        states, actions, rewards, next_states, _ = zip(*val_batch)

        # TODO: Compute MSE (L2) and MAE (L1) for both state and reward predictions
        with torch.no_grad():
            # Calculate metrics
            state_mse = ...
            reward_mse = ...
            state_mae = ...
            reward_mae = ...

        return {
            "state_mse": state_mse,
            "reward_mse": reward_mse,
            "state_mae": state_mae,
            "reward_mae": reward_mae,
        }

    def imagine_and_update(self) -> Tuple[float, float, float]:
        """
        Generate imagined rollouts from the learned model and perform PPO updates.

        Returns:
            Tuple of (policy_loss, value_loss, entropy_loss) from imagined data
        """
        if not self.use_model or not self.real_buffer:
            return 0.0, 0.0, 0.0

        # Collect all imagined trajectories
        all_imag_trajs = []
        imag_steps_count = 0

        for _ in range(self.imag_batches):
            # Start from a random real state
            s = random.choice(self.real_buffer)[0]
            imag_traj: List[Any] = []

            # TODO: Simulate a trajectory using the model
            for step in range(self.imag_horizon):
                # TODO: Predict action, log-probability, entropy, and value from the PPO policy
                action, logp, ent, val = ...

                # TODO: Prepare model input tensors
                a_oh = (  # noqa: F841
                    F.one_hot(torch.tensor(...), self.env.action_space.n)
                    .unsqueeze(0)
                    .float()
                )  # noqa: F841
                s_t = torch.tensor(..., dtype=torch.float32).unsqueeze(0)  # noqa: F841

                # TODO: Predict next state delta and reward
                with torch.no_grad():  # Don't track gradients during imagination
                    delta, r_pred = ...
                    s2 = ...
                    r_val = ...

                # Add some termination probability to make rollouts more realistic
                done_prob = 0.05  # 5% chance of termination per step
                done_flag = random.random() < done_prob or step == self.imag_horizon - 1

                imag_traj.append((s, action, logp, ent, r_val, float(done_flag), s2))
                imag_steps_count += 1

                if done_flag:
                    break

                s = s2

            # Only add non-empty trajectories
            if imag_traj:
                all_imag_trajs.extend(imag_traj)

        # Update imagination step counter
        self.imagination_steps += imag_steps_count

        # Perform PPO update on all imagined data using parent class method
        if all_imag_trajs:
            return super().update(all_imag_trajs)
        else:
            return 0.0, 0.0, 0.0

    def update(self, trajectory: List[Any]) -> Tuple[float, float, float]:
        """
        Override update to add gradient clipping for stability.
        """
        # Store original optimizer state
        old_zero_grad = self.optimizer.zero_grad  # noqa: F841
        old_step = self.optimizer.step

        def clipped_step():
            # Add gradient clipping before stepping
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            torch.nn.utils.clip_grad_norm_(self.value_fn.parameters(), max_norm=0.5)
            old_step()

        # Temporarily replace step method
        self.optimizer.step = clipped_step

        try:
            # Call parent update method
            result = super().update(trajectory)
        finally:
            # Restore original step method
            self.optimizer.step = old_step

        return result

    def save_checkpoint(self, filepath: str) -> None:
        """
        Save agent and model weights to a checkpoint file.

        Args:
            filepath (str): Path to save the checkpoint.
        """
        checkpoint = {
            "policy_state_dict": self.policy.state_dict(),
            "value_fn_state_dict": self.value_fn.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "real_steps": self.real_steps,
            "imagination_steps": self.imagination_steps,
            "total_episodes": self.total_episodes,
            "use_model": self.use_model,
        }

        if self.use_model:
            checkpoint.update(
                {
                    "model_state_dict": self.model.state_dict(),
                    "model_optimizer_state_dict": self.model_opt.state_dict(),
                    "real_buffer": self.real_buffer,
                }
            )

        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath: str, load_buffer: bool = True) -> None:
        """
        Load agent and model weights from a checkpoint file.

        Args:
            filepath (str): Path to the checkpoint file.
            load_buffer (bool): Whether to load the replay buffer.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

        checkpoint = torch.load(filepath, map_location="cpu")

        # Load policy and value function
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.value_fn.load_state_dict(checkpoint["value_fn_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Load step counters
        self.real_steps = checkpoint.get("real_steps", 0)
        self.imagination_steps = checkpoint.get("imagination_steps", 0)
        self.total_episodes = checkpoint.get("total_episodes", 0)

        # Load model components if they exist and model is enabled
        if self.use_model and "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model_opt.load_state_dict(checkpoint["model_optimizer_state_dict"])

            if load_buffer and "real_buffer" in checkpoint:
                self.real_buffer = checkpoint["real_buffer"]

        print(f"Checkpoint loaded from {filepath}")
        print(
            f"Real steps: {self.real_steps}, Imagination steps: {self.imagination_steps}, Episodes: {self.total_episodes}"
        )

    def get_step_statistics(self) -> Dict[str, int]:
        """
        Get current step statistics.

        Returns:
            Dictionary with step counts and ratios.
        """
        total_steps = self.real_steps + self.imagination_steps
        stats = {
            "real_steps": self.real_steps,
            "imagination_steps": self.imagination_steps,
            "total_steps": total_steps,
            "total_episodes": self.total_episodes,
        }

        if total_steps > 0:
            stats["imagination_ratio"] = self.imagination_steps / total_steps
        else:
            stats["imagination_ratio"] = 0.0

        return stats

    def train(
        self,
        total_steps: int,
        eval_interval: int = 10000,
        eval_episodes: int = 5,
        model_eval_interval: int = 50000,
        save_interval: int = 100000,
        save_dir: str = "./checkpoints",
    ) -> None:
        """
        Main training loop for Dyna-PPO or base PPO if use_model is False.

        Maintains unified logging format including losses and step tracking.

        Args:
            total_steps (int): Total number of real environment steps to run.
            eval_interval (int): Steps between policy evaluations.
            eval_episodes (int): Number of episodes for each evaluation.
            model_eval_interval (int): Steps between model evaluations.
            save_interval (int): Steps between checkpoint saves.
            save_dir (str): Directory to save checkpoints.
        """
        # Always run custom loop to log losses uniformly
        eval_env = gym.make(self.env.spec.id)
        set_seed(eval_env, self.seed)

        while self.real_steps < total_steps:
            state, _ = self.env.reset()
            done = False
            real_traj: List[Any] = []
            episode_steps = 0

            # TODO: Collect one real trajectory (episode)
            while not done and self.real_steps < total_steps:
                action, logp, ent, val = ...
                next_state, reward, term, trunc, _ = ...
                done = term or trunc
                real_traj.append(
                    (state, action, logp, ent, reward, float(done), next_state)
                )
                state = next_state
                self.real_steps += 1
                episode_steps += 1

                # Evaluation
                if self.real_steps % eval_interval == 0:
                    mean_r, std_r = self.evaluate(eval_env, num_episodes=eval_episodes)
                    stats = self.get_step_statistics()
                    if self.use_model:
                        print(
                            f"[Eval ] Real Steps {self.real_steps:6d} (Total: {stats['total_steps']:6d}, "
                            f"Imag: {self.imagination_steps:6d}, Ratio: {stats['imagination_ratio']:.2f}) "
                            f"AvgReturn {mean_r:5.1f} ± {std_r:4.1f}"
                        )
                    else:
                        print(
                            f"[Eval ] Step {self.real_steps:6d} AvgReturn {mean_r:5.1f} ± {std_r:4.1f}"
                        )

                # Model evaluation
                if self.use_model and self.real_steps % model_eval_interval == 0:
                    model_metrics = self.evaluate_model()
                    print(
                        f"[Model] Step {self.real_steps:6d} State MSE: {model_metrics['state_mse']:.4f}, "
                        f"Reward MSE: {model_metrics['reward_mse']:.4f}"
                    )

                # Save checkpoint
                if self.real_steps % save_interval == 0:
                    save_path = os.path.join(
                        save_dir, f"checkpoint_step_{self.real_steps}.pt"
                    )
                    self.save_checkpoint(save_path)

            self.total_episodes += 1

            # TODO: Perform PPO update on real transitions
            policy_loss, value_loss, entropy_loss = ...
            last_return = sum(r for *_, r, _, _ in real_traj)

            # 2) Model-based steps if enabled
            model_state_loss, model_reward_loss = 0.0, 0.0
            imag_policy_loss, imag_value_loss, imag_entropy_loss = 0.0, 0.0, 0.0

            # TODO: If using model, train it and perform imagined updates
            if self.use_model:
                self.store_real(real_traj)
                model_state_loss, model_reward_loss = ...
                imag_policy_loss, imag_value_loss, imag_entropy_loss = ...

            # Unified logging with step tracking
            stats = self.get_step_statistics()
            if self.use_model:
                print(
                    f"[Train] Real Steps {self.real_steps:6d} (Ep: {self.total_episodes:4d}, "
                    f"Total: {stats['total_steps']:6d}, Imag: {self.imagination_steps:6d}) "
                    f"Return {last_return:5.1f} "
                    f"Policy Loss {policy_loss:.3f} Value Loss {value_loss:.3f} Entropy Loss {entropy_loss:.3f} "
                    f"Model S-Loss {model_state_loss:.3f} R-Loss {model_reward_loss:.3f} "
                    f"Imag P-Loss {imag_policy_loss:.3f} V-Loss {imag_value_loss:.3f} E-Loss {imag_entropy_loss:.3f}"
                )
            else:
                print(
                    f"[Train] Step {self.real_steps:6d} (Ep: {self.total_episodes:4d}) "
                    f"Return {last_return:5.1f} "
                    f"Policy Loss {policy_loss:.3f} Value Loss {value_loss:.3f} Entropy Loss {entropy_loss:.3f}"
                )

        # Final checkpoint save
        final_save_path = os.path.join(save_dir, "final_checkpoint.pt")
        self.save_checkpoint(final_save_path)

        print("Dyna-PPO training complete.")
        final_stats = self.get_step_statistics()
        print(
            f"Final statistics: Real steps: {final_stats['real_steps']}, "
            f"Imagination steps: {final_stats['imagination_steps']}, "
            f"Total episodes: {final_stats['total_episodes']}"
        )


@hydra.main(config_path="../configs/agent/", config_name="dyna_ppo", version_base="1.1")
def main(cfg: DictConfig) -> None:
    env = gym.make(cfg.env.name)
    set_seed(env, cfg.seed)

    agent = DynaPPOAgent(
        env,
        use_model=cfg.agent.use_model,
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
        # Dyna-specific
        model_lr=cfg.agent.model_lr,
        model_epochs=cfg.agent.model_epochs,
        model_batch_size=cfg.agent.model_batch_size,
        imag_horizon=cfg.agent.imag_horizon,
        imag_batches=cfg.agent.imag_batches,
        max_buffer_size=cfg.agent.max_buffer_size,
    )

    # Load checkpoint if specified
    if hasattr(cfg, "checkpoint_path") and cfg.checkpoint_path:
        agent.load_checkpoint(cfg.checkpoint_path)

    agent.train(
        cfg.train.total_steps,
        cfg.train.eval_interval,
        cfg.train.eval_episodes,
        cfg.train.get("model_eval_interval", 50000),
        cfg.train.get("save_interval", 100000),
        cfg.train.get("save_dir", "./checkpoints"),
    )


if __name__ == "__main__":
    main()
