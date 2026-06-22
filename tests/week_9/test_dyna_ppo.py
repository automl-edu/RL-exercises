"""
Tests for DynamicsModel and DynaPPOAgent in dyna_ppo.py.

Verifies:
 - DynamicsModel forward pass produces correctly shaped outputs.
 - DynaPPOAgent inherits from PPOAgent.
 - store_real() fills and caps the replay buffer.
 - train_model() returns (0, 0) when buffer is too small, finite floats otherwise.
 - evaluate_model() returns correct keys and zeros when buffer is too small.
 - imagine_and_update() returns (0,0,0) when buffer empty, finite floats otherwise.
 - update() returns three finite floats.
 - get_step_statistics() returns correct keys and initial zero values.
 - use_model=False disables model attributes.
"""

import unittest

import gymnasium as gym
import numpy as np
import torch
from rl_exercises.week_6 import PPOAgent
from rl_exercises.week_9.dyna_ppo import DynamicsModel, DynaPPOAgent

STATE_DIM = 4
ACTION_DIM = 2


def make_agent(use_model=True, model_batch_size=4):
    env = gym.make("CartPole-v1")
    return DynaPPOAgent(
        env,
        use_model=use_model,
        model_lr=1e-3,
        model_epochs=1,
        model_batch_size=model_batch_size,
        imag_horizon=3,
        imag_batches=2,
        max_buffer_size=100,
        epochs=1,
        batch_size=4,
        seed=0,
    )


def fill_buffer(agent, n):
    """Add n synthetic transitions directly to the replay buffer."""
    s = np.zeros(STATE_DIM, dtype=np.float32)
    s2 = np.ones(STATE_DIM, dtype=np.float32) * 0.1
    for _ in range(n):
        agent.real_buffer.append((s, 0, 1.0, s2, False))


class TestDynamicsModel(unittest.TestCase):
    def setUp(self):
        self.model = DynamicsModel(STATE_DIM, ACTION_DIM, hidden=16)

    def test_forward_output_shapes(self):
        B = 8
        s = torch.zeros(B, STATE_DIM)
        a = torch.zeros(B, ACTION_DIM)
        delta_s, r_pred = self.model(s, a)
        self.assertEqual(delta_s.shape, (B, STATE_DIM))
        self.assertEqual(r_pred.shape, (B,))

    def test_forward_output_types(self):
        s = torch.zeros(1, STATE_DIM)
        a = torch.zeros(1, ACTION_DIM)
        delta_s, r_pred = self.model(s, a)
        self.assertIsInstance(delta_s, torch.Tensor)
        self.assertIsInstance(r_pred, torch.Tensor)


class TestDynaPPOAgent(unittest.TestCase):
    def setUp(self):
        self.agent = make_agent(use_model=True, model_batch_size=4)

    # -------------------------------------------------------------------------
    # Inheritance
    # -------------------------------------------------------------------------

    def test_inherits_from_ppo(self):
        self.assertIsInstance(self.agent, PPOAgent)

    # -------------------------------------------------------------------------
    # store_real()
    # -------------------------------------------------------------------------

    def test_store_real_fills_buffer(self):
        traj = self._make_traj(length=5)
        self.agent.store_real(traj)
        self.assertEqual(len(self.agent.real_buffer), 5)

    def test_store_real_caps_at_max_buffer_size(self):
        for _ in range(110):
            self.agent.store_real(self._make_traj(length=1))
        self.assertLessEqual(len(self.agent.real_buffer), self.agent.max_buffer_size)

    def test_store_real_noop_when_model_disabled(self):
        agent = make_agent(use_model=False)
        agent.store_real(self._make_traj(length=5))
        self.assertFalse(hasattr(agent, "real_buffer"))

    # -------------------------------------------------------------------------
    # train_model()
    # -------------------------------------------------------------------------

    def test_train_model_returns_zeros_when_buffer_small(self):
        state_loss, reward_loss = self.agent.train_model()
        self.assertEqual(state_loss, 0.0)
        self.assertEqual(reward_loss, 0.0)

    def test_train_model_returns_finite_losses(self):
        fill_buffer(self.agent, 10)
        state_loss, reward_loss = self.agent.train_model()
        self.assertIsInstance(state_loss, float)
        self.assertIsInstance(reward_loss, float)
        self.assertTrue(np.isfinite(state_loss))
        self.assertTrue(np.isfinite(reward_loss))

    # -------------------------------------------------------------------------
    # evaluate_model()
    # -------------------------------------------------------------------------

    def test_evaluate_model_returns_correct_keys(self):
        fill_buffer(self.agent, 10)
        metrics = self.agent.evaluate_model(num_samples=5)
        for key in ("state_mse", "reward_mse", "state_mae", "reward_mae"):
            self.assertIn(key, metrics)

    def test_evaluate_model_returns_zeros_when_buffer_small(self):
        metrics = self.agent.evaluate_model(num_samples=1000)
        self.assertEqual(metrics["state_mse"], 0.0)
        self.assertEqual(metrics["reward_mse"], 0.0)

    def test_evaluate_model_returns_finite_values(self):
        fill_buffer(self.agent, 20)
        metrics = self.agent.evaluate_model(num_samples=10)
        for key, val in metrics.items():
            self.assertTrue(np.isfinite(val), f"{key}={val} is not finite")

    # -------------------------------------------------------------------------
    # imagine_and_update()
    # -------------------------------------------------------------------------

    def test_imagine_and_update_returns_zeros_when_buffer_empty(self):
        self.assertEqual(self.agent.imagine_and_update(), (0.0, 0.0, 0.0))

    def test_imagine_and_update_returns_three_finite_floats(self):
        fill_buffer(self.agent, 10)
        losses = self.agent.imagine_and_update()
        self.assertEqual(len(losses), 3)
        for i, loss in enumerate(losses):
            self.assertIsInstance(loss, float, f"loss[{i}] is not a float")
            self.assertTrue(np.isfinite(loss), f"loss[{i}]={loss} is not finite")

    # -------------------------------------------------------------------------
    # update()
    # -------------------------------------------------------------------------

    def test_update_returns_three_finite_floats(self):
        traj = self._make_traj(length=8)
        losses = self.agent.update(traj)
        self.assertEqual(len(losses), 3)
        for i, loss in enumerate(losses):
            self.assertIsInstance(loss, float, f"loss[{i}] is not a float")
            self.assertTrue(np.isfinite(loss), f"loss[{i}]={loss} is not finite")

    # -------------------------------------------------------------------------
    # get_step_statistics()
    # -------------------------------------------------------------------------

    def test_get_step_statistics_has_required_keys(self):
        stats = self.agent.get_step_statistics()
        for key in (
            "real_steps",
            "imagination_steps",
            "total_steps",
            "total_episodes",
            "imagination_ratio",
        ):
            self.assertIn(key, stats)

    def test_get_step_statistics_initial_zeros(self):
        stats = self.agent.get_step_statistics()
        self.assertEqual(stats["real_steps"], 0)
        self.assertEqual(stats["imagination_steps"], 0)
        self.assertEqual(stats["imagination_ratio"], 0.0)

    # -------------------------------------------------------------------------
    # use_model=False
    # -------------------------------------------------------------------------

    def test_no_model_attributes_when_disabled(self):
        agent = make_agent(use_model=False)
        self.assertFalse(hasattr(agent, "model"))
        self.assertFalse(hasattr(agent, "real_buffer"))

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _make_traj(self, length=8):
        env = self.agent.env
        state, _ = env.reset(seed=0)
        traj = []
        for _ in range(length):
            action, logp, ent, val = self.agent.predict(state)
            next_state, reward, term, trunc, _ = env.step(action)
            done = float(term or trunc)
            traj.append((state, action, logp, ent, float(reward), done, next_state))
            state = next_state if not done else env.reset(seed=0)[0]
        return traj


if __name__ == "__main__":
    unittest.main()
