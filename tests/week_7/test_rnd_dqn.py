"""
Tests for RNDDQNAgent in rnd_dqn.py.

Verifies:
 - Inherits from DQNAgent.
 - get_rnd_bonus returns a non-negative float and scales with rnd_reward_weight.
 - update_rnd returns a finite non-negative MSE loss.
 - update_rnd updates predictor weights but leaves target weights unchanged.
 - Short training loop completes without error.
"""

import unittest

import gymnasium as gym
import numpy as np
import torch
from rl_exercises.week_4 import DQNAgent
from rl_exercises.week_7.rnd_dqn import RNDDQNAgent


class TestRNDDQNAgent(unittest.TestCase):
    def setUp(self):
        self.env = gym.make("CartPole-v1")
        self.agent = RNDDQNAgent(
            self.env,
            buffer_capacity=50,
            batch_size=8,
            lr=1e-3,
            rnd_hidden_size=16,
            rnd_lr=1e-3,
            rnd_update_freq=10,
            rnd_n_layers=1,
            rnd_reward_weight=0.1,
            seed=0,
        )

    def _fill_buffer(self):
        obs, _ = self.env.reset(seed=0)
        for _ in range(self.agent.batch_size):
            action = self.env.action_space.sample()
            ns, r, d, tr, _ = self.env.step(action)
            self.agent.buffer.add(obs, action, r, ns, d or tr, {})
            obs = ns if not (d or tr) else self.env.reset(seed=0)[0]

    def test_inherits_from_dqn(self):
        self.assertIsInstance(self.agent, DQNAgent)

    def test_get_rnd_bonus_is_nonneg_float(self):
        state, _ = self.env.reset(seed=0)
        bonus = self.agent.get_rnd_bonus(state)
        self.assertIsInstance(bonus, float)
        self.assertGreaterEqual(bonus, 0.0)

    def test_rnd_reward_weight_zero_gives_zero_bonus(self):
        """Scaling by rnd_reward_weight=0 must yield exactly 0."""
        agent_zero = RNDDQNAgent(
            self.env, rnd_hidden_size=16, rnd_n_layers=1, rnd_reward_weight=0.0, seed=1
        )
        state, _ = self.env.reset(seed=0)
        self.assertEqual(agent_zero.get_rnd_bonus(state), 0.0)

    def test_update_rnd_returns_finite_nonneg_loss(self):
        self._fill_buffer()
        batch = self.agent.buffer.sample(self.agent.batch_size)
        loss = self.agent.update_rnd(batch)
        self.assertIsInstance(loss, float)
        self.assertTrue(np.isfinite(loss))
        self.assertGreaterEqual(loss, 0.0)

    def test_update_rnd_changes_predictor_weights(self):
        """Gradient step must modify at least one predictor parameter."""
        self._fill_buffer()
        before = [p.clone() for p in self.agent.predictor_network_rnd.parameters()]
        self.agent.update_rnd(self.agent.buffer.sample(self.agent.batch_size))
        changed = any(
            not torch.equal(b, a)
            for b, a in zip(before, self.agent.predictor_network_rnd.parameters())
        )
        self.assertTrue(changed, "Predictor weights should change after update_rnd")

    def test_update_rnd_does_not_change_target_weights(self):
        """Frozen target RND network must remain unchanged after update_rnd."""
        self._fill_buffer()
        before = [p.clone() for p in self.agent.target_network_rnd.parameters()]
        self.agent.update_rnd(self.agent.buffer.sample(self.agent.batch_size))
        unchanged = all(
            torch.equal(b, a)
            for b, a in zip(before, self.agent.target_network_rnd.parameters())
        )
        self.assertTrue(unchanged, "Frozen target RND weights must not change")

    def test_train_smoke(self):
        """A short training run should complete without errors."""
        self.agent.train(num_frames=40, eval_interval=20)
        self.assertGreater(len(self.agent.buffer), 0)


if __name__ == "__main__":
    unittest.main()
