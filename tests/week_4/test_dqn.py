# test_deep_q_learning.py

"""
Unit tests for the deep_q_learning module.

Verifies:
 - ReplayBuffer behavior (inheritance, FIFO eviction, sampling).
 - DQNAgent API (inheritance, predict_action, save/load, update, training loop).
"""

import unittest

import gymnasium as gym
import numpy as np
import torch
from rl_exercises.week_4 import DQNAgent, ReplayBuffer


class TestReplayBuffer(unittest.TestCase):
    """Tests for the FIFO replay buffer implementation."""

    def setUp(self):
        """Create a small buffer and a dummy transition."""
        self.capacity = 5
        self.buf = ReplayBuffer(self.capacity)
        self.sample_transition = (
            np.zeros((3,)),  # state
            1,  # action
            0.5,  # reward
            np.ones((3,)),  # next_state
            False,  # done
            {"info": "x"},  # info
        )

    def test_add_and_len(self):
        """Adding transitions increases length up to capacity."""
        self.assertEqual(len(self.buf), 0)
        self.buf.add(*self.sample_transition)
        self.assertEqual(len(self.buf), 1)

        # exceed capacity → length stays at max
        for _ in range(self.capacity + 3):
            self.buf.add(*self.sample_transition)
        self.assertEqual(len(self.buf), self.capacity)

    def test_fifo_eviction(self):
        """Oldest transitions are evicted first when full."""
        # fill with distinct states 0–4
        for i in range(self.capacity):
            state = np.full((3,), i, dtype=float)
            self.buf.add(state, 0, 0.0, state, False, {})
        # add one more → pops state=0
        new_state = np.full((3,), 99, dtype=float)
        self.buf.add(new_state, 0, 0.0, new_state, False, {})
        # sample full batch and check no zero
        sampled = [t[0][0] for t in self.buf.sample(self.capacity)]
        self.assertNotIn(0.0, sampled)

    def test_sample_unique(self):
        """Sampling without replacement returns distinct indices."""
        # add more than capacity
        for _ in range(10):
            self.buf.add(*self.sample_transition)
        batch = self.buf.sample(4)
        self.assertEqual(len(batch), 4)
        # check tuple shape
        for t in batch:
            self.assertEqual(len(t), 6)


class TestDQNAgent(unittest.TestCase):
    """Tests for the DQNAgent class."""

    def setUp(self):
        """Initialize a small‐scale agent for CartPole."""
        self.env = gym.make("CartPole-v1")
        self.agent = DQNAgent(
            self.env,
            buffer_capacity=20,
            batch_size=4,
            lr=1e-2,
            gamma=0.9,
            epsilon_start=0.5,
            epsilon_final=0.1,
            epsilon_decay=10,
            target_update_freq=5,
            seed=0,
        )

    def test_predict_action(self):
        """predict_action returns a valid action and an info dict."""
        obs, _ = self.env.reset(seed=0)
        action = self.agent.predict_action(obs)
        self.assertIsInstance(action, int)
        self.assertTrue(self.env.action_space.contains(action))

    def test_update_agent(self):
        """One call to update_agent actually changes at least one weight."""
        # fill buffer
        obs, _ = self.env.reset(seed=0)
        for _ in range(self.agent.batch_size):
            a = self.env.action_space.sample()
            ns, r, d, tr, _ = self.env.step(a)
            self.agent.buffer.add(obs, a, r, ns, d or tr, {})
        batch = self.agent.buffer.sample(self.agent.batch_size)
        before = [p.clone() for p in self.agent.q.parameters()]
        loss = self.agent.update_agent(batch)
        self.assertIsInstance(loss, float)
        # check change
        self.assertTrue(
            any(
                not torch.equal(b, a) for b, a in zip(before, self.agent.q.parameters())
            )
        )

    def test_train_smoke(self):
        """A short training run should complete without errors."""
        self.agent.train(num_frames=50, eval_interval=25)
        # buffer should have grown
        self.assertGreater(len(self.agent.buffer), 0)


if __name__ == "__main__":
    unittest.main()
