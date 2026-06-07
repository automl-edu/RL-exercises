import unittest

import gymnasium as gym
import numpy as np
import torch

# Adjust import as needed
from rl_exercises.week_6 import PPOAgent


class DummyEnv(gym.Env):
    """
    A trivial 1-state, 1-action environment that always returns reward=1 and ends immediately.
    Used to deterministically test evaluate() and PPO updates.
    """

    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(1,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(1)

    def reset(self, *, seed=None, **kwargs):
        """
        Reset environment. Can accept seed for reproducibility.
        """
        if seed is not None:
            np.random.seed(seed)
        return np.array([0.0], dtype=np.float32), {}

    def step(self, action):
        """
        Take a step: always return next_state same, reward=1.0, done=True.
        """
        return np.array([0.0], dtype=np.float32), 1.0, True, False, {}


class TestPPOAgent(unittest.TestCase):
    def test_predict_signature(self):
        """
        predict() should return action(int), log_prob(torch.Tensor), entropy(torch.Tensor), and value(torch.Tensor).
        """
        env = gym.make("CartPole-v1")
        agent = PPOAgent(env, seed=0)
        state, _ = env.reset(seed=0)
        action, logp, ent, val = agent.predict(state)
        self.assertIsInstance(action, int)
        self.assertIsInstance(logp, torch.Tensor)
        self.assertIsInstance(ent, torch.Tensor)
        self.assertIsInstance(val, torch.Tensor)

    def test_compute_gae(self):
        """
        compute_gae should match manual GAE for zero value_fn stub and normalize advantages.
        """
        env = DummyEnv()
        agent = PPOAgent(env, gamma=0.9, gae_lambda=0.8, seed=0)
        # stub values and next_values as zeros
        rewards = [1.0, 2.0, 3.0]
        dones = torch.tensor([0.0, 0.0, 1.0])
        values = torch.zeros(3)
        next_values = torch.zeros(3)
        advs, returns = agent.compute_gae(rewards, values, next_values, dones)
        # manual deltas equal rewards
        # compute manual adv
        manual = []
        A = 0.0
        for r, d in zip(reversed(rewards), reversed(dones.tolist())):
            A = r + 0.9 * 0.8 * A * (1 - d)
            manual.insert(0, A)
        manual = torch.tensor(manual)
        # returns = manual (since values zero)
        self.assertTrue(torch.allclose(returns, manual, atol=1e-6))
        # advs normalized
        self.assertAlmostEqual(advs.mean().item(), 0.0, places=6)
        self.assertAlmostEqual(advs.std(unbiased=False).item(), 1.0, places=6)

    def make_dummy_trajectory(self, env, length=4):
        """
        Creates a dummy trajectory: (state, action, logp, entropy, reward, done, next_state)
        with constant values for testing update().
        """
        traj = []
        state, _ = env.reset()
        for _ in range(length):
            action = env.action_space.sample()
            logp = torch.tensor(0.0, requires_grad=True)
            ent = torch.tensor(0.0)
            next_state, reward, term, trunc, _ = env.step(action)
            done = float(term or trunc)
            traj.append((state, action, logp, ent, reward, done, next_state))
            state = next_state
            if done:
                state, _ = env.reset()
        return traj

    def test_update_no_errors(self):
        """
        update() should run without errors and return finite losses.
        """
        env = DummyEnv()
        agent = PPOAgent(env, epochs=2, batch_size=2, seed=0)
        traj = self.make_dummy_trajectory(env)
        policy_loss, value_loss, entropy_loss = agent.update(traj)
        self.assertIsInstance(policy_loss, float)
        self.assertIsInstance(value_loss, float)
        self.assertIsInstance(entropy_loss, float)
        self.assertTrue(np.isfinite(policy_loss))
        self.assertTrue(np.isfinite(value_loss))
        self.assertTrue(np.isfinite(entropy_loss))

    def test_evaluate_dummy_env(self):
        """
        evaluate() on DummyEnv should yield mean=1.0 and std=0.0.
        """
        env = DummyEnv()
        agent = PPOAgent(env, seed=0)
        mean, std = agent.evaluate(env, num_episodes=5)
        self.assertAlmostEqual(mean, 1.0, places=6)
        self.assertAlmostEqual(std, 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
