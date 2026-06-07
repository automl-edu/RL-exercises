import unittest

import gymnasium as gym
import numpy as np
import torch

# Adjust import as needed
from rl_exercises.week_6 import ActorCriticAgent


class DummyEnv(gym.Env):
    """
    A trivial 1-state, 1-action env that always returns reward=1 and ends immediately.
    Used to test evaluate() deterministically.
    """

    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(1,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(1)

    def reset(self, *, seed=None, **kwargs):
        """
        Reset environment optionally with seed for reproducibility.
        """
        if seed is not None:
            np.random.seed(seed)
        return np.array([0.0], dtype=np.float32), {}

    def step(self, action):
        """
        Take a step: always return reward=1.0 and done.
        """
        return np.array([0.0], dtype=np.float32), 1.0, True, False, {}


class TestActorCriticAgent(unittest.TestCase):
    def test_predict_action_signature(self):
        """
        Verify predict_action returns an integer action and torch.Tensor log_prob in training mode,
        and returns deterministic action with None log_prob in evaluation mode.
        """
        env = gym.make("CartPole-v1")
        agent = ActorCriticAgent(env, seed=0)
        state, _ = env.reset(seed=0)
        action, logp = agent.predict_action(state, evaluate=False)
        self.assertIsInstance(action, int)
        self.assertIsInstance(logp, torch.Tensor)

        action_eval, logp_eval = agent.predict_action(state, evaluate=True)
        self.assertIsInstance(action_eval, int)
        self.assertIsNone(logp_eval)

    def test_compute_returns(self):
        """
        Check compute_returns produces the correct discounted sum of rewards.
        """
        env = DummyEnv()
        agent = ActorCriticAgent(env, baseline_type="none", gamma=0.5)
        rewards = [1.0, 1.0, 1.0]
        returns = agent.compute_returns(rewards)
        expected = torch.tensor(
            [1.0 + 0.5 * (1.0 + 0.5 * 1.0), 1.0 + 0.5 * 1.0, 1.0], dtype=torch.float32
        )
        self.assertTrue(
            torch.allclose(returns, expected), f"Expected {expected}, got {returns}"
        )

    def test_compute_advantages_zero_value_baseline(self):
        """
        With a zero-valued stub value_fn, compute_advantages should match returns
        and yield zero-mean normalized advantages.
        """
        env = DummyEnv()
        agent = ActorCriticAgent(env, baseline_type="value")
        # stub out value network
        agent.value_fn = lambda x: torch.zeros(x.shape[0])
        states = [np.array([0.0], dtype=np.float32) for _ in range(3)]
        rewards = [1.0, 2.0, 3.0]
        advantages, returns = agent.compute_advantages(states, rewards)
        expected_returns = agent.compute_returns(rewards)
        self.assertTrue(torch.allclose(returns, expected_returns), "Returns mismatch")
        self.assertAlmostEqual(advantages.mean().item(), 0.0, places=6)

    def test_compute_gae_zero_baseline(self):
        """
        With a zero-valued stub value_fn, compute_gae should yield the correct GAE returns
        equal to the exponentially-weighted sum of deltas, and produce normalized advantages
        of zero mean and unit variance.
        """
        env = DummyEnv()
        gamma = 0.99
        lam = 0.95
        agent = ActorCriticAgent(env, baseline_type="gae", gamma=gamma, gae_lambda=lam)
        # stub out value network to zero
        agent.value_fn = lambda x: torch.zeros(x.shape[0])
        states = [np.array([0.0], dtype=np.float32) for _ in range(4)]
        next_states = states[1:] + [states[-1]]
        rewards = [0.5, 0.5, 0.5, 0.5]
        dones = [False, False, False, True]
        # compute via agent
        advantages, returns = agent.compute_gae(states, rewards, next_states, dones)
        # manually compute expected unnormalized advantages (returns)
        expected_adv = []
        A = 0.0
        # compute deltas = rewards (since zero values)
        deltas = rewards[::-1]
        for delta in deltas:
            A = delta + gamma * lam * A
            expected_adv.insert(0, A)
        expected_adv = torch.tensor(expected_adv, dtype=torch.float32)
        self.assertTrue(
            torch.allclose(returns, expected_adv, atol=1e-4),
            f"Expected GAE returns {expected_adv}, got {returns}",
        )
        # advantages should be normalized: mean ~ 0, std ~ 1
        self.assertAlmostEqual(advantages.mean().item(), 0.0, places=6)
        self.assertAlmostEqual(advantages.std(unbiased=False).item(), 1.0, places=6)

    def make_dummy_trajectory(self, env, length=3):
        """
        Generate a simple trajectory of specified length with constant log_probs and rewards.
        """
        state, _ = env.reset()
        traj = []
        for _ in range(length):
            action = env.action_space.sample()
            next_state, reward, term, trunc, _ = env.step(action)
            logp = torch.tensor(0.0)
            done = term or trunc
            traj.append((state, action, float(reward), next_state, done, logp))
            state = next_state
            if done:
                state, _ = env.reset()
        return traj

    def test_update_agent_baselines(self):
        """
        Test update_agent for all baseline types ('none', 'avg', 'value', 'gae'),
        checking valid policy_loss and appropriate value_loss behavior.
        """
        env = DummyEnv()
        traj = self.make_dummy_trajectory(env)

        # none
        agent_none = ActorCriticAgent(env, baseline_type="none")
        policy_n, value_n = agent_none.update_agent(traj)
        self.assertIsInstance(policy_n, float)
        self.assertEqual(value_n, 0.0)

        # avg
        agent_avg = ActorCriticAgent(env, baseline_type="avg", baseline_decay=0.5)
        initial = agent_avg.running_return
        policy_a, value_a = agent_avg.update_agent(traj)
        self.assertIsInstance(policy_a, float)
        self.assertEqual(value_a, 0.0)
        self.assertNotEqual(agent_avg.running_return, initial)

        # value
        agent_val = ActorCriticAgent(env, baseline_type="value")
        policy_v, value_v = agent_val.update_agent(traj)
        self.assertGreaterEqual(value_v, 0.0)

        # gae
        agent_gae = ActorCriticAgent(env, baseline_type="gae")
        policy_g, value_g = agent_gae.update_agent(traj)
        self.assertGreaterEqual(value_g, 0.0)


if __name__ == "__main__":
    unittest.main()
