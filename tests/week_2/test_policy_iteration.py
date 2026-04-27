import unittest

import numpy as np
from rl_exercises.environments import MarsRover
from rl_exercises.train_agent import evaluate
from rl_exercises.week_2.policy_iteration import (
    PolicyIteration,
    policy_evaluation,
    policy_improvement,
)


class TestPolicyIteration(unittest.TestCase):
    def test_policy_quality(self):
        """Running PI on MarsRover should yield a deterministic policy that
        achieves positive reward."""
        env = MarsRover()
        agent = PolicyIteration(env=env, seed=0)
        agent.update_agent()
        # policy_fitted flag should be True
        self.assertTrue(agent.policy_fitted)

        # Evaluate for a handful of episodes
        mean_r = evaluate(env=env, agent=agent, episodes=10)
        self.assertGreater(mean_r, 0.0)

    def test_policy_improvement_idempotent(self):
        """If Q is already maximized wrt itself, policy_improvement should return
        the same policy."""
        # Build a tiny MDP: 3 states, 2 actions
        R_sa = np.array([[1, 0], [0, 1], [1, 0]], dtype=float)
        # Transitions: stay in place with prob=1
        T = np.zeros((3, 2, 3), dtype=float)
        for s in range(3):
            for a in range(2):
                T[s, a, s] = 1.0

        gamma = 0.9
        # Hand‐craft a policy that is already greedy for R_sa
        pi_before = np.array([0, 1, 0], dtype=int)
        # Evaluate that policy to get V
        V = policy_evaluation(pi_before, T, R_sa, gamma)
        # Improve
        Q_new, pi_after = policy_improvement(V, T, R_sa, gamma)

        # policy must stay the same
        np.testing.assert_array_equal(pi_after, pi_before)
        # Q_new should reflect R_sa + γ·V
        expected_Q00 = R_sa[0, 0] + gamma * V[0]  # s=0,a=0
        self.assertAlmostEqual(Q_new[0, 0], expected_Q00)


if __name__ == "__main__":
    unittest.main()
