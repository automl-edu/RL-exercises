import unittest

import gymnasium as gym
import torch
from rl_exercises.week_7 import REINFORCE


# FIXME: update isn't being tested
class TestPolicyGradient(unittest.TestCase):
    def test_compute_returns(self):
        env = gym.make("LunarLander-v2")

        agent = REINFORCE(env, 0.1, 0.99)

        self.assertAlmostEqual(
            agent.compute_returns([1, 1, 1, 1, 1]),
            [4.90099501, 3.9403989999999998, 2.9701, 1.99, 1.0],
        )

    def test_policy_improvement(self):
        env = gym.make("CartPole-v1")
        agent = REINFORCE(env, 1e-2, 0.99)

        log_probs = [
            -0.8483388423919678,
            -0.8832764029502869,
            -0.4966419041156769,
            -0.5348740220069885,
            -0.5604409575462341,
        ]

        rewards = [1.0 for _ in range(len(log_probs))]

        trajectory = [
            (torch.tensor(log_probs[i], requires_grad=True), rewards[i])
            for i in range(len(log_probs))
        ]

        loss = agent.update_agent(*zip(*trajectory))

        self.assertAlmostEqual(loss, 0.5832083821296692)


if __name__ == "__main__":
    unittest.main()
