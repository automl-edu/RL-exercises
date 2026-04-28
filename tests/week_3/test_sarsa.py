import numpy as np
import pytest
from rl_exercises.week_3 import EpsilonGreedyPolicy, TDAgent


class DummyEnv2:
    action_space = type("A", (), {"n": 2})()


class DummyEnv5:
    action_space = type("A", (), {"n": 5})()


def make_batch(state, action, reward, next_state, done):
    return [(state, action, reward, next_state, done, {})]


class TestSARSA:
    def test_td_update_computation(self):
        """SARSA TD-error: Q[s,a] += alpha * (r + gamma * Q[s',a'] - Q[s,a])"""
        policy = EpsilonGreedyPolicy(DummyEnv2(), epsilon=0.0, seed=0)
        agent = TDAgent(DummyEnv2(), policy, alpha=0.5, gamma=0.9)

        agent.Q[0][1] = 2.0
        agent.Q[1][0] = 0.5  # epsilon=0 → next_action = argmax(Q[1]) = 0

        new_q = agent.update_agent(make_batch(0, 1, 1.0, 1, False))

        expected = 2.0 + 0.5 * ((1.0 + 0.9 * 0.5) - 2.0)
        assert new_q == pytest.approx(expected)
        assert agent.Q[0][1] == pytest.approx(expected)

    def test_predict_action_greedy_vs_random(self):
        """epsilon=0 is always greedy; epsilon=1 explores all actions."""
        policy0 = EpsilonGreedyPolicy(DummyEnv5(), epsilon=0.0, seed=42)
        agent0 = TDAgent(DummyEnv5(), policy0, alpha=0.1, gamma=0.9)
        agent0.Q["s"] = np.array([1, 1, 5, 1, 1])
        picks0 = {agent0.predict_action("s", evaluate=False)[0] for _ in range(20)}
        assert picks0 == {2}

        policy1 = EpsilonGreedyPolicy(DummyEnv5(), epsilon=1.0, seed=123)
        agent1 = TDAgent(DummyEnv5(), policy1, alpha=0.1, gamma=0.9)
        picks1 = [agent1.predict_action("s", evaluate=False)[0] for _ in range(200)]
        assert len(set(picks1)) >= 4

    def test_terminal_state_ignores_next_q(self):
        """When done=True, the next-Q term must be zeroed out."""
        policy = EpsilonGreedyPolicy(DummyEnv2(), epsilon=0.0, seed=0)
        agent = TDAgent(DummyEnv2(), policy, alpha=0.5, gamma=0.9)

        agent.Q[0][1] = 1.0
        agent.Q[1][0] = 100.0  # should be ignored

        new_q = agent.update_agent(make_batch(0, 1, 1.0, 1, True))

        assert new_q == pytest.approx(1.0)
        assert agent.Q[0][1] == pytest.approx(1.0)
