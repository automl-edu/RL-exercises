import unittest

from rl_exercises.environments import MarsRover
from rl_exercises.train_agent import evaluate
from rl_exercises.week_2.value_iteration import ValueIteration


class TestValueIteration(unittest.TestCase):
    def test_value_quality(self):
        seeds = range(1, 11)
        r = []
        for seed in seeds:
            # Subtask 3 evaluates Value Iteration on a probabilistic MarsRover
            env = MarsRover(transition_probabilities=[[0.5, 0.5]] * 5)
            agent = ValueIteration(env=env, seed=seed)
            agent.update_agent()
            mean_r = evaluate(env=env, agent=agent, episodes=1)
            r.append(mean_r)
        self.assertTrue(sum(r) > 0)
