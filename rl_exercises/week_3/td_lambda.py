from __future__ import annotations

from typing import Any, DefaultDict

from collections import defaultdict

import gymnasium as gym
import numpy as np
from rl_exercises.agent import AbstractAgent
from rl_exercises.week_3 import EpsilonGreedyPolicy

State = Any


class TDAgent(AbstractAgent):
    """Temporal Difference Learning Agent"""

    def __init__(
        self,
        env: gym.Env,
        policy: EpsilonGreedyPolicy,
        alpha: float = 0.5,
        gamma: float = 1.0,
        lmbda: float = 0.9,
    ) -> None:
        """Initialize the TD agent

        Parameters
        ----------
        env : gym.Env
            Environment for the agent
        alpha : float, optional
            Learning Rate, by default 0.5
        gamma : float, optional
            Discount Factor , by default 1.0
        lmbda : float, optional
            Trace decay, by default 0.9
        """
        # Check hyperparameter boundaries
        assert 0 <= gamma <= 1, "Gamma should be in [0, 1]"
        assert alpha > 0, "Learning rate has to be greater than 0"
        assert 0 <= lmbda <= 1, "Lambda should be in [0, 1]"

        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.lmbda = lmbda

        # number of actions → used by Q’s default factory
        self.n_actions = env.action_space.n

        # Build Q so that unseen states map to zero‐vectors
        self.Q: DefaultDict[Any, np.ndarray] = defaultdict(
            lambda: np.zeros(self.n_actions, dtype=float)
        )

        # Eligibility traces for each state-action pair
        self.E: DefaultDict[Any, np.ndarray] = defaultdict(
            lambda: np.zeros(self.n_actions, dtype=float)
        )

        self.policy = policy

    def predict_action(
        self, state: np.array, info: dict = {}, evaluate: bool = False
    ) -> Any:  # type: ignore # noqa
        """Predict the action for a given state"""
        return self.policy(self.Q, state, evaluate=evaluate), info

    def save(self, path: str) -> Any:  # type: ignore
        """Save the Q table

        Parameters
        ----------
        path :
            Path to save the Q table

        """
        np.save(path, dict(self.Q))  # type: ignore

    def load(self, path) -> Any:  # type: ignore
        """Load the Q table

        Parameters
        ----------
        path :
            Path to saved the Q table

        """
        loaded_q = np.load(path, allow_pickle=True).item()
        self.Q = defaultdict(
            lambda: np.zeros(self.n_actions, dtype=float),
            loaded_q,
        )

    def update_agent(self, batch) -> float:  # type: ignore
        """Unpack a batch from SimpleBuffer and apply a TD(lambda) update.

        Parameters
        ----------
        batch : list
            List of (state, action, reward, next_state, done, info) tuples

        Returns
        -------
        float
            New Q value for the state action pair
        """
        state, action, reward, next_state, done, _ = batch[0]
        next_action, _ = self.predict_action(next_state)
        return self.TD_lambda(state, action, reward, next_state, next_action, done)

    def TD_lambda(
        self,
        state: State,
        action: int,
        reward: float,
        next_state: State,
        next_action: int,
        done: bool,
    ) -> float:
        """Perform a TD(lambda) update with accumulating eligibility traces."""

        for traced_state in list(self.E.keys()):
            self.E[traced_state] *= self.gamma * self.lmbda
        self.E[state][action] += 1.0

        if done:
            next_value = 0.0
        else:
            next_value = self.Q[next_state][next_action]
        td_error = reward + self.gamma * next_value - self.Q[state][action]

        for traced_state, traces in self.E.items():
            self.Q[traced_state] += self.alpha * td_error * traces

        if done:
            self.E.clear()

        return self.Q[state][action]
