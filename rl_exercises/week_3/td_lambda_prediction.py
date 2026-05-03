from __future__ import annotations

import numpy as np


class TDLambdaPredictionAgent:
    """State-value TD(lambda) with accumulating eligibility traces."""

    def __init__(
        self,
        n_states: int,
        alpha: float = 0.1,
        gamma: float = 1.0,
        lmbda: float = 0.9,
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.lmbda = lmbda

        self.V = np.full(n_states, 0.5, dtype=float)
        self.V[0] = 0.0
        self.V[-1] = 1.0

        self.E = np.zeros(n_states, dtype=float)

    def reset_traces(self):
        self.E[:] = 0.0

    def update(self, state: int, reward: float, next_state: int, done: bool):
        next_value = 0.0 if done else self.V[next_state]

        td_error = reward + self.gamma * next_value - self.V[state]

        self.E[state] += 1.0
        self.V += self.alpha * td_error * self.E
        self.E *= self.gamma * self.lmbda

        self.V[0] = 0.0
        self.V[-1] = 1.0

        if done:
            self.reset_traces()

        return td_error
