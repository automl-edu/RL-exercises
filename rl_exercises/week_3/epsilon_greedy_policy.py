from __future__ import annotations

from typing import DefaultDict

import gymnasium as gym
import numpy as np


class EpsilonGreedyPolicy(object):
    """A Policy doing Epsilon Greedy Exploration."""

    def __init__(
        self,
        env: gym.Env,
        epsilon: float,
        seed: int = 0,
    ) -> None:
        """Init

        Parameters
        ----------
        env : gym.Env
            Environment
        epsilon: float
            Exploration rate
        seed : int, optional
            Seed, by default None
        """
        assert 0 <= epsilon <= 1, "ε must be in [0,1]"
        self.env = env
        self.epsilon = epsilon

        # our private RNG, so sampling is reproducible
        self.rng = np.random.default_rng(seed)

    def __call__(self, Q: DefaultDict, state: tuple, evaluate: bool = False) -> int:  # type: ignore # noqa: E501
        """Select action

        Parameters
        ----------
        Q : DefaultDict
            Q Table/Function

        state : tuple
            State

        evaluate: bool
            evaluation mode - if true, exploration should be turned off.

        Returns
        -------
        int
            action
        """

        # If evaluation mode, skip exploration entirely
        if evaluate:
            return int(np.argmax(Q[state]))

        # TODO: Implement epsilon-greedy action selection
        # With prob 1 - epsilon return the greedy action
        # Wtih prob epsilon, use the policy's RNG to select a random action
        # Return the selected action -- currently always returns 0
        return 0  # uncomment to run the solution
