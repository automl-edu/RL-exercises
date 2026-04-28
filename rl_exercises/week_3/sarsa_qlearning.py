from __future__ import annotations

from typing import Any, DefaultDict, Literal

from collections import defaultdict

import gymnasium as gym
import numpy as np
from rl_exercises.agent import AbstractAgent
from rl_exercises.week_3 import EpsilonGreedyPolicy

State = Any


class TDAgent(AbstractAgent):
    """SARSA and Q-Learning agent"""

    def __init__(
        self,
        env: gym.Env,
        policy: EpsilonGreedyPolicy,
        alpha: float = 0.5,
        gamma: float = 1.0,
        algorithm: Literal["sarsa", "qlearning"] = "sarsa",
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
        algorithm : Literal["sarsa", "qlearning"], optional
            Whether to use SARSA (on-policy) or Q-Learning (off-policy), by default "sarsa"
        """
        # Check hyperparameter boundaries
        assert 0 <= gamma <= 1, "Gamma should be in [0, 1]"
        assert alpha > 0, "Learning rate has to be greater than 0"
        assert algorithm in [
            "sarsa",
            "qlearning",
        ], "algorithm must be 'sarsa' or 'qlearning'"

        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.algorithm = algorithm

        # number of actions → used by Q’s default factory
        self.n_actions = env.action_space.n

        # Build Q so that unseen states map to zero‐vectors
        self.Q: DefaultDict[Any, np.ndarray] = defaultdict(
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
        """Unpack a batch from SimpleBuffer and route to the appropriate TD update.

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
        if self.algorithm == "sarsa":
            # TODO: Get the next action for the lookahead in SARSA using the policy of this agent.
            next_action = 0
            return self.SARSA(state, action, reward, next_state, next_action, done)
        else:
            return self.Q_Learning(state, action, reward, next_state, done)

    def SARSA(
        self,
        state: State,
        action: int,
        reward: float,
        next_state: State,
        next_action: int,
        done: bool,
    ) -> float:
        """Perform a SARSA update (on-policy)
        Q[s,a] ← Q[s,a] + alpha*[r + gamma*Q(s',a') - Q(s,a)]

        Parameters
        ----------
        state : State
            Current state
        action : int
            Action taken
        reward : float
            Reward received
        next_state : State
            Next state
        next_action : int
            Next action for lookahead
        done : bool
            Whether the episode is finished

        Returns
        -------
        float
            New Q value for the state action pair
        """

        # SARSA update rule
        # TODO: Implement the SARSA update rule here.
        # Use a value of 0. for terminal states and
        # update the new Q value in the Q table of this class.
        # Return the new Q value --currently always returns 0.0

        return 0.0

    def Q_Learning(
        self,
        state: State,
        action: int,
        reward: float,
        next_state: State,
        done: bool,
    ) -> float:
        """Perform a Q-Learning update (off-policy)
        Q[s,a] ← Q[s,a] + alpha*[r + gamma*max(Q(s',·)) - Q(s,a)]

        Parameters
        ----------
        state : State
            Current state
        action : int
            Action taken
        reward : float
            Reward received
        next_state : State
            Next state
        done : bool
            Whether the episode is finished

        Returns
        -------
        float
            New Q value for the state action pair
        """

        # Q learning update rule
        # TODO: Implement the Q-Learning update rule here.

        return 0.0
