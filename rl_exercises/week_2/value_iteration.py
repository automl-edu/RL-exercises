from __future__ import annotations

from typing import Any, Tuple

import gymnasium
import numpy as np
from rich import print as printr
from rl_exercises.agent import AbstractAgent
from rl_exercises.environments import MarsRover


class ValueIteration(AbstractAgent):
    """Agent that computes an optimal policy via Value Iteration."""

    def __init__(
        self,
        env: MarsRover | gymnasium.Env,
        gamma: float = 0.9,
        seed: int = 333,
        **kwargs: dict,
    ) -> None:
        if hasattr(env, "unwrapped"):
            env = env.unwrapped  # type: ignore

        super().__init__(**kwargs)

        self.env = env
        self.gamma = gamma
        self.seed = seed

        # extract MDP
        self.S = env.states
        self.A = env.actions
        self.T = env.transition_matrix
        self.R_sa = env.get_reward_per_action()
        self.n_states, self.n_actions = self.R_sa.shape

        # placeholders
        self.V = np.zeros(self.n_states, dtype=float)
        self.pi = np.zeros(self.n_states, dtype=int)
        self.policy_fitted = False

    def update_agent(self, *args: tuple[Any], **kwargs: dict) -> None:
        """Run value iteration and store the resulting V and policy."""
        if self.policy_fitted:
            return

        V_opt, pi_opt = value_iteration(
            T=self.T,
            R_sa=self.R_sa,
            gamma=self.gamma,
            seed=self.seed,
        )

        self.V = V_opt
        self.pi = pi_opt

        printr("Converged V:", self.V)
        printr("Derived policy π:", self.pi)

        self.policy_fitted = True

    def predict_action(
        self,
        observation: int,
        info: dict | None = None,
        evaluate: bool = False,
    ) -> tuple[int, dict]:
        """Choose action = pi(observation). Runs update if needed."""
        if not self.policy_fitted:
            self.update_agent()

        action = int(self.pi[observation])
        return action, {}


def value_iteration(
    *,
    T: np.ndarray,
    R_sa: np.ndarray,
    gamma: float,
    seed: int | None = None,
    epsilon: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run Value Iteration on a finite MDP."""

    n_states, n_actions = R_sa.shape
    V = np.zeros(n_states, dtype=float)
    rng = np.random.default_rng(seed)

    while True:
        delta = 0.0
        V_new = np.zeros_like(V)

        for s in range(n_states):
            q_values = np.zeros(n_actions)

            for a in range(n_actions):
                q_values[a] = R_sa[s, a] + gamma * np.sum(T[s, a, :] * V)

            V_new[s] = np.max(q_values)
            delta = max(delta, abs(V_new[s] - V[s]))

        V = V_new

        if delta < epsilon:
            break

    pi = np.zeros(n_states, dtype=int)

    for s in range(n_states):
        q_values = np.zeros(n_actions)

        for a in range(n_actions):
            q_values[a] = R_sa[s, a] + gamma * np.sum(T[s, a, :] * V)

        # random tie-breaking if multiple actions have same Q-value
        best_actions = np.flatnonzero(np.isclose(q_values, np.max(q_values)))
        pi[s] = int(rng.choice(best_actions))

    return V, pi
