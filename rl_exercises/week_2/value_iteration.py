from __future__ import annotations

from typing import Any, Tuple

import gymnasium
import numpy as np
from rich import print as printr
from rl_exercises.agent import AbstractAgent
from rl_exercises.environments import MarsRover


class ValueIteration(AbstractAgent):
    """Agent that computes an optimal policy via Value Iteration.

    Parameters
    ----------
    env : MarsRover or gymnasium.Env
        The target environment, must expose `.states`, `.actions`,
        `.transition_matrix` and `.get_reward_per_action()`.
    gamma : float, default=0.9
        Discount factor for future rewards.
    seed : int, default=333
        Random seed for tie‐breaking among equally‐good actions.

    Attributes
    ----------
    V : np.ndarray, shape (n_states,)
        The computed optimal value function.
    pi : np.ndarray, shape (n_states,)
        The greedy policy derived from V.
    policy_fitted : bool
        Whether value iteration has been run yet.
    """

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
        self.S = env.states  # array of state indices
        self.A = env.actions  # array of action indices
        self.T = env.transition_matrix  # shape (nS, nA, nS)
        self.R_sa = env.get_reward_per_action()  # shape (nS, nA)
        self.n_states, self.n_actions = self.R_sa.shape

        # placeholders
        self.V = np.zeros(self.n_states, dtype=float)
        self.pi = np.zeros(self.n_states, dtype=int)
        self.policy_fitted = False

    def update_agent(self, *args: tuple[Any], **kwargs: dict) -> None:
        """Run value iteration and store the resulting V and π."""
        if self.policy_fitted:
            return

        V_opt, pi_opt = value_iteration(
            T=self.T, R_sa=self.R_sa, gamma=self.gamma, seed=self.seed
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
        """Choose action = π(observation). Runs update if needed."""
        if not self.policy_fitted:
            self.update_agent()

        # Simply call and return after
        action = int(self.pi[observation])
        return action, {}


def value_iteration(
    *,
    T: np.ndarray,
    R_sa: np.ndarray,
    gamma: float,
    seed: int | None = None,
    epsilon: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run Value Iteration on a finite MDP.

    Solves for
        V*(s) = max_a [ R_sa[s,a] + γ ∑_{s'} T[s,a,s'] V*(s') ]
    and then
        π*(s) = argmax_a [ R_sa[s,a] + γ ∑_{s'} T[s,a,s'] V*(s') ].

    Parameters
    ----------
    T : np.ndarray, shape (n_states, n_actions, n_states)
        Transition probabilities.
    R_sa : np.ndarray, shape (n_states, n_actions)
        Rewards for each (state, action).
    gamma : float
        Discount factor (0 ≤ γ < 1).
    seed : int or None
        RNG seed for tie‐breaking among equal actions.
    epsilon : float
        Stopping threshold on max value‐update difference.

    Returns
    -------
    V : np.ndarray, shape (n_states,)
        Optimal state‐value function.
    pi : np.ndarray, shape (n_states,)
        Greedy policy w.r.t. V, with random tie‐breaking.
    """
    n_states, n_actions = R_sa.shape
    V = np.zeros(n_states, dtype=float)
    # rng = np.random.default_rng(seed)
    pi = np.zeros(n_states, dtype=int)

    # update V using the Q values until convergence
    while True:
        delta = 0
        V_old = V.copy()
        # Iterate over all actions for all states to calculate Q-Value
        for s in range(n_states):
            Q_s = np.zeros(n_actions)
            for a in range(n_actions):
                Q_s[a] = R_sa[s, a] + gamma * np.dot(T[s, a, :], V_old)

            # Value = max Q-Value
            V[s] = np.max(Q_s)
            delta = max(delta, np.abs(V[s] - V_old[s]))

        if delta < epsilon:
            break

    # Extract the greedy policy from V and update pi
    for s in range(n_states):
        Q_s = np.zeros(n_actions)
        for a in range(n_actions):
            Q_s[a] = R_sa[s, a] + gamma * np.dot(T[s, a, :], V)

        # Set policy for s via max Q-Value
        pi[s] = np.argmax(Q_s)

    return V, pi
