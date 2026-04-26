from __future__ import annotations

from typing import Any

import warnings

import numpy as np
from rich import print as printr
from rl_exercises.agent import AbstractAgent
from rl_exercises.environments import MarsRover


class PolicyIteration(AbstractAgent):
    """
    Policy Iteration Agent.

    This agent performs standard tabular policy iteration on an environment
    with known transition dynamics and rewards. The policy is evaluated and
    improved until convergence.

    Parameters
    ----------
    env : MarsRover
        Environment instance. This class is designed specifically for the MarsRover env.
    gamma : float, optional
        Discount factor for future rewards, by default 0.9.
    seed : int, optional
        Random seed for policy initialization, by default 333.
    filename : str, optional
        Path to save/load the policy, by default "policy.npy".
    """

    def __init__(
        self,
        env: MarsRover,
        gamma: float = 0.9,
        seed: int = 333,
        filename: str = "policy.npy",
        **kwargs: dict,
    ) -> None:
        if hasattr(env, "unwrapped"):
            env = env.unwrapped  # type: ignore[assignment]
        self.env = env
        self.seed = seed
        self.filename = filename

        super().__init__(**kwargs)

        self.n_obs = self.env.observation_space.n  # type: ignore[attr-defined]
        self.n_actions = self.env.action_space.n  # type: ignore[attr-defined]

        # Get the MDP components (states, actions, transitions, rewards)
        self.S = self.env.states
        self.A = self.env.actions
        self.T = self.env.get_transition_matrix()
        self.R = self.env.rewards
        self.gamma = gamma
        self.R_sa = self.env.get_reward_per_action()

        # Initialize policy and Q-values
        rng = np.random.default_rng(seed=self.seed)
        self.pi: np.ndarray = rng.integers(0, self.n_actions, self.n_obs)
        self.Q = np.zeros_like(self.R_sa)

        self.policy_fitted: bool = False
        self.steps: int = 0

    def predict_action(  # type: ignore[override]
        self, observation: int, info: dict | None = None, evaluate: bool = False
    ) -> tuple[int, dict]:
        """
        Predict an action using the current policy.

        Parameters
        ----------
        observation : int
            The current observation/state.
        info : dict or None, optional
            Additional info passed during prediction (unused).
        evaluate : bool, optional
            Evaluation mode toggle (unused here), by default False.

        Returns
        -------
        tuple[int, dict]
            The selected action and an empty info dictionary.
        """
        if not self.policy_fitted:
            self.update_agent()
        return int(self.pi[observation]), {}

    def update_agent(self, *args: tuple, **kwargs: dict) -> None:
        """Run policy iteration to compute the optimal policy and state-action values."""
        if not self.policy_fitted:
            printr("Initial policy: ", self.pi)
            MDP = (self.S, self.A, self.T, self.R_sa, self.gamma)
            self.Q, self.pi, self.steps = policy_iteration(
                Q=self.Q, pi=self.pi, MDP=MDP
            )
            printr("Q: ", self.Q)
            printr("Final policy: ", self.pi)
            printr("Policy iteration steps:", self.steps)
            self.policy_fitted = True

    def save(self, *args: tuple[Any], **kwargs: dict) -> None:
        """
        Save the learned policy to a `.npy` file.

        Raises
        ------
        Warning
            If the policy has not yet been fitted.
        """
        if self.policy_fitted:
            np.save(self.filename, np.array(self.pi))
        else:
            warnings.warn("Tried to save policy but policy is not fitted yet.")

    def load(self, *args: tuple[Any], **kwargs: dict) -> np.ndarray:
        """
        Load the policy from file.

        Returns
        -------
        np.ndarray
            The loaded policy array.
        """
        self.pi = np.load(self.filename)
        self.policy_fitted = True
        return self.pi


def policy_evaluation(
    pi: np.ndarray,
    T: np.ndarray,
    R_sa: np.ndarray,
    gamma: float,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """
    Perform policy evaluation for a fixed policy.

    Parameters
    ----------
    pi : np.ndarray
        The current policy (array of actions).
    T : np.ndarray
        Transition probabilities T[s, a, s'].
    R_sa : np.ndarray
        Reward matrix R[s, a].
    gamma : float
        Discount factor.
    epsilon : float, optional
        Convergence threshold, by default 1e-8.

    Returns
    -------
    np.ndarray
        The evaluated value function V[s] for all states.
    """
    nS = R_sa.shape[0]
    V = np.zeros(nS)

    while True:
        V_old = V.copy()
        for s in range(nS):
            a = pi[s]  # action according to current policy
            # Expected future value using transition probabilities
            V[s] = R_sa[s, a] + gamma * (T[s, a] @ V_old)

        # Check convergence
        if np.max(np.abs(V - V_old)) < epsilon:
            break

    return V


def policy_improvement(
    V: np.ndarray,
    T: np.ndarray,
    R_sa: np.ndarray,
    gamma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Improve the current policy based on the value function.

    Parameters
    ----------
    V : np.ndarray
        Current value function.
    T : np.ndarray
        Transition probabilities T[s, a, s'].
    R_sa : np.ndarray
        Reward matrix R[s, a].
    gamma : float
        Discount factor.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Q-function and the improved policy.
    """
    nS, nA = R_sa.shape
    Q = np.zeros((nS, nA))
    pi_new = np.zeros(nS, dtype=int)

    # Compute Q-values for each state-action pair
    for s in range(nS):
        for a in range(nA):
            immediate_reward = R_sa[s, a]

            # sum over all possible next states
            future_value = 0
            for s_next in range(nS):
                transition_prob = T[s, a, s_next]
                next_state_value = V[s_next]
                # accumulate expected future value
                future_value += transition_prob * next_state_value

            # combines immediate and discounted future rewards
            Q[s, a] = immediate_reward + gamma * future_value

    pi_new = np.argmax(Q, axis=1)

    return Q, pi_new


def policy_iteration(
    Q: np.ndarray,
    pi: np.ndarray,
    MDP: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float],
    epsilon: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Full policy iteration loop until convergence.

    Parameters
    ----------
    Q : np.ndarray
        Initial Q-table (can be zeros).
    pi : np.ndarray
        Initial policy.
    MDP : tuple
        A tuple (S, A, T, R_sa, gamma) representing the MDP.
    epsilon : float, optional
        Convergence threshold for value updates, by default 1e-8.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, int]
        Final Q-table, final policy, and number of improvement steps.
    """
    S, A, T, R_sa, gamma = MDP

    steps = 0
    while True:
        V = policy_evaluation(pi, T, R_sa, gamma, epsilon)
        Q, pi_new = policy_improvement(V, T, R_sa, gamma)

        steps += 1

        # is converged?
        if np.array_equal(pi, pi_new):
            break

        pi = pi_new

    return Q, pi, steps


if __name__ == "__main__":
    algo = PolicyIteration(env=MarsRover())
    algo.update_agent()
