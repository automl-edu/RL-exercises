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

        self.S = self.env.states
        self.A = self.env.actions
        self.T = self.env.get_transition_matrix()
        self.R = self.env.rewards
        self.gamma = gamma
        self.R_sa = self.env.get_reward_per_action()

        rng = np.random.default_rng(seed=self.seed)
        self.pi: np.ndarray = rng.integers(0, self.n_actions, self.n_obs)
        self.Q = np.zeros_like(self.R_sa)

        self.policy_fitted: bool = False
        self.steps: int = 0

    def predict_action(
        self, observation: int, info: dict | None = None, evaluate: bool = False
    ) -> tuple[int, dict]:
        """
        Predict an action using the current policy.
        """
        action = int(self.pi[observation])
        return action, {}

    def update_agent(self, *args: tuple, **kwargs: dict) -> None:
        """
        Run policy iteration to compute the optimal policy and state-action values.
        """
        if not self.policy_fitted:
            printr("Initial policy: ", self.pi)

            mdp = (self.S, self.A, self.T, self.R_sa, self.gamma)
            self.Q, self.pi, self.steps = policy_iteration(
                self.Q,
                self.pi,
                mdp,
            )

            printr("Q: ", self.Q)
            printr("Final policy: ", self.pi)
            printr("Policy iteration steps:", self.steps)

            self.policy_fitted = True

    def save(self, *args: tuple[Any], **kwargs: dict) -> None:
        """
        Save the learned policy to a `.npy` file.
        """
        if self.policy_fitted:
            np.save(self.filename, np.array(self.pi))
        else:
            warnings.warn("Tried to save policy but policy is not fitted yet.")

    def load(self, *args: tuple[Any], **kwargs: dict) -> np.ndarray:
        """
        Load the policy from file.
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
    """
    nS = R_sa.shape[0]
    V = np.zeros(nS)

    while True:
        delta = 0.0

        for s in range(nS):
            old_v = V[s]
            a = int(pi[s])

            V[s] = R_sa[s, a] + gamma * np.sum(T[s, a, :] * V)

            delta = max(delta, abs(old_v - V[s]))

        if delta < epsilon:
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
    """
    nS, nA = R_sa.shape
    Q = np.zeros((nS, nA))

    for s in range(nS):
        for a in range(nA):
            Q[s, a] = R_sa[s, a] + gamma * np.sum(T[s, a, :] * V)

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
    """
    S, A, T, R_sa, gamma = MDP

    steps = 0

    while True:
        old_pi = pi.copy()

        V = policy_evaluation(
            pi=pi,
            T=T,
            R_sa=R_sa,
            gamma=gamma,
            epsilon=epsilon,
        )

        Q, pi = policy_improvement(
            V=V,
            T=T,
            R_sa=R_sa,
            gamma=gamma,
        )

        steps += 1

        if np.array_equal(old_pi, pi):
            break

    return Q, pi, steps


if __name__ == "__main__":
    algo = PolicyIteration(env=MarsRover())
    algo.update_agent()
