"""GridCore Env taken from https://github.com/automl/TabularTempoRL/"""

from __future__ import annotations

from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np


class MarsRover(gym.Env):
    """
    Simple Environment for a Mars Rover that can move in a 1D Space.

    The rover starts at position 2 and moves left or right based on discrete actions.
    The environment is stochastic: with a probability defined by a transition matrix,
    the action may be flipped. Each cell has an associated reward.

    Actions
    -------
    Discrete(2):
    - 0: go left
    - 1: go right

    Observations
    ------------
    Discrete(n): The current position of the rover (int).

    Reward
    ------
    Depends on the resulting cell after action is taken.

    Start/Reset State
    -----------------
    Always starts at position 2.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        transition_probabilities: np.ndarray = np.ones((5, 2)),
        rewards: list[float] = [1, 0, 0, 0, 10],
        horizon: int = 10,
        seed: int | None = None,
    ):
        """
        Initialize the Mars Rover environment.

        Parameters
        ----------
        transition_probabilities : np.ndarray, optional
            A (num_states, 2) array specifying the probability of actions being followed.
        rewards : list of float, optional
            Rewards assigned to each position, by default [1, 0, 0, 0, 10].
        horizon : int, optional
            Maximum number of steps per episode, by default 10.
        seed : int or None, optional
            Random seed for reproducibility, by default None.
        """
        self.rng = np.random.default_rng(seed)

        self.rewards = list(rewards)
        self.P = np.array(transition_probabilities)
        self.horizon = int(horizon)
        self.current_steps = 0
        self.position = 2  # start at middle

        # spaces
        n = self.P.shape[0]
        self.observation_space = gym.spaces.Discrete(n)
        self.action_space = gym.spaces.Discrete(2)

        # helpers
        self.states = np.arange(n)
        self.actions = np.arange(2)

        # transition matrix
        self.transition_matrix = self.T = self.get_transition_matrix()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        """
        Reset the environment to its initial state.

        Parameters
        ----------
        seed : int, optional
            Seed for environment reset (unused).
        options : dict, optional
            Additional reset options (unused).

        Returns
        -------
        state : int
            Initial state (always 2).
        info : dict
            An empty info dictionary.
        """
        self.current_steps = 0
        self.position = 2
        return self.position, {}

    def step(
        self, action: int
    ) -> tuple[int, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        Take one step in the environment.

        Parameters
        ----------
        action : int
            Action to take (0: left, 1: right).

        Returns
        -------
        next_state : int
            The resulting position of the rover.
        reward : float
            The reward at the new position.
        terminated : bool
            Whether the episode ended due to task success (always False here).
        truncated : bool
            Whether the episode ended due to reaching the time limit.
        info : dict
            An empty dictionary.
        """
        action = int(action)
        if not self.action_space.contains(action):
            raise RuntimeError(f"{action} is not a valid action (needs to be 0 or 1)")

        self.current_steps += 1

        # stochastic flip with prob 1 - P[pos, action]
        p = float(self.P[self.position, action])
        follow = self.rng.random() < p
        a_used = action if follow else 1 - action

        self.position = self.get_next_state(self.position, a_used)

        reward = float(self.rewards[self.position])
        terminated = False
        truncated = self.current_steps >= self.horizon

        return self.position, reward, terminated, truncated, {}

    def get_reward_per_action(self) -> np.ndarray:
        """
        Return the expected reward function R[s, a] for each (state, action) pair.

        R[s, a] is the expected reward resulting from taking action a in state s,
        accounting for the transition probabilities.

        Returns
        -------
        R : np.ndarray
            A (num_states, num_actions) array of expected rewards.
        """
        nS, nA = self.observation_space.n, self.action_space.n
        R = np.zeros((nS, nA), dtype=float)
        T = self.get_transition_matrix()

        for s in range(nS):
            for a in range(nA):
                expected_reward = 0.0
                for next_s in range(nS):
                    expected_reward += T[s, a, next_s] * self.rewards[next_s]
                R[s, a] = float(expected_reward)
        return R

    def get_next_state(self, state: int, action: int) -> int:
        """
        Get the next state given a state and an action (assuming deterministic execution).

        Parameters
        ----------
        state : int
            The current state.
        action : int
            The action to take.

        Returns
        -------
        next_state : int
            The resulting state.
        """
        # TODO: Implement the environment dynamics to determine the next state
        return state

    def get_transition_matrix(
        self,
        S: np.ndarray | None = None,
        A: np.ndarray | None = None,
        P: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Construct a transition matrix T[s, a, s'].

        Parameters
        ----------
        S : np.ndarray, optional
            Array of states. Uses internal states if None.
        A : np.ndarray, optional
            Array of actions. Uses internal actions if None.
        P : np.ndarray, optional
            Action success probabilities. Uses internal P if None.

        Returns
        -------
        T : np.ndarray
            A (num_states, num_actions, num_states) tensor where
            T[s, a, s'] = probability of transitioning to s' from s via a.
        """
        if S is None or A is None or P is None:
            S, A, P = self.states, self.actions, self.P

        nS, nA = len(S), len(A)
        T = np.zeros((nS, nA, nS), dtype=float)
        # TODO: Determine the transition matrix using the get_next_state function
        # and the transition probabilities P.

        return T

    def render(self, mode: str = "human"):
        """
        Render the current state of the environment.

        Parameters
        ----------
        mode : str
            Render mode (only "human" is supported).
        """
        print(f"[MarsRover] pos={self.position}, steps={self.current_steps}")


class MarsRoverPartialObsWrapper(gym.Wrapper):
    """
    Partially-observable wrapper for the MarsRover environment.

    This wrapper injects observation noise to simulate partial observability.
    With a specified probability, the true state (position) is replaced by a randomly
    selected incorrect position in the state space.

    Parameters
    ----------
    env : MarsRover
        The fully observable MarsRover environment to wrap.
    noise : float, default=0.1
        Probability in [0, 1] of returning a random incorrect position.
    seed : int or None, default=None
        Optional RNG seed for reproducibility.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, env: MarsRover, noise: float = 0.1, seed: int | None = None):
        """
        Initialize the partial observability wrapper.

        Parameters
        ----------
        env : MarsRover
            The environment to wrap.
        noise : float, optional
            Probability of observing an incorrect state, by default 0.1.
        seed : int or None, optional
            Random seed for reproducibility, by default None.
        """
        super().__init__(env)
        assert 0.0 <= noise <= 1.0, "noise must be in [0,1]"
        self.noise = noise
        self.rng = np.random.default_rng(seed)

        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        """
        Reset the base environment and return a noisy observation.

        Parameters
        ----------
        seed : int or None, optional
            Seed for the reset, by default None.
        options : dict or None, optional
            Additional reset options, by default None.

        Returns
        -------
        obs : int
            The (possibly noisy) initial observation.
        info : dict
            Additional info returned by the environment.
        """
        true_obs, info = self.env.reset(seed=seed, options=options)
        return self._noisy_obs(true_obs), info

    def step(self, action: int) -> tuple[int, float, bool, bool, dict[str, Any]]:
        """
        Take a step in the environment and return a noisy observation.

        Parameters
        ----------
        action : int
            Action to take.

        Returns
        -------
        obs : int
            The (possibly noisy) resulting observation.
        reward : float
            The reward received.
        terminated : bool
            Whether the episode terminated.
        truncated : bool
            Whether the episode was truncated due to time limit.
        info : dict
            Additional information from the base environment.
        """
        true_obs, reward, terminated, truncated, info = self.env.step(action)
        return self._noisy_obs(true_obs), reward, terminated, truncated, info

    def _noisy_obs(self, true_obs: int) -> int:
        """
        Return a possibly noisy version of the true observation.

        With probability `noise`, replaces the true observation with
        a randomly selected incorrect state.

        Parameters
        ----------
        true_obs : int
            The true observation/state index.

        Returns
        -------
        obs : int
            A noisy (or true) observation.
        """
        if self.rng.random() < self.noise:
            n = self.observation_space.n
            others = [s for s in range(n) if s != true_obs]
            return int(self.rng.choice(others))
        else:
            return int(true_obs)

    def render(self, mode: str = "human"):
        """
        Render the current state of the environment.

        Parameters
        ----------
        mode : str, optional
            Render mode, by default "human".

        Returns
        -------
        Any
            Rendered output from the base environment.
        """
        return self.env.render(mode=mode)
