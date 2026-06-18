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
        if action == 0:  # go left
            if state > 0:
                state -= 1
        elif action == 1:  # go right
            if state < self.observation_space.n - 1:
                state += 1

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
        for s in range(nS):
            for a in range(nA):
                T[s, a, self.get_next_state(s, a)] += self.P[s, a]
                T[s, a, self.get_next_state(s, 1 - a)] += 1 - self.P[s, a]

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

class ContextualMarsRover(MarsRover):
    """
    MarsRover with hidden or observed context.

    Context features:
    - action_success_prob: probability that the commanded action is executed
    - left_goal_reward: reward at the leftmost state
    - right_goal_reward: reward at the rightmost state

    If context_visible=True, the observation is encoded as:
        obs = context_id * n_positions + position

    If context_visible=False, the agent only observes position.
    """

    def __init__(
        self,
        contexts: list[dict[str, float]],
        context_visible: bool = False,
        context_change: str = "round_robin",
        horizon: int = 10,
        seed: int | None = None,
    ):
        self.contexts = contexts
        self.context_visible = context_visible
        self.context_change = context_change
        self.context_id = -1
        self.n_positions = 5

        first_context = contexts[0]
        transition_probabilities, rewards = self._context_to_mdp(first_context)

        super().__init__(
            transition_probabilities=transition_probabilities,
            rewards=rewards,
            horizon=horizon,
            seed=seed,
        )

        if self.context_visible:
            self.observation_space = gym.spaces.Discrete(
                len(self.contexts) * self.n_positions
            )
            self.states = np.arange(self.observation_space.n)
        else:
            self.observation_space = gym.spaces.Discrete(self.n_positions)
            self.states = np.arange(self.n_positions)

        self.actions = np.arange(2)
        self.transition_matrix = self.T = self.get_transition_matrix()

    # Convert one context into the transition probabilities and rewards
    # needed by the tabular planning algorithms.
    def _context_to_mdp(
        self, context: dict[str, float]
    ) -> tuple[np.ndarray, list[float]]:
        action_success_prob = float(context["action_success_prob"])
        right_goal_reward = float(context["right_goal_reward"])
        left_goal_reward = float(context["left_goal_reward"])

        transition_probabilities = np.full(
            (self.n_positions, 2),
            action_success_prob,
            dtype=float,
        )

        rewards = [left_goal_reward, 0.0, 0.0, 0.0, right_goal_reward]
        return transition_probabilities, rewards

    # We rotate through the context list after every reset.
    def _advance_context(self) -> None:
        if self.context_change != "round_robin":
            raise ValueError(f"Unknown context_change: {self.context_change}")

        self.context_id = (self.context_id + 1) % len(self.contexts)
        self.P, self.rewards = self._context_to_mdp(self.contexts[self.context_id])

    # If context is visible, each context gets its own copy of the 5 positions.
    def _encode_obs(self, position: int, context_id: int | None = None) -> int:
        if not self.context_visible:
            return position

        if context_id is None:
            context_id = self.context_id

        return context_id * self.n_positions + position

    def _decode_obs(self, obs: int) -> tuple[int, int]:
        if self.context_visible:
            context_id = obs // self.n_positions
            position = obs % self.n_positions
            return context_id, position

        return self.context_id, obs

    def reset(self, *, seed=None, options=None):
        self.current_steps = 0
        self.position = 2
        self._advance_context()
        return self._encode_obs(self.position), {"context": self.contexts[self.context_id]}

    def step(self, action: int):
        action = int(action)
        if not self.action_space.contains(action):
            raise RuntimeError(f"{action} is not a valid action")

        self.current_steps += 1

        p = float(self.P[self.position, action])
        follow = self.rng.random() < p
        a_used = action if follow else 1 - action

        self.position = self._move_position(self.position, a_used)

        reward = float(self.rewards[self.position])
        terminated = False
        truncated = self.current_steps >= self.horizon

        return (
            self._encode_obs(self.position),
            reward,
            terminated,
            truncated,
            {"context": self.contexts[self.context_id]},
        )

    # Keep movement bounded to the 5 physical rover positions.
    def _move_position(self, position: int, action: int) -> int:
        if action == 0:
            return max(0, position - 1)

        if action == 1:
            return min(self.n_positions - 1, position + 1)

        raise RuntimeError(f"{action} is not a valid action")

    def get_next_state(self, state: int, action: int) -> int:
        context_id, position = self._decode_obs(state)
        next_position = self._move_position(position, action)

        if self.context_visible:
            return self._encode_obs(next_position, context_id)

        return next_position

    def get_transition_matrix(self, S=None, A=None, P=None):
        if self.context_visible:
            return self._get_context_visible_transition_matrix()

        return self._get_context_hidden_transition_matrix()

    # With visible context, the transition matrix keeps contexts separated.
    def _get_context_visible_transition_matrix(self):
        nS = len(self.contexts) * self.n_positions
        nA = 2
        T = np.zeros((nS, nA, nS), dtype=float)

        for c_id, context in enumerate(self.contexts):
            P, _ = self._context_to_mdp(context)

            for pos in range(self.n_positions):
                s = self._encode_obs(pos, c_id)

                for a in range(nA):
                    intended = self._move_position(pos, a)
                    flipped = self._move_position(pos, 1 - a)

                    T[s, a, self._encode_obs(intended, c_id)] += P[pos, a]
                    T[s, a, self._encode_obs(flipped, c_id)] += 1.0 - P[pos, a]

        return T

    # Without context in the observation, the agent plans on the average MDP.
    def _get_context_hidden_transition_matrix(self):
        nS = self.n_positions
        nA = 2
        T = np.zeros((nS, nA, nS), dtype=float)

        for context in self.contexts:
            P, _ = self._context_to_mdp(context)

            for s in range(nS):
                for a in range(nA):
                    intended = self._move_position(s, a)
                    flipped = self._move_position(s, 1 - a)

                    T[s, a, intended] += P[s, a] / len(self.contexts)
                    T[s, a, flipped] += (1.0 - P[s, a]) / len(self.contexts)

        return T

    # Policy iteration needs expected rewards for every state-action pair.
    def get_reward_per_action(self):
        T = self.get_transition_matrix()
        nS, nA, _ = T.shape
        R = np.zeros((nS, nA), dtype=float)

        for s in range(nS):
            for a in range(nA):
                for next_s in range(nS):
                    if self.context_visible:
                        c_id, next_pos = self._decode_obs(next_s)
                        _, rewards = self._context_to_mdp(self.contexts[c_id])
                    else:
                        next_pos = next_s
                        rewards = np.mean(
                            [
                                self._context_to_mdp(context)[1][next_pos]
                                for context in self.contexts
                            ]
                        )
                        R[s, a] += T[s, a, next_s] * rewards
                        continue

                    R[s, a] += T[s, a, next_s] * rewards[next_pos]

        return R

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
