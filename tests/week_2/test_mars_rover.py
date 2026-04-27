# tests/test_template_env.py

import gymnasium
import numpy as np
import pytest
from rl_exercises.environments import (  # adjust import path as needed
    MarsRover,
    MarsRoverPartialObsWrapper,
)


def test_env_has_spaces_and_methods():
    env = MarsRover()

    # spaces exist
    assert hasattr(env, "observation_space")
    assert isinstance(env.observation_space, gymnasium.spaces.Space)
    assert hasattr(env, "action_space")
    assert isinstance(env.action_space, gymnasium.spaces.Space)

    # reset API
    obs, info = env.reset(seed=0)
    assert isinstance(obs, int)
    assert env.observation_space.contains(obs)
    assert isinstance(info, dict)

    # step API
    action = env.action_space.sample()
    next_obs, reward, terminated, truncated, info2 = env.step(action)
    assert isinstance(next_obs, int)
    assert env.observation_space.contains(next_obs)
    assert isinstance(reward, (int, float))
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info2, dict)

    # invalid action
    with pytest.raises(RuntimeError):
        env.step(-1)


def test_reward_and_transition_methods_exist_and_shape():
    env = MarsRover()
    # get_reward_per_action
    assert hasattr(env, "get_reward_per_action")
    R = env.get_reward_per_action()
    assert isinstance(R, np.ndarray)
    assert R.shape == (env.observation_space.n, env.action_space.n)

    # get_transition_matrix
    assert hasattr(env, "get_transition_matrix")
    T = env.get_transition_matrix()
    assert isinstance(T, np.ndarray)
    assert T.shape == (
        env.observation_space.n,
        env.action_space.n,
        env.observation_space.n,
    )
    # probabilities should be between 0 and 1
    assert np.all(T >= 0) and np.all(T <= 1)


def test_partial_obs_zero_noise():
    """With noise=0, wrapper observations equal true state."""
    base = MarsRover(seed=0)
    wrapper = MarsRoverPartialObsWrapper(base, noise=0.0, seed=0)

    obs, _ = wrapper.reset(seed=0)
    assert obs == base.position

    for action in [0, 1, 1, 0]:
        noisy_obs, _, _, _, _ = wrapper.step(action)
        assert noisy_obs == base.position, (
            "With zero noise, wrapper obs must match true state"
        )


def test_partial_obs_full_noise():
    """With noise=1, wrapper observations never equal true state."""
    base = MarsRover(seed=42)
    wrapper = MarsRoverPartialObsWrapper(base, noise=1.0, seed=42)

    obs, _ = wrapper.reset()
    assert obs != base.position, (
        "With full noise, reset obs must differ from true state"
    )

    for action in [0, 1, 1, 0]:
        noisy_obs, _, _, _, _ = wrapper.step(action)
        assert noisy_obs != base.position, (
            "With full noise, step obs must differ from true state"
        )
