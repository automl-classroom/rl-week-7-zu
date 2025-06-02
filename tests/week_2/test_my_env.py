# tests/test_template_env.py

import gymnasium
import numpy as np
import pytest
from rl_exercises.week_2.my_env import MyEnv  # adjust import path as needed


def test_env_has_spaces_and_methods():
    env = MyEnv()

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
    env = MyEnv()
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
