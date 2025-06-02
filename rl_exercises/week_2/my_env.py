from __future__ import annotations

import gymnasium as gym


# ------------- TODO: Implement the following environment -------------
class MyEnv(gym.Env):
    """
    Simple 2-state, 2-action environment with deterministic transitions.

    Actions
    -------
    Discrete(2):
    - 0: move to state 0
    - 1: move to state 1

    Observations
    ------------
    Discrete(2): the current state (0 or 1)

    Reward
    ------
    Equal to the action taken.

    Start/Reset State
    -----------------
    Always starts in state 0.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self):
        """Initializes the observation and action space for the environment."""
        pass


class PartialObsWrapper(gym.Wrapper):
    """Wrapper that makes the underlying env partially observable by injecting
    observation noise: with probability `noise`, the true state is replaced by
    a random (incorrect) observation.

    Parameters
    ----------
    env : gym.Env
        The fully observable base environment.
    noise : float, default=0.1
        Probability in [0,1] of seeing a random wrong observation instead
        of the true one.
    seed : int | None, default=None
        Optional RNG seed for reproducibility.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, env: gym.Env, noise: float = 0.1, seed: int | None = None):
        pass
