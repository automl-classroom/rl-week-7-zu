from __future__ import annotations

from typing import DefaultDict

import gymnasium as gym
import numpy as np


class EpsilonGreedyPolicy(object):
    """
    A policy implementing epsilon-greedy action selection.

    This policy selects a random action with probability ε (exploration),
    and the greedy (best-known) action with probability 1 - ε (exploitation),
    based on the provided Q-values.
    """

    def __init__(
        self,
        env: gym.Env,
        epsilon: float,
        seed: int = 0,
    ) -> None:
        """Initialize the epsilon-greedy policy.

        Parameters
        ----------
        env : gym.Env
            The environment providing the action space.
        epsilon : float
            Exploration rate (probability of selecting a random action).
        seed : int, optional
            Seed for random number generation, by default 0.

        Raises
        ------
        AssertionError
            If `epsilon` is not in the range [0, 1].
        """
        assert 0 <= epsilon <= 1, "ε must be in [0,1]"
        self.env = env
        self.epsilon = epsilon

        # our private RNG, so sampling is reproducible
        self.rng = np.random.default_rng(seed)

    def __call__(self, Q: DefaultDict, state: tuple, evaluate: bool = False) -> int:  # type: ignore # noqa: E501
        """Select an action for the given state using epsilon-greedy strategy.

        Parameters
        ----------
        Q : DefaultDict
            A Q-table or function mapping states to action-value arrays.
        state : tuple
            The current state.
        evaluate : bool, optional
            If True, selects the greedy action without exploration.

        Returns
        -------
        int
            The selected action
        """

        # If evaluation mode, skip exploration entirely
        if evaluate:
            return int(np.argmax(Q[state]))

        # TODO: Implement epsilon-greedy action selection
        # With prob 1 - epsilon return the greedy action
        # Wtih prob epsilon, use the policy's RNG to select a random action
        # Return the selected action -- currently always returns 0

        return 0
