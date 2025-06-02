from __future__ import annotations

from typing import Any, DefaultDict

from collections import defaultdict

import gymnasium as gym
import numpy as np
from rl_exercises.agent import AbstractAgent
from rl_exercises.week_3 import EpsilonGreedyPolicy


class SARSAAgent(AbstractAgent):
    """SARSA algorithm"""

    def __init__(
        self,
        env: gym.Env,
        policy: EpsilonGreedyPolicy,
        alpha: float = 0.5,
        gamma: float = 1.0,
    ) -> None:
        """Initialize the SARSA agent.

        Parameters
        ----------
        env : gym.Env
            The environment in which the agent will interact.
        policy : EpsilonGreedyPolicy
            Policy for selecting actions, typically epsilon-greedy.
        alpha : float
            Learning rate (step size for updates), by default 0.5.
        gamma : float
            Discount factor for future rewards, by default 1.0.

        Raises
        ------
        AssertionError
            If `gamma` is not in [0, 1] or if `alpha` is not positive.
        """

        # Check hyperparameter boundaries
        assert 0 <= gamma <= 1, "Gamma should be in [0, 1]"
        assert alpha > 0, "Learning rate has to be greater than 0"

        self.env = env
        self.gamma = gamma
        self.alpha = alpha

        # number of actions → used by Q’s default factory
        self.n_actions = env.action_space.n

        # Build Q so that unseen states map to zero‐vectors
        self.Q: DefaultDict[Any, np.ndarray] = defaultdict(
            lambda: np.zeros(self.n_actions, dtype=float)
        )

        self.policy = policy

    def predict_action(self, state: np.array, evaluate: bool = False) -> Any:  # type: ignore # noqa
        """Select an action for the given state using the policy.

        Parameters
        ----------
        state : np.array
            The current state.
        evaluate : bool, optional
            If True, use the greedy policy without exploration; otherwise use exploration.

        Returns
        -------
        Any
            The selected action.
        """
        return self.policy(self.Q, state, evaluate=evaluate)

    def save(self, path: str) -> Any:  # type: ignore
        """Save the learned Q-table to a file.

        Parameters
        ----------
        path : str
            Path to the file where the Q-table will be saved (.npy format).
        """
        np.save(path, self.Q)  # type: ignore

    def load(self, path) -> Any:  # type: ignore
        """Load the Q table

        Parameters
        ----------
        path :
            Path to saved the Q table

        """
        self.Q = np.load(path)

    def update_agent(  # type: ignore
        self,
        state: Any,
        action: int,
        reward: float,
        next_state: Any,
        next_action: int,
        done: bool,
    ) -> float:
        """Update the Q-value for a state-action pair using the SARSA update rule.

        Parameters
        ----------
        state : State
            The current state.
        action : int
            The action taken in the current state.
        reward : float
            The reward received after taking the action.
        next_state : State
            The next state after taking the action.
        next_action : int
            The action selected in the next state.
        done : bool
            True if the episode has ended; False otherwise.

        Returns
        -------
        float
            The updated Q-value for the (state, action) pair.
        """

        # SARSA update rule
        # TODO: Implement the SARSA update rule here.
        # Use a value of 0. for terminal states and
        # update the new Q value in the Q table of this class.
        # Return the new Q value --currently always returns 0.0

        return 0.0
