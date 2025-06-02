from __future__ import annotations

from typing import Any, SupportsFloat

import compiler_gym
import gymnasium
from gymnasium.core import Env


class SpaceWrapper(gymnasium.Space):
    """A wrapper around a :code:`gym.spaces.Space` with additional functionality
    for action spaces.
    """

    @property  # type: ignore[misc]
    def __class__(self) -> gymnasium.Space:
        """Fake class"""
        return self.desired_space

    def __init__(
        self, space: compiler_gym.spaces.ActionSpace, desired_space: gymnasium.Space
    ) -> None:
        """Constructor.

        :param space: The space that this action space wraps.
        """
        self.wrapped = space
        self.desired_space = desired_space

    def __getattr__(self, name: str) -> Any:
        return getattr(self.wrapped, name)

    def __getitem__(self, name: str) -> Any:
        return self.wrapped[name]


class ActionWrapper(gymnasium.Wrapper):
    """Convert action to the target type"""

    def __init__(self, env: Env, target_type: type) -> None:
        super().__init__(env)
        self.target_type = target_type

    def step(
        self, action: Any
    ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        """Convert action to target type before stepping env"""
        action = self.target_type(action)
        return super().step(action)
