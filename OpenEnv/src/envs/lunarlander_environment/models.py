# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for LunarLander Environment.

This module defines the Action, Observation, and State types for LunarLander
via the Gymnasium LunarLander-v3 environment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

from core.env_server import Action, Observation, State


@dataclass
class LunarLanderAction(Action):
    """
    Action for LunarLander environments.

    LunarLander has a continuous action space with 2 actions:
    - main_engine: Main engine power (-1.0 to 1.0), where -1 = no thrust, 0 = medium thrust, 1 = maximum thrust
    - lateral_engine: Lateral engine power (-1.0 to 1.0, negative=left, positive=right)

    Attributes:
        main_engine: Main engine power [-1.0, 1.0]
        lateral_engine: Lateral engine power [-1.0, 1.0]
        render_mode: Render mode for the environment ("human", "rgb_array", None).
        max_steps: Maximum steps per episode (default: 1000).
        seed: Random seed for reproducibility.
    """
    main_engine: float
    lateral_engine: float
    render_mode: Optional[Literal["human", "rgb_array"]] = None
    max_steps: int = 1000
    seed: Optional[int] = None

    def __post_init__(self):
        """Validate action parameters."""
        if not (-1.0 <= self.main_engine <= 1.0):
            raise ValueError(f"main_engine must be in [-1.0, 1.0], got {self.main_engine}")
        if not (-1.0 <= self.lateral_engine <= 1.0):
            raise ValueError(f"lateral_engine must be in [-1.0, 1.0], got {self.lateral_engine}")


@dataclass
class LunarLanderObservation(Observation):
    """
    Observation from LunarLander environment.

    This represents what the agent sees after taking an action.
    LunarLander state consists of 8 continuous values:
    - x position (lateral position)
    - y position (vertical position, 0 = surface)
    - x velocity (lateral velocity)
    - y velocity (vertical velocity)
    - angle (lander tilt, 0 = upright)
    - angular velocity
    - left leg ground contact (0 or 1)
    - right leg ground contact (0 or 1)

    Attributes:
        state: Current state as a list of 8 floats.
        legal_actions: Description of legal actions (always valid in continuous space).
        episode_length: Current length of the episode.
        total_reward: Total reward accumulated in this episode.
    """
    state: List[float]
    legal_actions: str = "continuous actions: main_engine [-1,1], lateral_engine [-1,1]"
    episode_length: int = 0
    total_reward: float = 0.0


@dataclass
class LunarLanderState(State):
    """
    State for LunarLander environment.

    Attributes:
        render_mode: Render mode for the environment.
        max_steps: Maximum steps per episode.
        seed: Random seed used for the environment.
        episode_length: Current episode length.
        total_reward: Total reward accumulated in current episode.
    """
    render_mode: Optional[Literal["human", "rgb_array"]] = None
    max_steps: int = 1000
    seed: Optional[int] = None
    episode_length: int = 0
    total_reward: float = 0.0
