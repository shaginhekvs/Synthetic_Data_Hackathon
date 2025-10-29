# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for MountainCarContinuous Environment.

This module defines the Action, Observation, and State types for MountainCarContinuous games
via the Gymnasium MountainCarContinuous-v0 environment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

from core.env_server import Action, Observation, State


@dataclass
class MountainCarContinuousAction(Action):
    """
    Action for MountainCarContinuous environments.

    MountainCarContinuous has a continuous action space with 1 action:
    - engine_force: Continuous value between -1.0 and 1.0 (negative = left, positive = right)

    Attributes:
        engine_force: The continuous engine force to apply (-1.0 to 1.0).
        render_mode: Render mode for the environment ("human", "rgb_array", None).
        max_steps: Maximum steps per episode (default: 999).
        seed: Random seed for reproducibility.
    """
    engine_force: float
    render_mode: Optional[Literal["human", "rgb_array"]] = None
    max_steps: int = 999
    seed: Optional[int] = None


@dataclass
class MountainCarContinuousObservation(Observation):
    """
    Observation from MountainCarContinuous environment.

    This represents what the agent sees after taking an action.
    MountainCarContinuous state consists of 2 continuous values:
    - position: Car position (-1.2 to 0.6)
    - velocity: Car velocity (-0.07 to 0.07)

    Attributes:
        state: Current state as a list of 2 floats [position, velocity].
        legal_actions: List of legal action bounds [min_force, max_force].
        episode_length: Current length of the episode.
        total_reward: Total reward accumulated in this episode.
    """
    state: List[float]
    legal_actions: List[float] = field(default_factory=lambda: [-1.0, 1.0])
    episode_length: int = 0
    total_reward: float = 0.0


@dataclass
class MountainCarContinuousState(State):
    """
    State for MountainCarContinuous environment.

    Attributes:
        render_mode: Render mode for the environment.
        max_steps: Maximum steps per episode.
        seed: Random seed used for the environment.
        episode_length: Current episode length.
        total_reward: Total reward accumulated in current episode.
    """
    render_mode: Optional[Literal["human", "rgb_array"]] = None
    max_steps: int = 999
    seed: Optional[int] = None
    episode_length: int = 0
    total_reward: float = 0.0
