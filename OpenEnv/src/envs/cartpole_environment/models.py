# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for Cartpole Environment.

This module defines the Action, Observation, and State types for Cartpole games
via the Gymnasium CartPole-v1 environment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

from core.env_server import Action, Observation, State


@dataclass
class CartpoleAction(Action):
    """
    Action for Cartpole environments.

    Cartpole has a discrete action space with 2 actions:
    - 0: Push cart to the left
    - 1: Push cart to the right

    Attributes:
        action_id: The integer action ID to take (0 or 1).
        render_mode: Render mode for the environment ("human", "rgb_array", None).
        max_steps: Maximum steps per episode (default: 500).
        seed: Random seed for reproducibility.
    """
    action_id: int
    render_mode: Optional[Literal["human", "rgb_array"]] = None
    max_steps: int = 10000
    seed: Optional[int] = None


@dataclass
class CartpoleObservation(Observation):
    """
    Observation from Cartpole environment.

    This represents what the agent sees after taking an action.
    Cartpole state consists of 4 continuous values:
    - cart position (-4.8 to 4.8)
    - cart velocity (-inf to inf)
    - pole angle (-24 deg to 24 deg)
    - pole velocity at tip (-inf to inf)

    Attributes:
        state: Current state as a list of 4 floats [cart_pos, cart_vel, pole_angle, pole_vel].
        legal_actions: List of legal action IDs the agent can take [0, 1].
        episode_length: Current length of the episode.
        total_reward: Total reward accumulated in this episode.
    """
    state: List[float]
    legal_actions: List[int] = field(default_factory=lambda: [0, 1])
    episode_length: int = 0
    total_reward: float = 0.0


@dataclass
class CartpoleState(State):
    """
    State for Cartpole environment.

    Attributes:
        render_mode: Render mode for the environment.
        max_steps: Maximum steps per episode.
        seed: Random seed used for the environment.
        episode_length: Current episode length.
        total_reward: Total reward accumulated in current episode.
    """
    render_mode: Optional[Literal["human", "rgb_array"]] = None
    max_steps: int = 10000
    seed: Optional[int] = None
    episode_length: int = 0
    total_reward: float = 0.0
