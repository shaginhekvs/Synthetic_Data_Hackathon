# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for Sequential Environment.

This module defines the Action, Observation, and State types for a sequential environment
that interleaves Cartpole, MountainCarContinuous, and LunarLanderContinuous phases.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

from core.env_server import Action, Observation, State


@dataclass
class SequentialAction(Action):
    """
    Action for Sequential environments.

    Actions are provided for all sub-environments but only the one corresponding
    to the currently active phase will be used.

    Attributes:
        cartpole_action: Optional integer action for cartpole phase (0 or 1).
        mountaincar_action: Optional float action for mountaincar phase (-1.0 to 1.0).
        lunarlander_action: Optional list of 2 floats for lunarlander phase (main_engine, lateral_engine).
        bipedalwalker_action: Optional list of 4 floats for bipedalwalker phase (hip1, knee1, hip2, knee2 torques).
    """
    cartpole_action: Optional[int] = None
    mountaincar_action: Optional[float] = None
    lunarlander_action: Optional[List[float]] = None
    bipedalwalker_action: Optional[List[float]] = None


@dataclass
class SequentialObservation(Observation):
    """
    Observation from Sequential environment.

    Concatenates phase information with the current sub-environment observation:
    - First 4 elements: one-hot phase vector [cartpole, mountaincar, lunarlander, bipedalwalker]
    - Remaining elements: raw sub-environment observation

    Attributes:
        state: Concatenated state vector.
        phase: Current active phase name.
        sub_observation: Raw observation from the active sub-environment.
        episode_length: Current episode length across all phases.
        total_reward: Total reward accumulated across all phases.
    """
    state: List[float]
    phase: str
    sub_observation: List[float]
    episode_length: int = 0
    total_reward: float = 0.0


@dataclass
class SequentialState(State):
    """
    State for Sequential environment.

    Attributes:
        active_phase: Currently active environment phase.
        next_environment: Which environment will be stepped next.
        cartpole_done: Whether cartpole has reached done state.
        mountaincar_done: Whether mountaincar has reached done state.
        lunarlander_done: Whether lunarlander has reached done state.
        bipedalwalker_done: Whether bipedalwalker has reached done state.
        episode_length: Current episode length across all phases.
        total_reward: Total reward accumulated across all phases.
        max_steps: Maximum steps per episode.
        seed: Random seed for phase selection.
    """
    active_phase: str = ""
    next_environment: str = ""
    cartpole_done: bool = False
    mountaincar_done: bool = False
    lunarlander_done: bool = False
    bipedalwalker_done: bool = False
    episode_length: int = 0
    total_reward: float = 0.0
    max_steps: int = 1000
    seed: Optional[int] = None
