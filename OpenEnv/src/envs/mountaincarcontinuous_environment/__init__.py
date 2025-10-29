# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
MountainCarContinuous Environment for OpenEnv.

This module provides OpenEnv integration for the MountainCarContinuous-v0 environment
via the Gymnasium library.

Example:
    >>> from envs.mountaincarcontinuous_environment import MountainCarContinuousEnv, MountainCarContinuousAction
    >>>
    >>> # Connect to a running server or start via Docker
    >>> env = MountainCarContinuousEnv.from_docker_image("mountaincarcontinuous-env:latest")
    >>>
    >>> # Reset and interact
    >>> result = env.reset()
    >>> print(result.observation.state)  # [position, velocity]
    >>> result = env.step(MountainCarContinuousAction(engine_force=0.5))  # Apply right force
    >>> print(result.reward, result.done)
    >>>
    >>> # Cleanup
    >>> env.close()
"""

from .client import MountainCarContinuousEnv
from .models import MountainCarContinuousAction, MountainCarContinuousObservation, MountainCarContinuousState

__all__ = ["MountainCarContinuousEnv", "MountainCarContinuousAction", "MountainCarContinuousObservation", "MountainCarContinuousState"]

