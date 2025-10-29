# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
LunarLander Environment for OpenEnv.

This module provides OpenEnv integration for the LunarLander-v3 environment
via the Gymnasium library.

Example:
    >>> from envs.lunarlander_environment import LunarLanderEnv, LunarLanderAction
    >>>
    >>> # Connect to a running server or start via Docker
    >>> env = LunarLanderEnv.from_docker_image("lunarlander-env:latest")
    >>>
    >>> # Reset and interact
    >>> result = env.reset()
    >>> print(result.observation.state)  # [x_pos, y_pos, x_vel, y_vel, angle, ang_vel, left_contact, right_contact]
    >>> result = env.step(LunarLanderAction(main_engine=1.0, lateral_engine=0.0))  # Full main thrust
    >>> print(result.reward, result.done)
    >>>
    >>> # Cleanup
    >>> env.close()
"""

from .client import LunarLanderEnv
from .models import LunarLanderAction, LunarLanderObservation, LunarLanderState

__all__ = ["LunarLanderEnv", "LunarLanderAction", "LunarLanderObservation", "LunarLanderState"]
