# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Cartpole Environment for OpenEnv.

This module provides OpenEnv integration for the CartPole-v1 environment
via the Gymnasium library.

Example:
    >>> from envs.cartpole_environment import CartpoleEnv, CartpoleAction
    >>>
    >>> # Connect to a running server or start via Docker
    >>> env = CartpoleEnv.from_docker_image("cartpole-env:latest")
    >>>
    >>> # Reset and interact
    >>> result = env.reset()
    >>> print(result.observation.state)  # [cart_pos, cart_vel, pole_angle, pole_vel]
    >>> result = env.step(CartpoleAction(action_id=1))  # Push right
    >>> print(result.reward, result.done)
    >>>
    >>> # Cleanup
    >>> env.close()
"""

from .client import CartpoleEnv
from .models import CartpoleAction, CartpoleObservation, CartpoleState

__all__ = ["CartpoleEnv", "CartpoleAction", "CartpoleObservation", "CartpoleState"]
