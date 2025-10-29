# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Cartpole Environment HTTP Client.

This module provides the client for connecting to a Cartpole Environment server
over HTTP.
"""

from __future__ import annotations

from typing import Any, Dict, TYPE_CHECKING

from core.client_types import StepResult

from core.http_env_client import HTTPEnvClient

from .models import CartpoleAction, CartpoleObservation, CartpoleState

if TYPE_CHECKING:
    from core.containers.runtime import ContainerProvider


class CartpoleEnv(HTTPEnvClient[CartpoleAction, CartpoleObservation]):
    """
    HTTP client for Cartpole Environment.

    This client connects to a CartpoleEnvironment HTTP server and provides
    methods to interact with it: reset(), step(), and state access.

    Example:
        >>> # Connect to a running server
        >>> client = CartpoleEnv(base_url="http://localhost:8000")
        >>> result = client.reset()
        >>> print(result.observation.state)  # [cart_pos, cart_vel, pole_angle, pole_vel]
        >>>
        >>> # Take an action
        >>> result = client.step(CartpoleAction(action_id=1))  # Push right
        >>> print(result.reward, result.done)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = CartpoleEnv.from_docker_image("cartpole-env:latest")
        >>> result = client.reset()
        >>> result = client.step(CartpoleAction(action_id=0))  # Push left
    """

    def _step_payload(self, action: CartpoleAction) -> Dict[str, Any]:
        """
        Convert CartpoleAction to JSON payload for step request.

        Args:
            action: CartpoleAction instance.

        Returns:
            Dictionary representation suitable for JSON encoding.
        """
        return {
            "action_id": action.action_id,
            "render_mode": action.render_mode,
            "max_steps": action.max_steps,
            "seed": action.seed,
        }

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[CartpoleObservation]:
        """
        Parse server response into StepResult[CartpoleObservation].

        Args:
            payload: JSON response from server.

        Returns:
            StepResult with CartpoleObservation.
        """
        obs_data = payload.get("observation", {})

        observation = CartpoleObservation(
            state=obs_data.get("state", []),
            legal_actions=obs_data.get("legal_actions", [0, 1]),
            episode_length=obs_data.get("episode_length", 0),
            total_reward=obs_data.get("total_reward", 0.0),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> CartpoleState:
        """
        Parse server response into CartpoleState object.

        Args:
            payload: JSON response from /state endpoint.

        Returns:
            CartpoleState object with environment state information.
        """
        return CartpoleState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            render_mode=payload.get("render_mode"),
            max_steps=payload.get("max_steps", 10000),
            seed=payload.get("seed"),
            episode_length=payload.get("episode_length", 0),
            total_reward=payload.get("total_reward", 0.0),
        )
