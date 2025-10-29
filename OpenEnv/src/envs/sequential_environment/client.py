# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Sequential Environment HTTP Client.

This module provides the client for connecting to a Sequential Environment server
over HTTP.
"""

from __future__ import annotations

from typing import Any, Dict, TYPE_CHECKING

from core.client_types import StepResult

from core.http_env_client import HTTPEnvClient

from .models import SequentialAction, SequentialObservation, SequentialState

if TYPE_CHECKING:
    from core.containers.runtime import ContainerProvider


class SequentialEnv(HTTPEnvClient[SequentialAction, SequentialObservation]):
    """
    HTTP client for Sequential Environment.

    This client connects to a SequentialEnvironment HTTP server and provides
    methods to interact with it: reset(), step(), and state access.

    Example:
        >>> # Connect to a running server
        >>> client = SequentialEnv(base_url="http://localhost:8000")
        >>> result = client.reset()
        >>> print(result.observation.phase)  # Current active phase
        >>> print(result.observation.state)  # Concatenated [phase_vector + sub_env_state]
        >>>
        >>> # Take an action for current phase
        >>> action = SequentialAction(
        ...     cartpole_action=1,        # Push right if cartpole is active
        ...     mountaincar_action=0.5,   # Full engine if mountaincar is active
        ...     lunarlander_action=[0.0, -1.0]  # No thrust, full left if lunarlander is active
        ... )
        >>> result = client.step(action)
        >>> print(result.reward, result.done)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = SequentialEnv.from_docker_image("sequential-env:latest")
        >>> result = client.reset()
        >>> result = client.step(SequentialAction(cartpole_action=0))  # Push left
    """

    def _step_payload(self, action: SequentialAction) -> Dict[str, Any]:
        """
        Convert SequentialAction to JSON payload for step request.

        Args:
            action: SequentialAction instance.

        Returns:
            Dictionary representation suitable for JSON encoding.
        """
        payload = {}
        if action.cartpole_action is not None:
            payload["cartpole_action"] = action.cartpole_action
        if action.mountaincar_action is not None:
            payload["mountaincar_action"] = action.mountaincar_action
        if action.lunarlander_action is not None:
            payload["lunarlander_action"] = action.lunarlander_action
        if action.bipedalwalker_action is not None:
            payload["bipedalwalker_action"] = action.bipedalwalker_action
        return payload

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[SequentialObservation]:
        """
        Parse server response into StepResult[SequentialObservation].

        Args:
            payload: JSON response from server.

        Returns:
            StepResult with SequentialObservation.
        """
        obs_data = payload.get("observation", {})

        observation = SequentialObservation(
            state=obs_data.get("state", []),
            phase=obs_data.get("phase", ""),
            sub_observation=obs_data.get("sub_observation", []),
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

    def _parse_state(self, payload: Dict[str, Any]) -> SequentialState:
        """
        Parse server response into SequentialState object.

        Args:
            payload: JSON response from /state endpoint.

        Returns:
            SequentialState object with environment state information.
        """
        return SequentialState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            active_phase=payload.get("active_phase", ""),
            next_environment=payload.get("next_environment", ""),
            cartpole_done=payload.get("cartpole_done", False),
            mountaincar_done=payload.get("mountaincar_done", False),
            lunarlander_done=payload.get("lunarlander_done", False),
            episode_length=payload.get("episode_length", 0),
            total_reward=payload.get("total_reward", 0.0),
            max_steps=payload.get("max_steps", 1000),
            seed=payload.get("seed"),
        )
