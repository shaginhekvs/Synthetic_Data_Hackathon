# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
MountainCarContinuous Environment HTTP Client.

This module provides the client for connecting to a MountainCarContinuous Environment server
over HTTP.
"""

from __future__ import annotations

from typing import Any, Dict, TYPE_CHECKING

from core.client_types import StepResult

from core.http_env_client import HTTPEnvClient

from .models import MountainCarContinuousAction, MountainCarContinuousObservation, MountainCarContinuousState

if TYPE_CHECKING:
    from core.containers.runtime import ContainerProvider


class MountainCarContinuousEnv(HTTPEnvClient[MountainCarContinuousAction, MountainCarContinuousObservation]):
    """
    HTTP client for MountainCarContinuous Environment.

    This client connects to a MountainCarContinuousEnvironment HTTP server and provides
    methods to interact with it: reset(), step(), and state access.

    Example:
        >>> # Connect to a running server
        >>> client = MountainCarContinuousEnv(base_url="http://localhost:8000")
        >>> result = client.reset()
        >>> print(result.observation.state)  # [position, velocity]
        >>>
        >>> # Take an action
        >>> result = client.step(MountainCarContinuousAction(engine_force=0.5))  # Apply right force
        >>> print(result.reward, result.done)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = MountainCarContinuousEnv.from_docker_image("mountaincarcontinuous-env:latest")
        >>> result = client.reset()
        >>> result = client.step(MountainCarContinuousAction(engine_force=-0.3))  # Apply left force
    """

    def _step_payload(self, action: MountainCarContinuousAction) -> Dict[str, Any]:
        """
        Convert MountainCarContinuousAction to JSON payload for step request.

        Args:
            action: MountainCarContinuousAction instance.

        Returns:
            Dictionary representation suitable for JSON encoding.
        """
        return {
            "engine_force": action.engine_force,
            "render_mode": action.render_mode,
            "max_steps": action.max_steps,
            "seed": action.seed,
        }

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[MountainCarContinuousObservation]:
        """
        Parse server response into StepResult[MountainCarContinuousObservation].

        Args:
            payload: JSON response from server.

        Returns:
            StepResult with MountainCarContinuousObservation.
        """
        obs_data = payload.get("observation", {})

        observation = MountainCarContinuousObservation(
            state=obs_data.get("state", []),
            legal_actions=obs_data.get("legal_actions", [-1.0, 1.0]),
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

    def _parse_state(self, payload: Dict[str, Any]) -> MountainCarContinuousState:
        """
        Parse server response into MountainCarContinuousState object.

        Args:
            payload: JSON response from /state endpoint.

        Returns:
            MountainCarContinuousState object with environment state information.
        """
        return MountainCarContinuousState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            render_mode=payload.get("render_mode"),
            max_steps=payload.get("max_steps", 999),
            seed=payload.get("seed"),
            episode_length=payload.get("episode_length", 0),
            total_reward=payload.get("total_reward", 0.0),
        )
