# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Atari Environment HTTP Client.

This module provides the client for connecting to an Atari Environment server
over HTTP.
"""

from __future__ import annotations

from typing import Any, Dict, TYPE_CHECKING

from core.client_types import StepResult

from core.http_env_client import HTTPEnvClient

from .models import AtariAction, AtariObservation, AtariState

if TYPE_CHECKING:
    from core.containers.runtime import ContainerProvider


class AtariEnv(HTTPEnvClient[AtariAction, AtariObservation]):
    """
    HTTP client for Atari Environment.

    This client connects to an AtariEnvironment HTTP server and provides
    methods to interact with it: reset(), step(), and state access.

    Example:
        >>> # Connect to a running server
        >>> client = AtariEnv(base_url="http://localhost:8000")
        >>> result = client.reset()
        >>> print(result.observation.screen_shape)
        >>>
        >>> # Take an action
        >>> result = client.step(AtariAction(action_id=2))  # UP
        >>> print(result.reward, result.done)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = AtariEnv.from_docker_image("atari-env:latest")
        >>> result = client.reset()
        >>> result = client.step(AtariAction(action_id=0))  # NOOP
    """

    def _step_payload(self, action: AtariAction) -> Dict[str, Any]:
        """
        Convert AtariAction to JSON payload for step request.

        Args:
            action: AtariAction instance.

        Returns:
            Dictionary representation suitable for JSON encoding.
        """
        return {
            "action_id": action.action_id,
            "game_name": action.game_name,
            "obs_type": action.obs_type,
            "full_action_space": action.full_action_space,
        }

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[AtariObservation]:
        """
        Parse server response into StepResult[AtariObservation].

        Args:
            payload: JSON response from server.

        Returns:
            StepResult with AtariObservation.
        """
        obs_data = payload.get("observation", {})

        observation = AtariObservation(
            screen=obs_data.get("screen", []),
            screen_shape=obs_data.get("screen_shape", []),
            legal_actions=obs_data.get("legal_actions", []),
            lives=obs_data.get("lives", 0),
            episode_frame_number=obs_data.get("episode_frame_number", 0),
            frame_number=obs_data.get("frame_number", 0),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> AtariState:
        """
        Parse server response into AtariState object.

        Args:
            payload: JSON response from /state endpoint.

        Returns:
            AtariState object with environment state information.
        """
        return AtariState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            game_name=payload.get("game_name", "unknown"),
            obs_type=payload.get("obs_type", "rgb"),
            full_action_space=payload.get("full_action_space", False),
            mode=payload.get("mode"),
            difficulty=payload.get("difficulty"),
            repeat_action_probability=payload.get("repeat_action_probability", 0.0),
            frameskip=payload.get("frameskip", 4),
        )
