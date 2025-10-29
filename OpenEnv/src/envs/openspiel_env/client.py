# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
OpenSpielEnv HTTP Client.

This module provides the client for connecting to an OpenSpiel Environment server
over HTTP.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, TYPE_CHECKING

from core.client_types import StepResult

from core.http_env_client import HTTPEnvClient

from .models import OpenSpielAction, OpenSpielObservation, OpenSpielState

if TYPE_CHECKING:
    from core.containers.runtime import ContainerProvider


class OpenSpielEnv(HTTPEnvClient[OpenSpielAction, OpenSpielObservation]):
    """
    HTTP client for OpenSpiel Environment.

    This client connects to an OpenSpielEnvironment HTTP server and provides
    methods to interact with it: reset(), step(), and state access.

    Example:
        >>> # Connect to a running server
        >>> client = OpenSpielEnv(base_url="http://localhost:8000")
        >>> result = client.reset()
        >>> print(result.observation.info_state)
        >>>
        >>> # Take an action
        >>> result = client.step(OpenSpielAction(action_id=1, game_name="catch"))
        >>> print(result.observation.reward)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = OpenSpielEnv.from_docker_image("openspiel-env:latest")
        >>> result = client.reset()
        >>> result = client.step(OpenSpielAction(action_id=0))
    """

    def _step_payload(self, action: OpenSpielAction) -> Dict[str, Any]:
        """
        Convert OpenSpielAction to JSON payload for step request.

        Args:
            action: OpenSpielAction instance.

        Returns:
            Dictionary representation suitable for JSON encoding.
        """
        return {
            "action_id": action.action_id,
            "game_name": action.game_name,
            "game_params": action.game_params,
        }

    def _parse_result(
        self, payload: Dict[str, Any]
    ) -> StepResult[OpenSpielObservation]:
        """
        Parse server response into StepResult[OpenSpielObservation].

        Args:
            payload: JSON response from server.

        Returns:
            StepResult with OpenSpielObservation.
        """
        obs_data = payload.get("observation", {})

        observation = OpenSpielObservation(
            info_state=obs_data.get("info_state", []),
            legal_actions=obs_data.get("legal_actions", []),
            game_phase=obs_data.get("game_phase", "playing"),
            current_player_id=obs_data.get("current_player_id", 0),
            opponent_last_action=obs_data.get("opponent_last_action"),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> OpenSpielState:
        """
        Parse server response into OpenSpielState object.

        Args:
            payload: JSON response from /state endpoint.

        Returns:
            OpenSpielState object with environment state information.
        """
        return OpenSpielState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            game_name=payload.get("game_name", "unknown"),
            agent_player=payload.get("agent_player", 0),
            opponent_policy=payload.get("opponent_policy", "random"),
            game_params=payload.get("game_params", {}),
            num_players=payload.get("num_players", 1),
        )
