# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
LunarLander Environment HTTP Client.

This module provides the client for connecting to a LunarLander Environment server
over HTTP.
"""

from __future__ import annotations

from typing import Any, Dict, TYPE_CHECKING

from core.client_types import StepResult

from core.http_env_client import HTTPEnvClient

from .models import LunarLanderAction, LunarLanderObservation, LunarLanderState

if TYPE_CHECKING:
    from core.containers.runtime import ContainerProvider


class LunarLanderEnv(HTTPEnvClient[LunarLanderAction, LunarLanderObservation]):
    """
    HTTP client for LunarLander Environment.

    This client connects to a LunarLanderEnvironment HTTP server and provides
    methods to interact with it: reset(), step(), and state access.

    Example:
        >>> # Connect to a running server
        >>> client = LunarLanderEnv(base_url="http://localhost:8070")
        >>> result = client.reset()
        >>> print(result.observation.state)  # [x_pos, y_pos, x_vel, y_vel, angle, ang_vel, left_contact, right_contact]
        >>>
        >>> # Take an action
        >>> result = client.step(LunarLanderAction(main_engine=1.0, lateral_engine=0.0))  # Full main thrust
        >>> print(result.reward, result.done)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = LunarLanderEnv.from_docker_image("lunarlander-env:latest")
        >>> result = client.reset()
        >>> result = client.step(LunarLanderAction(main_engine=0.5, lateral_engine=-0.2))  # Moderate thrust left
    """

    def _step_payload(self, action: LunarLanderAction) -> Dict[str, Any]:
        """
        Convert LunarLanderAction to JSON payload for step request.

        Args:
            action: LunarLanderAction instance.

        Returns:
            Dictionary representation suitable for JSON encoding.
        """
        return {
            "main_engine": action.main_engine,
            "lateral_engine": action.lateral_engine,
            "render_mode": action.render_mode,
            "max_steps": action.max_steps,
            "seed": action.seed,
        }

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[LunarLanderObservation]:
        """
        Parse server response into StepResult[LunarLanderObservation].

        Args:
            payload: JSON response from server.

        Returns:
            StepResult with LunarLanderObservation.
        """
        obs_data = payload.get("observation", {})

        observation = LunarLanderObservation(
            state=obs_data.get("state", []),
            legal_actions=obs_data.get("legal_actions", "continuous actions: main_engine [0,1], lateral_engine [-1,1]"),
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

    def _parse_state(self, payload: Dict[str, Any]) -> LunarLanderState:
        """
        Parse server response into LunarLanderState object.

        Args:
            payload: JSON response from /state endpoint.

        Returns:
            LunarLanderState object with environment state information.
        """
        return LunarLanderState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            render_mode=payload.get("render_mode"),
            max_steps=payload.get("max_steps", 1000),
            seed=payload.get("seed"),
            episode_length=payload.get("episode_length", 0),
            total_reward=payload.get("total_reward", 0.0),
        )
