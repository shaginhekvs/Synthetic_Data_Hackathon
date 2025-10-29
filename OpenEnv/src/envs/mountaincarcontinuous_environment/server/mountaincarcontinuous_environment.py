# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
MountainCarContinuous Environment Server Implementation.

This module wraps Gymnasium's MountainCarContinuous-v0 environment and exposes it
via the OpenEnv Environment interface with comprehensive logging.
"""

import logging
import uuid
from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np

from core.env_server import Environment

from ..models import MountainCarContinuousAction, MountainCarContinuousObservation, MountainCarContinuousState

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class MountainCarContinuousEnvironment(Environment):
    """
    MountainCarContinuous Environment wrapper for OpenEnv.

    This environment wraps Gymnasium's MountainCarContinuous-v0 environment and provides
    a clean interface for RL training with comprehensive logging.

    The MountainCarContinuous task involves driving an underpowered car up a mountain
    by applying continuous engine forces. The episode ends when the car reaches the goal
    position or the maximum episode length is reached.

    Args:
        render_mode: Render mode for visualization ("human", "rgb_array", None).
        max_steps: Maximum steps per episode (default: 999, minimum: 999).
        seed: Random seed for reproducibility.

    Example:
        >>> env = MountainCarContinuousEnvironment(render_mode="human", max_steps=1000)
        >>> obs = env.reset()
        >>> print(obs.state)  # [position, velocity]
        >>> obs = env.step(MountainCarContinuousAction(engine_force=0.5))  # Apply right force
        >>> print(obs.reward, obs.done)
    """

    def __init__(
        self,
        render_mode: Optional[str] = None,
        max_steps: int = 999,
        seed: Optional[int] = None,
    ):
        """Initialize MountainCarContinuous environment."""
        super().__init__()

        self.render_mode = render_mode
        self.max_steps = max(999, max_steps)  # Ensure minimum 999 steps
        self.seed = seed

        # Create gymnasium environment
        logger.info(f"Creating MountainCarContinuous-v0 environment with render_mode={render_mode}, max_steps={self.max_steps}, seed={seed}")
        self.env = gym.make("MountainCarContinuous-v0", render_mode=render_mode)

        # Standardize episode length (Gymnasium wrapper)
        if max_steps > 0:
            self.env = gym.wrappers.TimeLimit(self.env, max_episode_steps=self.max_steps)

        # Set seed if provided
        if seed is not None:
            self.env.reset(seed=seed)
            logger.info(f"Environment seeded with seed={seed}")

        # Initialize state
        self._state = MountainCarContinuousState(
            render_mode=render_mode,
            max_steps=self.max_steps,
            seed=seed,
        )

        logger.info("MountainCarContinuousEnvironment initialized successfully")

    def reset(self) -> MountainCarContinuousObservation:
        """
        Reset the environment and return initial observation.

        Returns:
            Initial observation for the agent.
        """
        logger.info("Resetting environment")

        # Reset gymnasium environment
        self.seed = self.seed + 1 if self.seed is not None else None
        obs, info = self.env.reset(seed=self.seed)

        # Reset state tracking
        self._state.episode_id = str(uuid.uuid4())
        self._state.step_count = 0
        self._state.episode_length = 0
        self._state.total_reward = 0.0

        # Create observation
        observation = self._make_observation(obs, reward=0.0, done=False)

        logger.info(f"Environment reset - Episode {self._state.episode_id} started")
        logger.info(f"Initial state: {obs}")

        return observation

    def step(self, action: MountainCarContinuousAction) -> MountainCarContinuousObservation:
        """
        Execute agent's action and return resulting observation.

        Args:
            action: MountainCarContinuousAction containing the engine_force to execute.

        Returns:
            Observation after action execution.

        Raises:
            ValueError: If action is not a MountainCarContinuousAction or engine_force is invalid.
        """
        if not isinstance(action, MountainCarContinuousAction):
            raise ValueError(f"Expected MountainCarContinuousAction, got {type(action)}")

        # Validate engine_force
        if not (-1.0 <= action.engine_force <= 1.0):
            raise ValueError(f"Invalid engine_force: {action.engine_force}. Valid range: [-1.0, 1.0]")

        logger.info(f"Taking action: engine_force={action.engine_force}")

        # Execute action in gymnasium environment
        obs, reward, terminated, truncated, info = self.env.step(np.array([action.engine_force]))

        # Update state tracking
        self._state.step_count += 1
        self._state.episode_length += 1
        self._state.total_reward += reward

        # Determine if episode is done
        done = terminated or truncated

        # Create observation
        observation = self._make_observation(obs, reward, done)

        # Log step details
        logger.info(f"Step {self._state.step_count}: reward={reward}, terminated={terminated}, truncated={truncated}")
        logger.info(f"New state: {obs}")
        logger.info(f"Episode progress: length={self._state.episode_length}, total_reward={self._state.total_reward}")

        if done:
            logger.info(f"Episode {self._state.episode_id} ended - length={self._state.episode_length}, total_reward={self._state.total_reward}")
            if terminated:
                logger.info("Episode ended due to termination (goal reached)")
            if truncated:
                logger.info(f"Episode ended due to truncation (max_steps={self.max_steps} reached)")

        return observation

    @property
    def state(self) -> MountainCarContinuousState:
        """Get current environment state."""
        return self._state

    def _make_observation(self, obs: np.ndarray, reward: float, done: bool) -> MountainCarContinuousObservation:
        """
        Create a MountainCarContinuousObservation from current environment state.

        Args:
            obs: Numpy array observation from gymnasium.
            reward: Reward from the step.
            done: Whether the episode is done.

        Returns:
            MountainCarContinuousObservation for the agent.
        """
        # Convert numpy array to list for JSON serialization
        state_list = obs.tolist() if hasattr(obs, 'tolist') else list(obs)

        # Create observation
        observation = MountainCarContinuousObservation(
            state=state_list,
            done=done,
            reward=reward,
            episode_length=self._state.episode_length,
            total_reward=self._state.total_reward,
            metadata={
                "render_mode": self.render_mode,
                "max_steps": self.max_steps,
                "seed": self.seed,
                "action_bounds": [-1.0, 1.0],
                "position_range": [-1.2, 0.6],
                "velocity_range": [-0.07, 0.07],
                "goal_position": 0.45,
            },
        )

        return observation

    def close(self) -> None:
        """Close the environment and clean up resources."""
        logger.info("Closing MountainCarContinuousEnvironment")
        if hasattr(self.env, 'close'):
            self.env.close()
        logger.info("MountainCarContinuousEnvironment closed successfully")
