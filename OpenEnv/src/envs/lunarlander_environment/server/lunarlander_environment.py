# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
LunarLander Environment Server Implementation.

This module wraps Gymnasium's LunarLander-v3 environment and exposes it
via the OpenEnv Environment interface with comprehensive logging.
"""

import logging
import uuid
from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np

from core.env_server import Environment

from ..models import LunarLanderAction, LunarLanderObservation, LunarLanderState

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class LunarLanderEnvironment(Environment):
    """
    LunarLander Environment wrapper for OpenEnv.

    This environment wraps Gymnasium's LunarLander-v3 environment and provides
    a clean interface for RL training with comprehensive logging.

    The LunarLander task involves landing a spacecraft safely between two flags
    on the moon surface. The lander has a main engine and lateral thrusters.

    Args:
        render_mode: Render mode for visualization ("human", "rgb_array", None).
        max_steps: Maximum steps per episode (default: 1000).
        seed: Random seed for reproducibility.

    Example:
        >>> env = LunarLanderEnvironment(render_mode="human", max_steps=1500)
        >>> obs = env.reset()
        >>> print(obs.state)  # [x_pos, y_pos, x_vel, y_vel, angle, ang_vel, left_contact, right_contact]
        >>> obs = env.step(LunarLanderAction(main_engine=1.0, lateral_engine=0.0))  # Full main thrust
        >>> print(obs.reward, obs.done)
    """

    def __init__(
        self,
        render_mode: Optional[str] = None,
        max_steps: int = 1000,
        seed: Optional[int] = None,
    ):
        """Initialize LunarLander environment."""
        super().__init__()

        self.render_mode = render_mode
        self.max_steps = max(1000, max_steps)  # Ensure minimum 1000 steps
        self.seed = seed

        # Create gymnasium environment
        logger.info(f"Creating LunarLanderContinuous-v3 environment with render_mode={render_mode}, max_steps={self.max_steps}, seed={seed}")
        self.env = gym.make("LunarLanderContinuous-v3", render_mode=render_mode)

        # Set seed if provided
        if seed is not None:
            self.env.reset(seed=seed)
            logger.info(f"Environment seeded with seed={seed}")

        # Initialize state
        self._state = LunarLanderState(
            render_mode=render_mode,
            max_steps=self.max_steps,
            seed=seed,
        )

        logger.info("LunarLanderEnvironment initialized successfully")

    def reset(self) -> LunarLanderObservation:
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
        logger.info(f"Initial state: {obs.tolist()}")

        return observation

    def step(self, action: LunarLanderAction) -> LunarLanderObservation:
        """
        Execute agent's action and return resulting observation.

        Args:
            action: LunarLanderAction containing main_engine and lateral_engine powers.

        Returns:
            Observation after action execution.

        Raises:
            ValueError: If action is not a LunarLanderAction or action values are invalid.
        """
        if not isinstance(action, LunarLanderAction):
            raise ValueError(f"Expected LunarLanderAction, got {type(action)}")

        logger.info(f"Taking action: main_engine={action.main_engine:.3f}, lateral_engine={action.lateral_engine:.3f}")

        # Convert to gymnasium action format [main_engine, lateral_engine]
        # LunarLander-v3 expects actions in [-1, 1] range as numpy array
        gym_action = np.array([action.main_engine, action.lateral_engine], dtype=np.float32)
        gym_action = np.clip(gym_action, -1.0, 1.0)

        # Execute action in gymnasium environment
        obs, reward, terminated, truncated, info = self.env.step(gym_action)

        # Update state tracking
        self._state.step_count += 1
        self._state.episode_length += 1
        self._state.total_reward += reward

        # Determine if episode is done
        done = terminated or truncated

        # Create observation
        observation = self._make_observation(obs, reward, done)

        # Log step details
        x_pos, y_pos, x_vel, y_vel, angle, ang_vel, left_contact, right_contact = obs
        logger.info(f"Step {self._state.step_count}: reward={reward:.1f}, pos=[{x_pos:.2f},{y_pos:.2f}], vel=[{x_vel:.2f},{y_vel:.2f}]")

        if done:
            logger.info(f"Episode {self._state.episode_id} ended - length={self._state.episode_length}, total_reward={self._state.total_reward:.2f}")
            if terminated:
                logger.info("Episode ended due to termination (landed or crashed)")
            if truncated:
                logger.info(f"Episode ended due to truncation (max_steps={self.max_steps} reached)")

        return observation

    @property
    def state(self) -> LunarLanderState:
        """Get current environment state."""
        return self._state

    def _make_observation(self, obs: np.ndarray, reward: float, done: bool) -> LunarLanderObservation:
        """
        Create a LunarLanderObservation from current environment state.

        Args:
            obs: Numpy array observation from gymnasium.
            reward: Reward from the step.
            done: Whether the episode is done.

        Returns:
            LunarLanderObservation for the agent.
        """
        # Convert numpy array to list for JSON serialization
        state_list = obs.tolist() if hasattr(obs, 'tolist') else list(obs)

        # Create observation
        observation = LunarLanderObservation(
            state=state_list,
            done=done,
            reward=reward,
            episode_length=self._state.episode_length,
            total_reward=self._state.total_reward,
            metadata={
                "render_mode": self.render_mode,
                "max_steps": self.max_steps,
                "seed": self.seed,
                "action_meanings": "main_engine [0,1], lateral_engine [-1,1]",
            },
        )

        return observation

    def close(self) -> None:
        """Close the environment and clean up resources."""
        logger.info("Closing LunarLanderEnvironment")
        if hasattr(self.env, 'close'):
            self.env.close()
        logger.info("LunarLanderEnvironment closed successfully")
